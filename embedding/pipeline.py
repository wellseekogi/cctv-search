import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .cache import Cache
from .contracts import (SCHEMA, check_vectors, crop_box, digest, file_sha256,
                        read_jsonl, rgb_image, within, write_json, write_jsonl)


class ItemError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def integer(value, name):
    if type(value) is not int or value < 0:
        raise ItemError("INVALID_INPUT", name + " must be a nonnegative integer")
    return value


def validate_sample(sample, video):
    required = ("item_id", "video_id", "frame_index", "timestamp_ms", "representation",
                "track_uid", "bbox", "event_ids", "reasons", "policy", "source_run_id", "track_source_id")
    if any(key not in sample for key in required) or sample.get("schema_version") != SCHEMA:
        raise ItemError("INVALID_INPUT", "Missing sample fields or incompatible schema")
    for key in ("item_id", "video_id", "policy", "source_run_id", "track_source_id"):
        if not isinstance(sample[key], str) or not sample[key]:
            raise ItemError("INVALID_INPUT", key + " must be a nonempty string")
    for key in ("event_ids", "reasons"):
        if not isinstance(sample[key], list) or any(not isinstance(x, str) for x in sample[key]):
            raise ItemError("INVALID_INPUT", key + " must be a string list")
    integer(sample["frame_index"], "frame_index")
    stamp = integer(sample["timestamp_ms"], "timestamp_ms")
    if not video["available_start_ms"] <= stamp < video["available_end_ms"]:
        raise ItemError("OUT_OF_RANGE", "Sample timestamp outside available original interval")
    if sample["representation"] == "scene":
        if sample["track_uid"] is not None or sample["bbox"] is not None:
            raise ItemError("INVALID_INPUT", "Scene must have null track_uid and bbox")
    elif sample["representation"] == "object":
        if not isinstance(sample["track_uid"], str) or not sample["track_uid"]:
            raise ItemError("INVALID_INPUT", "Object requires track_uid")
        crop_box(sample["bbox"], video["width"], video["height"])
    else:
        raise ItemError("INVALID_INPUT", "Unknown representation")


def error_record(errors, item_id, stage, exc, attempt, recovered=False):
    errors.append({"schema_version": SCHEMA, "error_id": f"embedding-{len(errors):06d}",
                   "stage": stage, "item_id": item_id, "attempt": attempt,
                   "code": getattr(exc, "code", type(exc).__name__),
                   "message": str(exc), "recovered": recovered})


def retry(operation, item_id, stage, errors):
    start = len(errors)
    for attempt in (1, 2):
        try:
            result = operation()
            for row in errors[start:]:
                row["recovered"] = True
            return result
        except Exception as exc:
            error_record(errors, item_id, stage, exc, attempt)
    return None


def validate_index(run_dir, expected_metadata=None, require_success=True):
    root = Path(run_dir)
    index = json.loads((root / "index_manifest.json").read_text(encoding="utf-8"))
    if index["schema_version"] != SCHEMA or (require_success and index["status"] != "success"):
        raise ValueError("Index schema or stage status mismatch")
    if index["dtype"] != "float32" or index["normalization"] != "l2":
        raise ValueError("Index dtype or normalization mismatch")
    if index["encoder"]["embedding_dim"] != index["embedding_dim"]:
        raise ValueError("Index dimension metadata mismatch")
    if expected_metadata is not None:
        identity = ("model_id", "model_revision", "weights_sha256", "embedding_dim",
                    "preprocess_hash", "dtype", "normalization")
        if any(index["encoder"].get(k) != expected_metadata.get(k) for k in identity):
            raise ValueError("Query encoder does not match index encoder")
        for key in ("clip_code_sha256", "encoder_code_sha256"):
            if index["encoder"].get(key) != expected_metadata.get(key):
                raise ValueError("Query encoder implementation mismatch")
    if set(index["file_hashes"]) != {"items.jsonl", "vectors.npy", "embedding_status.json",
                                    "errors.jsonl", "embedding_config.json"}:
        raise ValueError("Incomplete index file hash list")
    for relative, expected in index["file_hashes"].items():
        if file_sha256(within(root, relative)) != expected:
            raise ValueError("Index file hash mismatch: " + relative)
    items = read_jsonl(root / "items.jsonl")
    vectors = np.load(root / "vectors.npy", allow_pickle=False)
    check_vectors(vectors, len(items), index["embedding_dim"])
    if index["row_count"] != len(items) or [r["vector_row"] for r in items] != list(range(len(items))):
        raise ValueError("Vector row correspondence failed")
    if len({r["item_id"] for r in items}) != len(items):
        raise ValueError("Duplicate output item_id")
    for row in items:
        for name in ("media_path", "full_frame_path"):
            path = within(root, row[name])
            if file_sha256(path) != row[name.replace("_path", "_sha256")]:
                raise ValueError("Media hash mismatch")
            with Image.open(path) as image:
                image.verify()
        if any(row[k] != index["encoder"][k] for k in
               ("model_id", "model_revision", "embedding_dim", "preprocess_hash")):
            raise ValueError("Per-item model metadata mismatch")
    return index


def build_index(samples_path, manifest_path, data_root, output_dir, cache_dir,
                read_frame, encoder, decoder_id, margin=0.10, batch_size=None):
    if not decoder_id or not math.isfinite(margin) or margin < 0:
        raise ValueError("Decoder identity and a finite nonnegative margin are required")
    if batch_size is None:
        batch_size = 16 if encoder.device.startswith("cuda") else 4
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Batch size must be a positive integer")
    if encoder.metadata["embedding_dim"] != encoder.dim:
        raise ValueError("Encoder dimension metadata mismatch")
    videos = {}
    for video in read_jsonl(manifest_path):
        if video.get("schema_version") != SCHEMA or video["video_id"] in videos:
            raise ValueError("Manifest schema mismatch or duplicate video_id")
        for key in ("width", "height", "duration_ms", "available_start_ms", "available_end_ms", "source_offset_ms"):
            integer(video[key], key)
        if not (video["width"] > 0 and video["height"] > 0 and
                video["source_offset_ms"] <= video["available_start_ms"] <
                video["available_end_ms"] <= video["duration_ms"]):
            raise ValueError("Invalid manifest dimensions or original interval")
        if video["split"] not in ("dev", "eval"):
            raise ValueError("Unknown manifest split")
        source_path = within(data_root, video["relative_path"])
        if file_sha256(source_path) != video["file_sha256"]:
            raise ValueError("Original source hash mismatch: " + video["video_id"])
        videos[video["video_id"]] = video
    samples = read_jsonl(samples_path)
    ids = [row.get("item_id") for row in samples]
    if any(not isinstance(key, str) or not key for key in ids) or len(set(ids)) != len(ids):
        raise ValueError("Sample item_id must be nonempty and unique")
    samples.sort(key=lambda r: (str(r.get("video_id", "")),
                               r.get("timestamp_ms") if type(r.get("timestamp_ms")) is int else -1,
                               str(r.get("representation", "")), str(r.get("track_uid") or ""), r["item_id"]))
    if len({videos[r["video_id"]]["split"] for r in samples if r.get("video_id") in videos}) > 1:
        raise ValueError("Build separate indexes for dev and eval")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    targets = ("items.jsonl", "vectors.npy", "index_manifest.json", "media",
               "embedding_status.json", "errors.jsonl", "embedding_config.json")
    if any((output / name).exists() for name in targets):
        raise FileExistsError("Embedding outputs already exist; use a fresh output directory")
    cache = Cache(cache_dir)
    errors, successful, rows, batches = [], {}, [], []
    config = {"schema_version": SCHEMA, "encoder": encoder.metadata,
              "crop_margin_ratio": margin, "initial_batch_size": batch_size,
              "decoder_id": decoder_id, "color_rule": "RGB_uint8_PNG_lossless",
              "dtype": "float32", "normalization": "l2"}
    config_hash = digest(config)
    encoder.synchronize()
    started = time.perf_counter()
    read_crop_ms, encode_ms = 0.0, 0.0
    last_frame = {}

    with tempfile.TemporaryDirectory(prefix=".embedding-", dir=output) as temp:
        stage = Path(temp)

        def image_file(key, image_factory, subdir):
            cached = cache.get(key, "image")
            if cached is None:
                cached = cache.put(key, "image", image_factory())
            relative = f"media/{subdir}/{key}.png"
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copyfile(cached, destination)
            return relative, file_sha256(destination)

        def prepare(sample):
            if sample.get("video_id") not in videos:
                raise ItemError("UNKNOWN_VIDEO", "Video not registered in manifest")
            video = videos[sample["video_id"]]
            validate_sample(sample, video)
            frame_key = (sample["video_id"], sample["frame_index"], sample["timestamp_ms"])
            if last_frame.get("key") != frame_key:
                frame = read_frame(sample["video_id"], sample["timestamp_ms"])
                if not isinstance(frame, dict):
                    raise ItemError("INVALID_FRAME", "B must return a frame dictionary")
                if frame.get("error"):
                    failure = frame["error"]
                    if not isinstance(failure, dict):
                        raise ItemError("INVALID_FRAME", "B error must contain code and message")
                    raise ItemError(failure.get("code", "DECODE_FAILED"), failure.get("message", "B frame read failed"))
                image = rgb_image(frame["image_rgb"])
                actual = integer(frame["actual_timestamp_ms"], "actual_timestamp_ms")
                for name in ("frame_index", "width", "height", "source_offset_ms"):
                    integer(frame[name], name)
                # Samples already refer to actual frames. A nearby frame would move B's bbox.
                if actual != sample["timestamp_ms"] or frame["frame_index"] != sample["frame_index"]:
                    raise ItemError("FRAME_MISMATCH", f"Selected vs decoded frame mismatch, delta_ms={actual - sample['timestamp_ms']}")
                if image.size != (video["width"], video["height"]) or (frame["width"], frame["height"]) != image.size:
                    raise ItemError("FRAME_SIZE_MISMATCH", "B dimensions differ from manifest")
                if frame["source_offset_ms"] != video["source_offset_ms"] or not frame["time_source"]:
                    raise ItemError("TIME_METADATA_MISMATCH", "B time metadata differs from manifest")
                last_frame.update(key=frame_key, frame=frame, image=image)
            frame, image = last_frame["frame"], last_frame["image"]
            identity = {"source_sha256": video["file_sha256"],
                        "actual_timestamp_ms": frame["actual_timestamp_ms"],
                        "frame_index": frame["frame_index"], "decoder": decoder_id,
                        "color_rule": config["color_rule"], "bbox": None, "margin": 0}
            full_key = digest(identity)
            full_path, full_hash = image_file(full_key, lambda: image, "frames")
            box = None
            media_path, media_hash, media_key = full_path, full_hash, full_key
            if sample["representation"] == "object":
                box = crop_box(sample["bbox"], *image.size, margin)
                media_key = digest({**identity, "bbox": sample["bbox"], "margin": margin})
                media_path, media_hash = image_file(media_key, lambda: image.crop(box), "crops")
            vector_key = digest({"image_cache_key": media_key, "image_sha256": media_hash,
                                 "encoder": encoder.metadata, "schema_version": SCHEMA})
            row = {**sample, "split": video["split"], "width": video["width"], "height": video["height"],
                   "requested_timestamp_ms": sample["timestamp_ms"],
                   "actual_timestamp_ms": frame["actual_timestamp_ms"],
                   "source_offset_ms": frame["source_offset_ms"], "time_source": frame["time_source"],
                   "crop_bbox": box, "media_path": media_path, "full_frame_path": full_path,
                   "media_sha256": media_hash, "full_frame_sha256": full_hash,
                   "image_cache_key": media_key, "vector_cache_key": vector_key,
                   "encoding_status": "success", **{k: encoder.metadata[k] for k in
                   ("model_id", "model_revision", "embedding_dim", "preprocess_hash")}}
            return row

        for sample in samples:
            phase_start = time.perf_counter()
            row = retry(lambda: prepare(sample), sample["item_id"], "read_crop", errors)
            read_crop_ms += (time.perf_counter() - phase_start) * 1000
            if row is not None:
                rows.append(row)
                vector = cache.get(row["vector_cache_key"], "vector", encoder.dim)
                if vector is not None:
                    successful[row["item_id"]] = vector

        pending = [r for r in rows if r["item_id"] not in successful]
        position, active_batch = 0, batch_size
        while position < len(pending):
            chunk = pending[position:position + active_batch]
            encoded = None
            resized = False
            for attempt in (1, 2):
                try:
                    images = []
                    for row in chunk:
                        with Image.open(stage / row["media_path"]) as image:
                            images.append(image.copy())
                    encoder.synchronize()
                    encode_start = time.perf_counter()
                    encoded = encoder.encode_image(images)
                    encoder.synchronize()
                    encode_ms += (time.perf_counter() - encode_start) * 1000
                    check_vectors(encoded, len(chunk), encoder.dim)
                    batches.append({"requested": active_batch, "actual": len(chunk), "attempt": attempt, "status": "success"})
                    break
                except Exception as exc:
                    encoded = None
                    oom = isinstance(exc, MemoryError) or type(exc).__name__ in ("OutOfMemoryError", "CUDAOutOfMemoryError")
                    batches.append({"requested": active_batch, "actual": len(chunk), "attempt": attempt,
                                    "status": "oom" if oom else "failed"})
                    for row in chunk:
                        error_record(errors, row["item_id"], "encode", exc, attempt)
                    if oom:
                        encoder.clear_memory()
                        if active_batch > 1:
                            active_batch = next((n for n in (8, 4, 1) if n < active_batch), 1)
                            resized = True
                            break
                    if len(chunk) > 1 and attempt == 2:
                        # Isolate a bad item instead of dropping every member of its batch.
                        active_batch = 1
                        resized = True
                        break
            if encoded is None:
                if resized:
                    continue
                position += 1
                continue
            for row, vector in zip(chunk, encoded):
                cache.put(row["vector_cache_key"], "vector", vector)
                successful[row["item_id"]] = vector
            position += len(chunk)

        items = []
        for row in rows:
            if row["item_id"] in successful:
                items.append({**row, "vector_row": len(items)})
        for error in errors:
            error["run_id"] = output.name
            error["config_hash"] = config_hash
            if error["item_id"] in successful:
                error["recovered"] = True
        vectors = np.stack([successful[r["item_id"]] for r in items]) if items else np.empty((0, encoder.dim), np.float32)
        check_vectors(vectors, len(items), encoder.dim)
        encoder.synchronize()
        elapsed = (time.perf_counter() - started) * 1000
        status = {"schema_version": SCHEMA, "stage": "embedding", "run_id": output.name,
                  "status": "failed" if len(items) != len(samples) else "success",
                  "requested": len(samples), "succeeded": len(items), "failed": len(samples) - len(items),
                  "cache": cache.counts, "batches": batches, "wall_ms": elapsed,
                  "read_crop_ms": read_crop_ms, "encode_success_ms": encode_ms,
                  "config_hash": config_hash}
        write_jsonl(stage / "items.jsonl", items)
        np.save(stage / "vectors.npy", vectors, allow_pickle=False)
        write_jsonl(stage / "errors.jsonl", errors)
        write_json(stage / "embedding_status.json", status)
        write_json(stage / "embedding_config.json", config)
        used_videos = sorted({r["video_id"] for r in samples if r.get("video_id") in videos})
        index = {"schema_version": SCHEMA, "status": status["status"], "stage": "embedding", "run_id": output.name,
                 "row_count": len(items), "embedding_dim": encoder.dim, "dtype": "float32",
                 "normalization": "l2", "encoder": encoder.metadata, "config_hash": config_hash,
                 "inputs": {"samples_sha256": file_sha256(samples_path), "manifest_sha256": file_sha256(manifest_path)},
                 "splits": sorted({videos[v]["split"] for v in used_videos}),
                 "track_source_ids": sorted({r["track_source_id"] for r in items}),
                 "source_run_ids": sorted({r["source_run_id"] for r in items}),
                 "sources": {v: videos[v]["file_sha256"] for v in used_videos},
                 "file_hashes": {name: file_sha256(stage / name) for name in targets
                                 if name not in ("media", "index_manifest.json")}}
        write_json(stage / "index_manifest.json", index)
        validate_index(stage, encoder.metadata, require_success=False)
        for name in targets:
            if name != "index_manifest.json" and (stage / name).exists():
                os.replace(stage / name, output / name)
        # This file is the commit marker. Consumers never open an unpublished bundle.
        os.replace(stage / "index_manifest.json", output / "index_manifest.json")
    return status
