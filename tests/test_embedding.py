import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from embedding.contracts import (SCHEMA, crop_box, file_sha256,
                                 normalized, read_jsonl, write_jsonl)
from embedding.pipeline import build_index, validate_index


class TestEncoder:
    dim = 8
    device = "cpu"

    def __init__(self):
        self.calls = []
        self.metadata = {"model_id": "test-only", "model_revision": "1", "weights_sha256": "test-only",
                         "embedding_dim": self.dim, "preprocess_hash": "test-only", "dtype": "float32",
                         "normalization": "l2"}

    def synchronize(self):
        pass

    def clear_memory(self):
        pass

    def encode_image(self, images):
        self.calls.append(len(images))
        return normalized([[*np.asarray(im).mean(axis=(0, 1)), *im.size, 1, 2, 3] for im in images], self.dim)


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.png"
        Image.new("RGB", (100, 80), (240, 10, 30)).save(self.source)
        self.video = {"schema_version": SCHEMA, "video_id": "v1", "relative_path": self.source.name,
                      "width": 100, "height": 80, "duration_ms": 1000, "available_start_ms": 100,
                      "available_end_ms": 1000, "source_offset_ms": 100, "split": "dev",
                      "file_sha256": file_sha256(self.source)}
        self.sample = {"schema_version": SCHEMA, "item_id": "scene0", "video_id": "v1",
                       "frame_index": 0, "timestamp_ms": 100, "representation": "scene",
                       "track_uid": None, "bbox": None, "event_ids": ["e1"], "reasons": ["TRACK_START"],
                       "policy": "event", "source_run_id": "a1", "track_source_id": "pred1"}
        self.encoder = TestEncoder()

    def reader(self, video_id, timestamp):
        with Image.open(self.source) as image:
            rgb = np.asarray(image).copy()
        return {"image_rgb": rgb, "actual_timestamp_ms": timestamp,
                "frame_index": (timestamp - 100) // 200, "width": 100, "height": 80,
                "source_offset_ms": 100, "time_source": "pts"}

    def build(self, samples=None, name="out", reader=None, **kwargs):
        write_jsonl(self.root / "manifest.jsonl", [self.video])
        write_jsonl(self.root / "samples.jsonl", [self.sample] if samples is None else samples)
        return build_index(self.root / "samples.jsonl", self.root / "manifest.jsonl", self.root,
                           self.root / name, self.root / "cache", reader or self.reader,
                           self.encoder, "test-reader-v1", **kwargs)

    def object(self, name="object0", box=None):
        return {**self.sample, "item_id": name, "representation": "object", "track_uid": "pred1:v1:1",
                "bbox": [1.2, 2.5, 21.8, 32.1] if box is None else box}

    def test_crop_fractional_floor_ceil_margin(self):
        self.assertEqual(crop_box([10.2, 10.5, 30.8, 40.1], 100, 80), [8, 7, 33, 44])

    def test_crop_all_edges_and_single_pixel(self):
        self.assertEqual(crop_box([0, 0, 100, 80], 100, 80), [0, 0, 100, 80])
        self.assertEqual(crop_box([99, 79, 100, 80], 100, 80), [98, 78, 100, 80])

    def test_crop_rejects_invalid_box(self):
        for box in ([-1, 0, 1, 1], [1, 1, 1, 3], [0, 0, 101, 80], [0, 0, float("nan"), 4]):
            with self.subTest(box=box), self.assertRaises(ValueError):
                crop_box(box, 100, 80)

    def test_sorted_rows_original_bbox_media_and_unit_vectors(self):
        self.assertEqual(self.build([self.sample, self.object()])["succeeded"], 2)
        rows = read_jsonl(self.root / "out/items.jsonl")
        self.assertEqual([r["representation"] for r in rows], ["object", "scene"])
        self.assertEqual(rows[0]["bbox"], [1.2, 2.5, 21.8, 32.1])
        self.assertEqual(rows[0]["crop_bbox"], [0, 0, 24, 36])
        self.assertEqual(rows[0]["full_frame_path"], rows[1]["media_path"])
        self.assertEqual(rows[0]["source_run_id"], "a1")
        self.assertEqual(rows[0]["source_offset_ms"], 100)
        validate_index(self.root / "out", self.encoder.metadata)

    def test_empty_success_shape(self):
        status = self.build([])
        self.assertEqual((status["status"], status["requested"]), ("success", 0))
        self.assertEqual(np.load(self.root / "out/vectors.npy").shape, (0, 8))
        self.assertEqual(read_jsonl(self.root / "out/items.jsonl"), [])
        validate_index(self.root / "out")

    def test_cache_hit_skips_encoding(self):
        self.build()
        self.encoder.calls.clear()
        status = self.build(name="warm")
        self.assertEqual(status["cache"]["vector_hit"], 1)
        self.assertEqual(self.encoder.calls, [])

    def test_model_and_preprocess_invalidate_vectors(self):
        self.build()
        for key in ("model_revision", "weights_sha256", "preprocess_hash"):
            self.encoder.metadata[key] += "changed"
            status = self.build(name=key)
            self.assertEqual(status["cache"]["vector_miss"], 1)

    def test_margin_and_decoder_invalidate_images(self):
        self.build([self.object()])
        status = self.build([self.object()], name="margin", margin=0.2)
        self.assertEqual(status["cache"]["vector_miss"], 1)
        write_jsonl(self.root / "samples.jsonl", [self.object()])
        status = build_index(self.root / "samples.jsonl", self.root / "manifest.jsonl", self.root,
                             self.root / "decoder", self.root / "cache", self.reader, self.encoder, "new-decoder")
        self.assertEqual(status["cache"]["image_miss"], 2)

    def test_corrupt_cache_regenerates(self):
        self.build()
        for folder in ("image", "vector"):
            next((self.root / "cache" / folder).glob("*." + ("png" if folder == "image" else "npy"))).write_bytes(b"bad")
        status = self.build(name="repair")
        self.assertEqual(status["cache"]["vector_miss"], 1)
        validate_index(self.root / "repair")

    def test_retry_transient_decode_and_record_recovery(self):
        calls = []

        def reader(video_id, timestamp):
            calls.append(timestamp)
            if len(calls) == 1:
                raise RuntimeError("DECODE_FAILED")
            return self.reader(video_id, timestamp)

        self.assertEqual(self.build(reader=reader)["status"], "success")
        self.assertEqual(len(calls), 2)
        self.assertTrue(read_jsonl(self.root / "out/errors.jsonl")[0]["recovered"])

    def test_b_returned_decode_error_preserves_code(self):
        status = self.build(reader=lambda *_: {"error": {"code": "FRAME_NOT_FOUND", "message": "missing PTS"}})
        self.assertEqual(status["failed"], 1)
        errors = read_jsonl(self.root / "out/errors.jsonl")
        self.assertEqual([row["code"] for row in errors], ["FRAME_NOT_FOUND", "FRAME_NOT_FOUND"])

    def test_failed_item_skipped_no_row_holes(self):
        status = self.build([self.object(box=[0, 0, 0, 1]), self.sample])
        self.assertEqual((status["succeeded"], status["failed"], status["status"]), (1, 1, "failed"))
        self.assertEqual([r["vector_row"] for r in read_jsonl(self.root / "out/items.jsonl")], [0])
        self.assertEqual(len(read_jsonl(self.root / "out/errors.jsonl")), 2)
        with self.assertRaises(ValueError):
            validate_index(self.root / "out")
        validate_index(self.root / "out", require_success=False)

    def test_frame_mismatch_rejected_even_within_100ms(self):
        def reader(video_id, timestamp):
            return {**self.reader(video_id, timestamp), "actual_timestamp_ms": timestamp + 20}

        self.assertEqual(self.build(reader=reader)["failed"], 1)
        self.assertEqual(read_jsonl(self.root / "out/errors.jsonl")[0]["code"], "FRAME_MISMATCH")

    def test_size_and_offset_mismatches_rejected(self):
        for key, value in (("width", 99), ("source_offset_ms", 0), ("time_source", None)):
            def reader(video_id, timestamp):
                return {**self.reader(video_id, timestamp), key: value}

            self.assertEqual(self.build(name=key, reader=reader)["failed"], 1)

    def test_rgb_contract_rejects_float_image(self):
        def reader(video_id, timestamp):
            return {**self.reader(video_id, timestamp), "image_rgb": np.zeros((80, 100, 3), np.float32)}

        self.assertEqual(self.build(reader=reader)["failed"], 1)

    def test_time_range_is_half_open(self):
        self.sample["timestamp_ms"] = 1000
        self.assertEqual(self.build()["failed"], 1)

    def test_unknown_video_is_item_failure(self):
        self.sample["video_id"] = "missing"
        self.assertEqual(self.build()["failed"], 1)

    def test_oom_sequence_and_recovery(self):
        base = self.encoder.encode_image
        attempts = []

        def encode(images):
            attempts.append(len(images))
            if len(images) > 1:
                raise MemoryError("simulated OOM")
            return base(images)

        self.encoder.encode_image = encode
        samples = [{**self.sample, "item_id": f"s{i}", "timestamp_ms": 100 + i * 200,
                    "frame_index": i} for i in range(4)]
        status = self.build(samples, batch_size=16)
        self.assertEqual(status["succeeded"], 4)
        self.assertEqual([r["requested"] for r in status["batches"][:4]], [16, 8, 4, 1])
        self.assertEqual(attempts[:4], [4, 4, 4, 1])
        self.assertTrue(all(r["recovered"] for r in read_jsonl(self.root / "out/errors.jsonl")))

    def test_oom_singleton_retries_once_at_batch_one(self):
        self.encoder.encode_image = lambda _: (_ for _ in ()).throw(MemoryError("OOM"))
        status = self.build(batch_size=16)
        self.assertEqual([b["requested"] for b in status["batches"]], [16, 8, 4, 1, 1])
        self.assertEqual(status["failed"], 1)

    def test_generic_encoding_error_isolated_and_retried(self):
        base = self.encoder.encode_image
        attempts = []

        def encode(images):
            attempts.append(len(images))
            if len(attempts) <= 2:
                raise RuntimeError("bad batch")
            return base(images)

        self.encoder.encode_image = encode
        self.assertEqual(self.build([self.sample, self.object()])["succeeded"], 2)
        self.assertEqual(attempts, [2, 2, 1, 1])

    def test_invalid_vectors_are_excluded(self):
        for name, value in (("nan", np.full((1, 8), np.nan, np.float32)),
                            ("zero", np.zeros((1, 8), np.float32)),
                            ("shape", np.ones((1, 4), np.float32))):
            self.encoder.encode_image = lambda images: value
            status = self.build(name=name)
            self.assertEqual(status["failed"], 1)
            self.assertEqual(np.load(self.root / name / "vectors.npy").shape, (0, 8))

    def test_published_index_tampering_detected(self):
        self.build()
        with (self.root / "out/items.jsonl").open("a") as stream:
            stream.write("{}\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_index(self.root / "out")

    def test_media_tampering_detected(self):
        self.build()
        next((self.root / "out/media/frames").glob("*.png")).write_bytes(b"bad")
        with self.assertRaisesRegex(ValueError, "Media hash"):
            validate_index(self.root / "out")

    def test_query_encoder_mismatch_detected(self):
        self.build()
        with self.assertRaisesRegex(ValueError, "encoder"):
            validate_index(self.root / "out", {**self.encoder.metadata, "model_revision": "other"})

    def test_query_device_may_differ_without_changing_model(self):
        self.build()
        validate_index(self.root / "out", {**self.encoder.metadata, "device": "cuda"})

    def test_hash_list_cannot_omit_vectors(self):
        self.build()
        import json
        path = self.root / "out/index_manifest.json"
        index = json.loads(path.read_text(encoding="utf-8"))
        del index["file_hashes"]["vectors.npy"]
        path.write_text(json.dumps(index), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash list"):
            validate_index(self.root / "out")

    def test_dev_and_eval_cannot_share_bundle(self):
        write_jsonl(self.root / "manifest.jsonl", [self.video, {**self.video, "video_id": "v2", "split": "eval"}])
        write_jsonl(self.root / "samples.jsonl", [self.sample, {**self.sample, "item_id": "s2", "video_id": "v2"}])
        with self.assertRaisesRegex(ValueError, "separate indexes"):
            build_index(self.root / "samples.jsonl", self.root / "manifest.jsonl", self.root,
                        self.root / "out", self.root / "cache", self.reader, self.encoder, "test-reader-v1")

    def test_source_hash_mismatch_rejected_before_output(self):
        self.video["file_sha256"] = "bad"
        with self.assertRaisesRegex(ValueError, "source hash"):
            self.build()
        self.assertFalse((self.root / "out").exists())

    def test_path_escape_rejected(self):
        self.video["relative_path"] = "../secret.mp4"
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.build()

    def test_duplicate_id_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            self.build([self.sample, self.sample])

    def test_nonobject_jsonl_row_rejected(self):
        (self.root / "bad.jsonl").write_text("[]\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "must be an object"):
            read_jsonl(self.root / "bad.jsonl")

    def test_refuses_existing_output(self):
        self.build()
        original = file_sha256(self.root / "out/items.jsonl")
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual(file_sha256(self.root / "out/items.jsonl"), original)


if __name__ == "__main__":
    unittest.main()
