"""Verify a configured encoder, cold/warm/empty bundles, and English/Korean queries."""

import argparse
import platform
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image

from embedding import build_index, create_encoder, load_config, validate_index
from embedding.contracts import check_vectors, digest, file_sha256, write_json
from fixtures.embedding.generate import generate
from fixtures.embedding.reader import read_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Fresh verification output directory")
    parser.add_argument("--config", default="configs/embedding.yaml")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    generate()
    fixture = Path("fixtures/embedding")
    loading = time.perf_counter()
    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    encoder = create_encoder(config)
    dim = encoder.dim
    model_load_ms = (time.perf_counter() - loading) * 1000
    encoder.synchronize()
    if encoder.device.startswith("cuda"):
        encoder.torch.cuda.reset_peak_memory_stats()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, check=False)
    packages = ["torch", "torchvision", "numpy", "Pillow", "PyYAML"]
    if config.get("backend") == "siglip2":
        packages.extend(["transformers", "huggingface-hub", "tokenizers", "safetensors"])
    else:
        packages.extend(["clip", "ftfy"])
    report = {"checked_at": datetime.now(timezone(timedelta(hours=9))).isoformat(),
              "checker": "Codex", "python": sys.version, "os": platform.platform(),
              "device": encoder.device, "encoder": encoder.metadata,
              "model_load_ms": model_load_ms,
              "packages": {name: version(name) for name in packages},
              "requested_config": config, "requested_config_hash": digest(config),
              "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
              "git_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
              "input_hashes": {name: file_sha256(fixture / name) for name in ("manifest.jsonl", "samples.jsonl")},
              "checks": []}
    statuses = {}
    for name, sample_file in (("cold", "samples.jsonl"), ("warm", "samples.jsonl"), ("empty", "empty_samples.jsonl")):
        status = build_index(fixture / sample_file, fixture / "manifest.jsonl", fixture,
                             root / name, root / "cache", read_frame, encoder, "fixture-rgb-v1",
                             margin=config["crop_margin_ratio"],
                             batch_size=config["batch_size_gpu"] if encoder.device.startswith("cuda") else config["batch_size_cpu"])
        validate_index(root / name, encoder.metadata)
        expected = 0 if name == "empty" else 4
        assert status["succeeded"] == expected and status["failed"] == 0
        values = np.load(root / name / "vectors.npy", allow_pickle=False)
        error = float(np.max(np.abs(np.linalg.norm(values, axis=1) - 1))) if expected else 0.0
        report["checks"].append({"input": name, "expected": {"rows": expected, "dim": dim, "max_norm_error": 0.0001},
                                 "actual": {"shape": list(values.shape), "max_norm_error": error}, "result": "PASS"})
        statuses[name] = status
    assert statuses["warm"]["cache"]["vector_hit"] == 4 and not statuses["warm"]["batches"]
    np.testing.assert_array_equal(np.load(root / "cold/vectors.npy"), np.load(root / "warm/vectors.npy"))
    images = []
    for stamp in (0, 200):
        with Image.open(fixture / f"frame_{stamp}.png") as image:
            images.append(image.copy())
    texts = ["a red car on a road", "a blue car on a road", "도로 위의 빨간 자동차", "도로 위의 파란 자동차"]
    image_vectors = encoder.encode_image(images)
    text_vectors = encoder.encode_text(texts)
    check_vectors(image_vectors, 2, dim)
    check_vectors(text_vectors, len(texts), dim)
    np.save(root / "image_queries.npy", image_vectors, allow_pickle=False)
    np.save(root / "text_queries.npy", text_vectors, allow_pickle=False)
    report["query_example"] = {"texts": texts, "images": ["fixtures/embedding/frame_0.png", "fixtures/embedding/frame_200.png"],
                               "image_shape": list(image_vectors.shape), "text_shape": list(text_vectors.shape),
                               "image_text_scores": (image_vectors @ text_vectors.T).tolist()}
    indexed_scenes = np.load(root / "cold/vectors.npy")[[1, 3]]
    # FP16 kernels vary with batch shape; compare the direction used by retrieval.
    same_image_cosine = np.sum(image_vectors * indexed_scenes, axis=1)
    assert np.all(same_image_cosine >= 0.99999)
    report["query_index_consistency"] = {
        "minimum_cosine": 0.99999, "actual_cosines": same_image_cosine.tolist(),
        "max_coordinate_difference": float(np.max(np.abs(image_vectors - indexed_scenes)))}
    scores = image_vectors @ text_vectors.T
    report["toy_color_top1"] = {
        "english": float(np.mean(np.argmax(scores[:, :2], axis=1) == np.arange(2))),
        "korean": float(np.mean(np.argmax(scores[:, 2:], axis=1) == np.arange(2))),
        "note": "Two synthetic images only; not a CCTV retrieval benchmark"}
    if hasattr(encoder, "text_token_lengths"):
        report["query_token_lengths"] = encoder.text_token_lengths(texts)
    report["checks"].append({"input": "two RGB images and four English/Korean texts",
                             "expected": f"finite normalized [2,{dim}] / [4,{dim}]; same-image cosine >= 0.99999",
                             "actual": f"matched, cosine={same_image_cosine.tolist()}", "result": "PASS"})
    for invalid_text in ([""], ["word " * 100]):
        try:
            encoder.encode_text(invalid_text)
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError("Invalid text accepted")
    assert encoder.encode_text([]).shape == (0, dim)
    assert encoder.encode_image([]).shape == (0, dim)
    report["checks"].append({"input": "empty and token-overflow text; empty query lists", "expected": f"reject invalid text, preserve [0,{dim}]",
                             "actual": "matched", "result": "PASS"})
    report["statuses"] = statuses
    if encoder.device.startswith("cuda"):
        report["gpu"] = {"name": encoder.torch.cuda.get_device_name(),
                         "peak_allocated_bytes": encoder.torch.cuda.max_memory_allocated(),
                         "peak_reserved_bytes": encoder.torch.cuda.max_memory_reserved()}
    report["limitations"] = ["Synthetic RGB fixture, no B decoder or actual A samples supplied",
                              "OOM fallback is covered by injected MemoryError in unit tests"]
    write_json(root / "verification_report.json", report)
    print(f"PASS: {len(report['checks'])} real-model checks; report={root / 'verification_report.json'}")


if __name__ == "__main__":
    main()
