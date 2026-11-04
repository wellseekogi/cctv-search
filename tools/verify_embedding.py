"""Run actual CLIP, cold/warm/empty indexes, and D's two-image/two-text example."""

import argparse
import platform
import sys
import time
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image

from embedding import ClipEncoder, build_index, validate_index
from embedding.contracts import check_vectors, file_sha256, write_json
from fixtures.embedding.generate import generate
from fixtures.embedding.reader import read_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Fresh verification output directory")
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    generate()
    fixture = Path("fixtures/embedding")
    loading = time.perf_counter()
    encoder = ClipEncoder()
    model_load_ms = (time.perf_counter() - loading) * 1000
    encoder.synchronize()
    if encoder.device.startswith("cuda"):
        encoder.torch.cuda.reset_peak_memory_stats()
    report = {"checked_at": datetime.now(timezone(timedelta(hours=9))).isoformat(),
              "checker": "Codex", "python": sys.version, "os": platform.platform(),
              "device": encoder.device, "encoder": encoder.metadata,
              "model_load_ms": model_load_ms,
              "packages": {name: version(name) for name in
                           ("torch", "torchvision", "numpy", "Pillow", "clip", "ftfy", "PyYAML")},
              "input_hashes": {name: file_sha256(fixture / name) for name in ("manifest.jsonl", "samples.jsonl")},
              "checks": []}
    statuses = {}
    for name, sample_file in (("cold", "samples.jsonl"), ("warm", "samples.jsonl"), ("empty", "empty_samples.jsonl")):
        status = build_index(fixture / sample_file, fixture / "manifest.jsonl", fixture,
                             root / name, root / "cache", read_frame, encoder, "fixture-rgb-v1")
        validate_index(root / name, encoder.metadata)
        expected = 0 if name == "empty" else 4
        assert status["succeeded"] == expected and status["failed"] == 0
        values = np.load(root / name / "vectors.npy", allow_pickle=False)
        error = float(np.max(np.abs(np.linalg.norm(values, axis=1) - 1))) if expected else 0.0
        report["checks"].append({"input": name, "expected": {"rows": expected, "dim": 512, "max_norm_error": 0.0001},
                                 "actual": {"shape": list(values.shape), "max_norm_error": error}, "result": "PASS"})
        statuses[name] = status
    assert statuses["warm"]["cache"]["vector_hit"] == 4 and not statuses["warm"]["batches"]
    np.testing.assert_array_equal(np.load(root / "cold/vectors.npy"), np.load(root / "warm/vectors.npy"))
    images = []
    for stamp in (0, 200):
        with Image.open(fixture / f"frame_{stamp}.png") as image:
            images.append(image.copy())
    texts = ["a red car on a road", "a blue car on a road"]
    image_vectors = encoder.encode_image(images)
    text_vectors = encoder.encode_text(texts)
    check_vectors(image_vectors, 2, 512)
    check_vectors(text_vectors, 2, 512)
    np.save(root / "image_queries.npy", image_vectors, allow_pickle=False)
    np.save(root / "text_queries.npy", text_vectors, allow_pickle=False)
    report["query_example"] = {"texts": texts, "images": ["fixtures/embedding/frame_0.png", "fixtures/embedding/frame_200.png"],
                               "image_shape": list(image_vectors.shape), "text_shape": list(text_vectors.shape),
                               "image_text_scores": (image_vectors @ text_vectors.T).tolist()}
    # Input order is object, scene at each timestamp. Query and indexed scene must agree.
    np.testing.assert_allclose(image_vectors, np.load(root / "cold/vectors.npy")[[1, 3]], atol=0.0001)
    report["checks"].append({"input": "two RGB images and two English texts", "expected": "finite normalized [2,512]; scene/query equality",
                             "actual": "both [2,512], equality within 0.0001", "result": "PASS"})
    for invalid_text in ([""], ["word " * 100]):
        try:
            encoder.encode_text(invalid_text)
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError("Invalid text accepted")
    assert encoder.encode_text([]).shape == (0, 512)
    assert encoder.encode_image([]).shape == (0, 512)
    report["checks"].append({"input": "empty and token-overflow text; empty query lists", "expected": "reject invalid text, preserve [0,512]",
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
