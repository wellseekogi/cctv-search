import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

SCHEMA = "cctv-3.0"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def file_sha256(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Each JSONL row must be an object")
    return rows


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def within(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Path escapes its registered root")
    return path


def rgb_image(value):
    if isinstance(value, Image.Image):
        if value.mode != "RGB":
            raise ValueError("B must return an RGB image")
        return value
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("RGB must be uint8 [height,width,3]")
    return Image.fromarray(array)


def crop_box(bbox, width, height, margin=0.1):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("bbox must contain four coordinates")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
           not math.isfinite(v) for v in bbox):
        raise ValueError("bbox coordinates must be finite numbers")
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("bbox outside original frame or zero area")
    dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
    return [max(0, math.floor(x1 - dx)), max(0, math.floor(y1 - dy)),
            min(width, math.ceil(x2 + dx)), min(height, math.ceil(y2 + dy))]


def normalized(vectors, dim):
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != dim or not np.isfinite(values).all():
        raise ValueError("Embedding shape or finite-value check failed")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("Invalid or zero embedding norm")
    return (values / norms).astype(np.float32)


def check_vectors(vectors, count, dim):
    if vectors.dtype != np.float32 or vectors.shape != (count, dim):
        raise ValueError("Expected float32 [N,d]")
    if not np.isfinite(vectors).all():
        raise ValueError("Nonfinite vector")
    if np.any(np.abs(np.linalg.norm(vectors, axis=1) - 1) > 0.0001):
        raise ValueError("Normalization error exceeds 0.0001")
