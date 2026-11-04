import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from .contracts import check_vectors, file_sha256, write_json


class Cache:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.counts = {"image_hit": 0, "image_miss": 0, "vector_hit": 0, "vector_miss": 0}

    def get(self, key, kind, dim=None):
        import json

        suffix = ".png" if kind == "image" else ".npy"
        path = self.root / kind / (key + suffix)
        try:
            record = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            if record["status"] != "success" or file_sha256(path) != record["sha256"]:
                raise ValueError("Cache hash mismatch")
            if kind == "image":
                with Image.open(path) as image:
                    if image.mode != "RGB":
                        raise ValueError("Invalid cached color mode")
                    image.verify()
                value = path
            else:
                value = np.load(path, allow_pickle=False)
                check_vectors(value[None, :], 1, dim)
        except (OSError, ValueError, KeyError, EOFError, IndexError, TypeError):
            self.counts[kind + "_miss"] += 1
            return None
        self.counts[kind + "_hit"] += 1
        return value

    def put(self, key, kind, value):
        folder = self.root / kind
        folder.mkdir(exist_ok=True)
        suffix = ".png" if kind == "image" else ".npy"
        path = folder / (key + suffix)
        with tempfile.TemporaryDirectory(dir=folder) as temp:
            candidate = Path(temp) / path.name
            if kind == "image":
                value.save(candidate, format="PNG")
            else:
                with candidate.open("wb") as stream:
                    np.save(stream, value, allow_pickle=False)
            metadata = candidate.with_suffix(".json")
            write_json(metadata, {"status": "success", "sha256": file_sha256(candidate)})
            os.replace(candidate, path)
            os.replace(metadata, path.with_suffix(".json"))
        return path
