from pathlib import Path

import yaml

from .encoder import ClipEncoder, MODEL_ID, REVISION, WEIGHTS_SHA256
from .siglip2 import Siglip2Encoder


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["embedding"]
    if config["normalize"] is not True:
        raise ValueError("L2 normalization is required")
    if config.get("backend", "clip") not in ("clip", "siglip2"):
        raise ValueError("Unsupported encoder backend")
    return config


def create_encoder(config):
    if config.get("backend", "clip") == "clip":
        if (config["model_id"], config["model_revision"], config["weights_sha256"], config["embedding_dim"]) != (
                MODEL_ID, REVISION, WEIGHTS_SHA256, 512):
            raise ValueError("Unsupported CLIP model identity")
        encoder = ClipEncoder(config["model_root"], config["device"])
    elif config["backend"] == "siglip2":
        encoder = Siglip2Encoder(config["model_id"], config["model_revision"], config["weights_sha256"],
                                config["model_root"], config["device"], config.get("precision", "auto"),
                                config["text_max_length"])
    else:
        raise ValueError("Unsupported encoder backend")
    if encoder.dim != config["embedding_dim"]:
        raise ValueError("Configured dimension differs from actual model dimension")
    return encoder
