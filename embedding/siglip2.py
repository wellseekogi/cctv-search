"""Pinned SigLIP2 FixRes encoder with local weights and explicit text-length checks."""

import re
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .contracts import digest, file_sha256, normalized, rgb_image

MODEL_FILES = ("config.json", "model.safetensors", "preprocessor_config.json",
               "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json")


def model_snapshot(model_id, revision, weights_sha256, model_root):
    if not model_id.startswith("google/siglip2-") or "naflex" in model_id:
        raise ValueError("Expected an official google/siglip2 fixed-resolution checkpoint")
    if not re.fullmatch(r"[a-f0-9]{40}", revision) or not re.fullmatch(r"[a-f0-9]{64}", weights_sha256):
        raise ValueError("Pin a full model revision and weights SHA256")
    folder = Path(model_root).resolve() / revision
    if not all((folder / name).is_file() for name in MODEL_FILES):
        from huggingface_hub import snapshot_download

        folder.mkdir(parents=True, exist_ok=True)
        snapshot_download(model_id, revision=revision, local_dir=str(folder),
                          cache_dir=str(Path(model_root).resolve() / ".hf-cache"),
                          allow_patterns=list(MODEL_FILES), max_workers=1)
    if file_sha256(folder / "model.safetensors") != weights_sha256:
        raise ValueError("SigLIP2 weights hash mismatch; remove the corrupt local snapshot and download again")
    return folder


class Siglip2Encoder:
    def __init__(self, model_id, revision, weights_sha256, model_root,
                 device="auto", precision="auto", text_max_length=64):
        import torch
        from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

        self.torch = torch
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        if precision == "auto":
            precision = "float16" if self.device.startswith("cuda") else "float32"
        if precision not in ("float16", "float32") or (self.device.startswith("cpu") and precision != "float32"):
            raise ValueError("Use float32 on CPU, or float16/float32 on CUDA")
        self.model_dir = model_snapshot(model_id, revision, weights_sha256, model_root)
        self.dtype = getattr(torch, precision)
        self.model = AutoModel.from_pretrained(str(self.model_dir), local_files_only=True,
                                              trust_remote_code=False, torch_dtype=self.dtype,
                                              use_safetensors=True).to(self.device).eval()
        self.preprocess = AutoImageProcessor.from_pretrained(str(self.model_dir), local_files_only=True,
                                                             use_fast=False, trust_remote_code=False)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True,
                                                       use_fast=True, trust_remote_code=False)
        self.dim = int(self.model.config.text_config.hidden_size)
        self.text_max_length = int(self.model.config.text_config.max_position_embeddings)
        if text_max_length != self.text_max_length:
            raise ValueError("text_max_length must match the model's trained context length")
        processor_files = {name: file_sha256(self.model_dir / name)
                           for name in MODEL_FILES if name != "model.safetensors"}
        text_rules = {"lowercase": True, "padding": "max_length",
                      "max_length": self.text_max_length, "overflow": "reject"}
        self.metadata = {
            "backend": "siglip2", "model_id": model_id, "model_revision": revision,
            "weights_sha256": weights_sha256, "embedding_dim": self.dim,
            "preprocess_hash": digest({"files": processor_files, "text_rules": text_rules,
                                       "transformers": version("transformers"),
                                       "tokenizers": version("tokenizers"), "pillow": version("Pillow")}),
            "dtype": "float32", "normalization": "l2", "compute_dtype": str(self.dtype),
            "torch_version": torch.__version__, "device": self.device,
            "text_max_length": self.text_max_length, "text_rules": text_rules,
            "processor_file_hashes": processor_files,
            "encoder_code_sha256": digest({name: file_sha256(Path(__file__).parent / name)
                                           for name in ("siglip2.py", "contracts.py")}),
        }

    def synchronize(self):
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize(self.device)

    def clear_memory(self):
        if self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()

    def encode_image(self, images_rgb):
        if not images_rgb:
            return np.empty((0, self.dim), dtype=np.float32)
        batch = self.preprocess(images=[rgb_image(image) for image in images_rgb], return_tensors="pt")
        pixels = batch["pixel_values"].to(self.device, dtype=self.dtype)
        with self.torch.inference_mode():
            vectors = self.model.get_image_features(pixel_values=pixels).float().cpu().numpy()
        return normalized(vectors, self.dim)

    def text_token_lengths(self, texts):
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Text must be a nonempty string")
        tokens = self.tokenizer([text.lower() for text in texts], padding=False, truncation=False)
        return [len(ids) for ids in tokens["input_ids"]]

    def encode_text(self, texts):
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        if any(length > self.text_max_length for length in self.text_token_lengths(texts)):
            raise ValueError(f"TEXT_TOO_LONG: maximum {self.text_max_length} tokens including special tokens")
        batch = self.tokenizer([text.lower() for text in texts], padding="max_length",
                               max_length=self.text_max_length, truncation=False, return_tensors="pt")
        with self.torch.inference_mode():
            vectors = self.model.get_text_features(input_ids=batch["input_ids"].to(self.device)).float().cpu().numpy()
        return normalized(vectors, self.dim)
