from importlib.metadata import version
from pathlib import Path

import numpy as np

from .contracts import digest, file_sha256, normalized, rgb_image

REVISION = "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
WEIGHTS_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
MODEL_ID = "openai/CLIP/ViT-B-32"


class ClipEncoder:
    def __init__(self, model_root="models", device="auto"):
        import clip
        import torch

        self.torch = torch
        self.clip = clip
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device,
                                               download_root=str(Path(model_root).resolve()))
        self.model.eval()
        weights_hash = file_sha256(Path(model_root) / "ViT-B-32.pt")
        if weights_hash != WEIGHTS_SHA256:
            raise ValueError("CLIP weights hash mismatch")
        self.dim = int(self.model.text_projection.shape[1])
        self.metadata = {
            "model_id": MODEL_ID, "model_revision": REVISION,
            "weights_sha256": weights_hash, "embedding_dim": self.dim,
            "preprocess_hash": digest({"official_transform": repr(self.preprocess),
                                       "torchvision": version("torchvision"),
                                       "pillow": version("Pillow"), "clip_revision": REVISION}),
            "dtype": "float32", "normalization": "l2",
            "compute_dtype": str(self.model.dtype),
            "torch_version": torch.__version__, "device": self.device,
        }
        # Hash the installed encoder code too, so a different CLIP install cannot share cache entries.
        self.metadata["clip_code_sha256"] = digest({
            name: file_sha256(Path(clip.__file__).parent / name)
            for name in ("clip.py", "model.py", "simple_tokenizer.py",
                         "bpe_simple_vocab_16e6.txt.gz")})
        self.metadata["encoder_code_sha256"] = digest({
            name: file_sha256(Path(__file__).parent / name)
            for name in ("encoder.py", "contracts.py")})

    def synchronize(self):
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize(self.device)

    def clear_memory(self):
        if self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()

    def encode_image(self, images_rgb):
        if not images_rgb:
            return np.empty((0, self.dim), dtype=np.float32)
        images = self.torch.stack([self.preprocess(rgb_image(im)) for im in images_rgb]).to(self.device)
        with self.torch.inference_mode():
            result = self.model.encode_image(images).float().cpu().numpy()
        return normalized(result, self.dim)

    def encode_text(self, texts):
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        if any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("Text must be a nonempty string")
        tokens = self.clip.tokenize(texts, truncate=False).to(self.device)
        with self.torch.inference_mode():
            result = self.model.encode_text(tokens).float().cpu().numpy()
        return normalized(result, self.dim)
