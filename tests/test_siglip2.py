import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from embedding.config import create_encoder, load_config
from embedding.contracts import file_sha256
from embedding.siglip2 import MODEL_FILES, Siglip2Encoder, model_snapshot


class ModelSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.revision = "a" * 40
        self.folder = self.root / self.revision
        self.folder.mkdir()
        for name in MODEL_FILES:
            (self.folder / name).write_bytes(b"fixture only")
        self.sha = file_sha256(self.folder / "model.safetensors")

    def test_local_snapshot_hash_and_offline_reuse(self):
        self.assertEqual(model_snapshot("google/siglip2-test", self.revision, self.sha, self.root), self.folder)

    def test_corrupt_weights_are_rejected(self):
        (self.folder / "model.safetensors").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            model_snapshot("google/siglip2-test", self.revision, self.sha, self.root)

    def test_unpinned_revision_hash_and_unofficial_model_rejected(self):
        for model, revision, sha in (("other/model", self.revision, self.sha),
                                     ("google/siglip2-test", "main", self.sha),
                                     ("google/siglip2-test", self.revision, "bad")):
            with self.subTest(model=model, revision=revision), self.assertRaises(ValueError):
                model_snapshot(model, revision, sha, self.root)


class EncoderConfigTests(unittest.TestCase):
    def test_clip_and_siglip2_selection(self):
        for name, target, dim in (("clip", "ClipEncoder", 512), ("siglip2", "Siglip2Encoder", 768)):
            config = load_config(f"configs/embedding.{name}.yaml")
            with patch(f"embedding.config.{target}") as constructor:
                constructor.return_value.dim = dim
                self.assertIs(create_encoder(config), constructor.return_value)

    def test_actual_dimension_must_match_config(self):
        config = load_config("configs/embedding.siglip2.yaml")
        with patch("embedding.config.Siglip2Encoder") as constructor:
            constructor.return_value.dim = 512
            with self.assertRaisesRegex(ValueError, "dimension differs"):
                create_encoder(config)

    def test_modified_clip_identity_rejected_before_model_load(self):
        config = load_config("configs/embedding.clip.yaml")
        config["weights_sha256"] = "bad"
        with patch("embedding.config.ClipEncoder") as constructor:
            with self.assertRaisesRegex(ValueError, "model identity"):
                create_encoder(config)
            constructor.assert_not_called()


class Siglip2QueryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch

    def setUp(self):
        self.encoder = Siglip2Encoder.__new__(Siglip2Encoder)
        self.encoder.dim = 768
        self.encoder.device = "cpu"
        self.encoder.dtype = self.torch.float32
        self.encoder.torch = self.torch
        self.encoder.text_max_length = 64
        self.encoder.model = Mock()
        self.encoder.tokenizer = Mock()

    def test_65_tokens_rejected_without_truncating_or_running_model(self):
        self.encoder.tokenizer.return_value = {"input_ids": [list(range(65))]}
        with self.assertRaisesRegex(ValueError, "TEXT_TOO_LONG"):
            self.encoder.encode_text(["Long Query"])
        self.encoder.model.get_text_features.assert_not_called()
        self.encoder.tokenizer.assert_called_once_with(["long query"], padding=False, truncation=False)

    def test_64_tokens_accepted_and_text_lowercased_padded_normalized(self):
        ids = self.torch.ones((1, 64), dtype=self.torch.long)
        self.encoder.tokenizer.side_effect = [{"input_ids": [list(range(64))]}, {"input_ids": ids}]
        self.encoder.model.get_text_features.return_value = self.torch.ones((1, 768))
        vectors = self.encoder.encode_text(["A RED CAR"])
        self.assertEqual(vectors.shape, (1, 768))
        self.assertEqual(vectors.dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(vectors[0])), 1, places=6)
        self.encoder.tokenizer.assert_called_with(["a red car"], padding="max_length", max_length=64,
                                                  truncation=False, return_tensors="pt")

    def test_empty_queries_keep_model_dimension(self):
        self.assertEqual(self.encoder.encode_text([]).shape, (0, 768))
        self.assertEqual(self.encoder.encode_image([]).shape, (0, 768))
        self.encoder.model.get_text_features.assert_not_called()

    def test_blank_or_nonstring_queries_rejected(self):
        for text in ("", "   ", None, 4):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.encoder.encode_text([text])
        self.encoder.tokenizer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
