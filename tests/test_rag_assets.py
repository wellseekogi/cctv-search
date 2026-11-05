import copy
import hashlib
import io
import json
import stat
import tempfile
import unittest
import urllib.error
import zipfile
from pathlib import Path
from unittest.mock import patch

from rag.assets import ensure_assets


def runtime_zip(extra=None):
    value = io.BytesIO()
    with zipfile.ZipFile(value, "w") as archive:
        archive.writestr("bin/llama-server.exe", b"fixture executable")
        archive.writestr("bin/ggml.dll", b"fixture library")
        if extra is not None:
            archive.writestr(extra, b"unsafe fixture")
    return value.getvalue()


class AssetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.asset_root = self.root / "assets"
        self.model = b"small fixture model"
        self.zip = runtime_zip()
        self.manifest = {
            "model": {"id": "fixture/model", "revision": "a" * 40, "filename": "model.gguf",
                      "url": "https://example.test/model", "bytes": len(self.model),
                      "sha256": hashlib.sha256(self.model).hexdigest()},
            "runtime": {"tag": "b-fixture", "filename": "runtime.zip", "url": "https://example.test/runtime",
                        "bytes": len(self.zip), "sha256": hashlib.sha256(self.zip).hexdigest()}}
        self.manifest_path = self.root / "manifest.json"

    def download(self, url, timeout):
        self.assertEqual(timeout, 60)
        return io.BytesIO(self.model if url.endswith("/model") else self.zip)

    def ensure(self):
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        return ensure_assets(self.manifest_path, self.asset_root)

    def set_runtime(self, payload):
        self.zip = payload
        self.manifest["runtime"]["bytes"] = len(payload)
        self.manifest["runtime"]["sha256"] = hashlib.sha256(payload).hexdigest()

    def test_download_extract_and_offline_cache(self):
        with patch("rag.assets.urllib.request.urlopen", side_effect=self.download) as network:
            result = self.ensure()
            self.assertEqual(network.call_count, 2)
        self.assertEqual(Path(result["model_path"]).read_bytes(), self.model)
        self.assertEqual(Path(result["server_path"]).read_bytes(), b"fixture executable")
        self.assertTrue(Path(result["server_path"]).is_absolute())
        with patch("rag.assets.urllib.request.urlopen") as network:
            self.assertEqual(self.ensure(), result)
            network.assert_not_called()

    def test_hash_mismatch_leaves_no_model_or_partial_file(self):
        self.manifest["model"]["sha256"] = "0" * 64
        with patch("rag.assets.urllib.request.urlopen", side_effect=self.download):
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                self.ensure()
        self.assertFalse((self.asset_root / "model/model.gguf").exists())
        self.assertFalse(list(self.asset_root.rglob("*.part")))

    def test_truncated_and_oversized_download_rejected(self):
        for difference in (-1, 1):
            with self.subTest(difference=difference):
                self.manifest["model"]["bytes"] = len(self.model) + difference
                with patch("rag.assets.urllib.request.urlopen", side_effect=self.download):
                    with self.assertRaises(ValueError):
                        self.ensure()
                self.assertFalse(list(self.asset_root.rglob("*.part")))

    def test_failed_download_preserves_existing_file(self):
        target = self.asset_root / "model/model.gguf"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"old local file")
        failure = urllib.error.HTTPError("https://example.test/model", 503, "unavailable", {}, None)
        with patch("rag.assets.urllib.request.urlopen", side_effect=failure):
            with self.assertRaises(urllib.error.HTTPError):
                self.ensure()
        self.assertEqual(target.read_bytes(), b"old local file")
        self.assertFalse(list(self.asset_root.rglob("*.part")))

    def test_corrupt_model_redownloaded_and_runtime_repaired_without_network(self):
        with patch("rag.assets.urllib.request.urlopen", side_effect=self.download):
            result = self.ensure()
        Path(result["model_path"]).write_bytes(b"bad model")
        Path(result["server_path"]).write_bytes(b"bad runtime")
        Path(result["server_path"]).with_name("ggml.dll").unlink()
        with patch("rag.assets.urllib.request.urlopen", side_effect=self.download) as network:
            self.ensure()
            self.assertEqual(network.call_count, 1)
        self.assertEqual(Path(result["server_path"]).read_bytes(), b"fixture executable")
        self.assertEqual(Path(result["server_path"]).with_name("ggml.dll").read_bytes(), b"fixture library")

    def test_bad_runtime_zip_hash_rejected(self):
        self.manifest["runtime"]["sha256"] = "0" * 64
        with patch("rag.assets.urllib.request.urlopen", side_effect=self.download):
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                self.ensure()
        self.assertFalse((self.asset_root / "runtime").exists())

    def test_url_size_hash_and_filename_inputs_checked_before_network(self):
        original = copy.deepcopy(self.manifest)
        for field, value in (("url", "http://example.test/model"), ("url", "file:///tmp/model"),
                             ("filename", "../escape.gguf"), ("filename", "C:\\escape.gguf"),
                             ("filename", "NUL.gguf"), ("filename", "NUL.extra.gguf"),
                             ("sha256", "short"), ("bytes", 0), ("bytes", True)):
            self.manifest = copy.deepcopy(original)
            self.manifest["model"][field] = value
            with self.subTest(field=field, value=value), patch("rag.assets.urllib.request.urlopen") as network:
                with self.assertRaises(ValueError):
                    self.ensure()
                network.assert_not_called()

    def test_unsafe_zip_paths_rejected_before_extracting(self):
        for name in ("../escape.txt", "/absolute.txt", "C:/drive.txt", "a\\..\\escape.txt", "a:stream"):
            self.set_runtime(runtime_zip(name))
            with self.subTest(name=name), patch("rag.assets.urllib.request.urlopen", side_effect=self.download):
                with self.assertRaisesRegex(ValueError, "Unsafe"):
                    self.ensure()
            self.assertFalse((self.asset_root / "runtime/b-fixture/bin/llama-server.exe").exists())
            self.assertFalse((self.root / "escape.txt").exists())

    def test_zip_symlink_rejected(self):
        link = zipfile.ZipInfo("symlink")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        self.set_runtime(runtime_zip(link))
        with patch("rag.assets.urllib.request.urlopen", side_effect=self.download):
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                self.ensure()

    def test_runtime_tag_escape_rejected(self):
        self.manifest["runtime"]["tag"] = "../escape"
        with patch("rag.assets.urllib.request.urlopen") as network:
            with self.assertRaisesRegex(ValueError, "tag"):
                self.ensure()
            network.assert_not_called()

    def test_case_colliding_zip_members_rejected(self):
        self.set_runtime(runtime_zip("BIN/LLAMA-SERVER.EXE"))
        with patch("rag.assets.urllib.request.urlopen", side_effect=self.download):
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                self.ensure()

    def test_runtime_without_server_rejected(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("readme.txt", b"not a runtime")
        self.set_runtime(buffer.getvalue())
        with patch("rag.assets.urllib.request.urlopen", side_effect=self.download):
            with self.assertRaisesRegex(ValueError, "exactly one"):
                self.ensure()


if __name__ == "__main__":
    unittest.main()
