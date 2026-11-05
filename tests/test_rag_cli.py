import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag.__main__ import protect_local_configs, read_key


class LocalKeyTests(unittest.TestCase):
    def test_local_exclusion_is_persistent_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "info/exclude"
            result = subprocess.CompletedProcess([], 0, stdout=str(path), stderr="")
            with patch("rag.__main__.subprocess.run", return_value=result):
                protect_local_configs()
                protect_local_configs()
            self.assertEqual(path.read_text(encoding="utf-8").count("*.local.json"), 1)

    def test_zip_install_without_git_does_not_require_repository(self):
        with patch("rag.__main__.subprocess.run", side_effect=FileNotFoundError):
            protect_local_configs()

    def test_invalid_environment_key_is_rejected(self):
        with patch.dict("os.environ", {"CCTV_RAG_API_KEY": "short"}):
            with self.assertRaises(ValueError):
                read_key(None)


if __name__ == "__main__":
    unittest.main()
