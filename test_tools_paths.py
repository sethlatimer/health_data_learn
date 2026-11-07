import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tools
from config import get_settings


class SafePathTests(unittest.TestCase):
    def test_default_allowed_root_comes_from_runtime_configuration(self):
        self.assertEqual(tools.ALLOWED_ROOT, get_settings().allowed_file_root)

    def test_safe_path_allows_descendants_and_rejects_paths_outside_root(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "allowed"
            root.mkdir()
            inside = root / "notes.txt"
            inside.write_text("fixture", encoding="utf-8")
            outside = Path(temporary_directory) / "outside.txt"
            outside.write_text("fixture", encoding="utf-8")

            with patch.object(tools, "ALLOWED_ROOT", root.resolve()):
                self.assertEqual(tools.safe_path(inside), inside.resolve())
                with self.assertRaises(PermissionError):
                    tools.safe_path(outside)

    def test_safe_path_rejects_symlink_escape(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "allowed"
            root.mkdir()
            outside = Path(temporary_directory) / "outside.txt"
            outside.write_text("fixture", encoding="utf-8")
            link = root / "escape.txt"
            link.symlink_to(outside)

            with patch.object(tools, "ALLOWED_ROOT", root.resolve()):
                with self.assertRaises(PermissionError):
                    tools.safe_path(link)


if __name__ == "__main__":
    unittest.main()
