import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import database


class DatabasePathTests(unittest.TestCase):
    def test_initialize_finds_schema_when_called_outside_project(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "test.db"
            outside_directory = Path(temporary_directory) / "outside"
            outside_directory.mkdir()
            old_cwd = Path.cwd()
            try:
                os.chdir(outside_directory)
                with patch.object(database, "DB", str(database_path)):
                    database.initialize()
                    connection = sqlite3.connect(database_path)
                    try:
                        tables = {
                            row[0]
                            for row in connection.execute(
                                "SELECT name FROM sqlite_master WHERE type='table'"
                            )
                        }
                    finally:
                        connection.close()
            finally:
                os.chdir(old_cwd)

        self.assertIn("dataset_structures", tables)
        self.assertIn("literature_records", tables)


if __name__ == "__main__":
    unittest.main()
