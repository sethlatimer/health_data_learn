import unittest
from pathlib import Path

from config import get_settings


class RuntimeSettingsTests(unittest.TestCase):
    def test_defaults_are_local_and_keep_database_at_project_root(self):
        settings = get_settings(environ={})
        project_root = Path(__file__).resolve().parent

        self.assertEqual(settings.project_root, project_root)
        self.assertEqual(settings.database_path, project_root / "agent.db")
        self.assertEqual(settings.corpus_dir, project_root / "corpus")
        self.assertEqual(settings.allowed_file_root, project_root / "corpus")
        self.assertEqual(settings.host, "127.0.0.1")
        self.assertEqual(settings.port, 5000)
        self.assertFalse(settings.debug)

    def test_settings_can_be_overridden_by_environment(self):
        settings = get_settings(environ={
            "HEALTH_DATA_DB_PATH": "~/tmp/health-test.db",
            "HEALTH_DATA_CORPUS_DIR": "./fixture-corpus",
            "HEALTH_DATA_ALLOWED_FILE_ROOT": "./fixture-corpus/uploads",
            "HEALTH_DATA_DOCUMENT_CACHE_DIR": "./fixture-cache",
            "HEALTH_DATA_HOST": "localhost",
            "HEALTH_DATA_PORT": "8123",
            "HEALTH_DATA_DEBUG": "yes",
        })
        project_root = Path(__file__).resolve().parent

        self.assertEqual(settings.database_path, Path("~/tmp/health-test.db").expanduser().resolve())
        self.assertEqual(settings.corpus_dir, (project_root / "fixture-corpus").resolve())
        self.assertEqual(settings.allowed_file_root, (project_root / "fixture-corpus/uploads").resolve())
        self.assertEqual(settings.document_cache_dir, (project_root / "fixture-cache").resolve())
        self.assertEqual(settings.host, "localhost")
        self.assertEqual(settings.port, 8123)
        self.assertTrue(settings.debug)

    def test_invalid_port_and_boolean_are_rejected(self):
        for environment in (
            {"HEALTH_DATA_PORT": "70000"},
            {"HEALTH_DATA_PORT": "not-a-port"},
            {"HEALTH_DATA_DEBUG": "sometimes"},
        ):
            with self.subTest(environment=environment):
                with self.assertRaises(ValueError):
                    get_settings(environ=environment)


if __name__ == "__main__":
    unittest.main()
