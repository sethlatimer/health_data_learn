import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import server


class ServerSettingsTests(unittest.TestCase):
    @patch("server.app.run")
    @patch(
        "server.get_settings",
        return_value=SimpleNamespace(
            host="127.0.0.1",
            port=8123,
            debug=False,
        ),
    )
    def test_main_runs_with_configured_safe_server_settings(
        self,
        mocked_settings,
        mocked_run,
    ):
        server.main()

        mocked_settings.assert_called_once_with()
        mocked_run.assert_called_once_with(
            host="127.0.0.1",
            port=8123,
            debug=False,
        )

    @patch.object(server.agent, "index_folder", return_value={"indexed": []})
    @patch(
        "server.get_settings",
        return_value=SimpleNamespace(
            corpus_dir=Path("/tmp/health-data-corpus"),
        ),
    )
    def test_index_endpoint_uses_configured_corpus_directory(
        self,
        mocked_settings,
        mocked_index,
    ):
        response = server.app.test_client().post("/index")

        self.assertEqual(response.status_code, 200)
        mocked_settings.assert_called_once_with()
        mocked_index.assert_called_once_with("/tmp/health-data-corpus")


if __name__ == "__main__":
    unittest.main()
