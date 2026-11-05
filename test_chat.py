"""Tests for server.py /chat endpoint structured output."""

import json
import unittest
from unittest.mock import patch, MagicMock


class TestChatEndpoint(unittest.TestCase):
    """Validate that /chat returns 'answer' always and structured fields
    for OECD analysis paths without breaking backward compatibility."""

    def setUp(self):
        # Import inside method to avoid eager side-effects
        from server import app
        self.app = app
        self.client = app.test_client()

    @patch("agent.Agent.ask_corpus")
    def test_hello_returns_answer_alone(self, mock_ask_corpus):
        """Ordinary 'hello' → answer string only (no structured noise)."""
        mock_ask_corpus.return_value = "Hello! How can I help you?"
        resp = self.client.post(
            "/chat",
            data=json.dumps({"prompt": "hello"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertIn("answer", body)
        self.assertIsInstance(body["answer"], str)

    @patch("agent.Agent.ask_corpus")
    def test_asthma_analysis_returns_structured_fields(self, mock_ask_corpus):
        """Australia asthma analysis must expose deterministic values."""
        structured = {
            "route": "oecd_analysis",
            "answer": "For 2023, 52.0 admissions per 100,000.",
            "request": {
                "dataset_id": "OECD.ELS.HD,DSD_HCQO@DF_PC,2.1",
                "measure_code": "ADMRASTH",
                "filters": {"REF_AREA": "AUS", "FREQ": "A"},
                "start_period": "2016",
                "end_period": "2023",
            },
            "measure": {"code": "ADMRASTH", "label": "Asthma hospitalisations"},
            "filters": {
                "REF_AREA": {"code": "AUS", "label": "Australia"}
            },
            "observations": [
                {"TIME_PERIOD": "2016", "OBS_VALUE": 70.4},
                {"TIME_PERIOD": "2023", "OBS_VALUE": 52.0},
            ],
            "statistics": {
                "first_period": "2016",
                "last_period": "2023",
                "first_value": 70.4,
                "last_value": 52.0,
                "absolute_change": -18.4,
                "percent_change": -26.1363636364,
                "mean": 55.5,
                "year_over_year_changes": [],
            },
            "provenance": {
                "dataset_id": "OECD.ELS.HD,DSD_HCQO@DF_PC,2.1",
                "dataset_title": "Health Data",
            },
            "availability": None,
        }
        mock_ask_corpus.return_value = structured

        # Use the asthma question from analysis.py test block
        resp = self.client.post(
            "/chat",
            data=json.dumps({
                "prompt": (
                    "What was the trend in asthma hospital admission "
                    "rates in Australia from 2016 to 2023?"
                )
            }),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()

        # Always has answer (backward compatibility)
        self.assertIn("answer", body)
        self.assertIsInstance(body["answer"], str)

        # Structured fields present for OECD analysis route
        self.assertEqual(body.get("route"), "oecd_analysis")
        self.assertIsNotNone(body.get("request"))
        self.assertIsNotNone(body.get("measure"))
        self.assertIsNotNone(body.get("filters"))
        self.assertIsNotNone(body.get("observations"))
        self.assertIsNotNone(body.get("statistics"))
        self.assertIsNotNone(body.get("provenance"))

        # Validate deterministic values match analysis.py assertions
        stats = body["statistics"]
        self.assertEqual(stats["first_period"], "2016")
        self.assertEqual(stats["last_period"], "2023")
        self.assertEqual(stats["first_value"], 70.4)
        self.assertEqual(stats["last_value"], 52.0)
        self.assertAlmostEqual(stats["absolute_change"], -18.4, places=5)
        self.assertAlmostEqual(
            stats["percent_change"], -26.1363636364, places=5
        )
        self.assertEqual(stats["mean"], 55.5)


if __name__ == "__main__":
    unittest.main()
