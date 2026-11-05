import tempfile
import unittest
from datetime import date
from unittest.mock import Mock, call, patch

import requests
import api_data
import database
import embedding
from agent import Agent
from api_data import build_dataset_concepts


class OECDCatalogRuntimeTests(unittest.TestCase):
    def test_inpatient_per_capita_query_uses_discharge_rate_and_latest_data(self):
        dataset_id = (
            "TEST.AGENCY,DSD_HOSPITAL@DF_HOSPITAL_DISCHARGE,1.0"
        )
        structure = {
            "dataset_id": dataset_id,
            "dataflow": {
                "id": "DF_HOSPITAL_DISCHARGE",
                "agency_id": "TEST.AGENCY",
                "version": "1.0",
                "name": "Hospital discharges",
                "description": "Hospital discharges by diagnostic categories",
            },
            "data_structure": {
                "id": "DSD_HOSPITAL",
                "agency_id": "TEST.AGENCY",
                "version": "1.0",
                "dimensions": [
                    {
                        "id": "REF_AREA",
                        "name": "Reference area",
                        "type": "Dimension",
                        "position": 1,
                        "codelist_id": "CL_AREA",
                        "codes": [{"id": "CAN", "label": "Canada"}],
                    },
                    {
                        "id": "MEASURE",
                        "name": "Measure",
                        "type": "Dimension",
                        "position": 2,
                        "codelist_id": "CL_MEASURE",
                        "codes": [{
                            "id": "DISCHARGE",
                            "label": "Hospital discharges",
                        }],
                    },
                    {
                        "id": "UNIT_MEASURE",
                        "name": "Unit of measure",
                        "type": "Dimension",
                        "position": 3,
                        "codelist_id": "CL_UNIT",
                        "codes": [{
                            "id": "DSC_10P5PS",
                            "label": "Discharges per 100 000 persons",
                        }],
                    },
                    {
                        "id": "DIAGNOSTIC_TYPE",
                        "name": "Diagnostic category",
                        "type": "Dimension",
                        "position": 4,
                        "codelist_id": "CL_DIAGNOSTIC",
                        "codes": [{
                            "id": "DICDA000",
                            "label": "All causes",
                        }],
                    },
                    {
                        "id": "TIME_PERIOD",
                        "name": "Time period",
                        "type": "TimeDimension",
                        "position": 5,
                        "codelist_id": None,
                        "codes": [],
                    },
                ],
            },
            "codelists": [],
        }
        concepts = build_dataset_concepts(structure)
        for concept in concepts:
            if concept["concept_role"] == "semantic":
                concept["semantic_text"] = (
                    "OECD dataset: Hospital discharges\n"
                    "Dimension: Measure\n"
                    "Code: DISCHARGE\n"
                    "Label: Hospital discharges"
                )
                concept["embedding"] = [1.0, 0.0]
                concept["embedding_model"] = "nomic-local"

        def response(status_code, content, url):
            result = requests.Response()
            result.status_code = status_code
            result._content = content
            result.url = url
            result.headers["Content-Type"] = "text/csv"
            return result

        dataset_path = (
            "TEST.AGENCY,DF_HOSPITAL_DISCHARGE,1.0/"
            "CAN.DISCHARGE.DSC_10P5PS.DICDA000"
        )
        requested_year = date.today().year - 1
        returned_year = requested_year - 1
        observation_url = (
            "https://sdmx.oecd.org/public/rest/v1/data/"
            + dataset_path
        )
        not_found = response(
            404,
            b"NoRecordsFound",
            observation_url,
        )
        observation = response(
            200,
            (
                "REF_AREA,MEASURE,UNIT_MEASURE,DIAGNOSTIC_TYPE,"
                "TIME_PERIOD,OBS_VALUE\n"
                f"CAN,DISCHARGE,DSC_10P5PS,DICDA000,{returned_year},7710.8\n"
            ).encode(),
            observation_url,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(database, "DB", f"{temp_dir}\\agent.db"):
                database.save_dataset_structure(structure)
                database.save_dataset_concepts(concepts)

                with (
                    patch.object(
                        embedding,
                        "embed_text",
                        return_value=[1.0, 0.0],
                    ),
                    patch.object(
                        api_data.requests,
                        "get",
                        side_effect=[not_found, observation],
                    ) as oecd_get,
                ):
                    agent = Agent()
                    query = (
                        "How many inpatient events per capita were there "
                        "in Canada last year"
                    )
                    result = agent.analyze_oecd_question(query)
                    with patch(
                        "agent.requests.post",
                        side_effect=AssertionError(
                            "A single-value response should not call the LLM"
                        ),
                    ):
                        answer = agent._answer_oecd_analysis(query, result)

        self.assertEqual(
            result["provenance"]["dataset_id"],
            dataset_id,
        )
        self.assertEqual(result["measure"]["code"], "DISCHARGE")
        self.assertEqual(
            result["filters"]["UNIT_MEASURE"]["code"],
            "DSC_10P5PS",
        )
        self.assertEqual(
            result["filters"]["DIAGNOSTIC_TYPE"]["code"],
            "DICDA000",
        )
        self.assertEqual(
            result["availability"]["requested_period"],
            str(requested_year),
        )
        self.assertEqual(
            result["availability"]["returned_period"],
            str(returned_year),
        )
        self.assertEqual(result["statistics"]["last_value"], 7710.8)
        self.assertIn("No observations were available for", answer)
        self.assertIn("2024", answer)
        self.assertIn("7,710.8", answer)
        self.assertIn("per 100 000 persons", answer)
        self.assertNotIn("change of 0", answer)
        self.assertEqual(oecd_get.call_count, 2)
        self.assertTrue(all(
            "/v1/data/" in call.args[0]
            and "/v1/dataflow/" not in call.args[0]
            for call in oecd_get.call_args_list
        ))
        self.assertEqual(
            [
                call.kwargs["params"]
                for call in oecd_get.call_args_list
            ],
            [
                {
                    "startPeriod": str(requested_year),
                    "endPeriod": str(requested_year),
                },
                {
                    "startPeriod": str(returned_year),
                    "endPeriod": str(returned_year),
                },
            ],
        )

    @patch("api_data.time.sleep")
    @patch("api_data.requests.get")
    def test_observation_request_retries_rate_limit(self, get, sleep):
        limited_response = Mock(status_code=429)
        limited_response.headers = {}
        success_response = Mock(status_code=200)
        get.side_effect = [limited_response, success_response]

        response = api_data._get_oecd_response(
            "https://example.test/observations",
            headers={"Accept": "text/csv;version=2"},
        )

        self.assertIs(response, success_response)
        self.assertEqual(get.call_count, 2)
        sleep.assert_called_once_with(2)
        limited_response.raise_for_status.assert_not_called()
        success_response.raise_for_status.assert_called_once_with()

    @patch("api_data.time.sleep")
    @patch("api_data.requests.get")
    def test_observation_request_surfaces_rate_limit_after_bounded_retries(
        self,
        get,
        sleep,
    ):
        limited_response = Mock(status_code=429)
        limited_response.headers = {}
        limited_response.raise_for_status.side_effect = api_data.requests.HTTPError(
            "429 Too Many Requests"
        )
        get.return_value = limited_response

        with self.assertRaisesRegex(api_data.requests.HTTPError, "429"):
            api_data._get_oecd_response("https://example.test/observations")

        self.assertEqual(get.call_count, api_data.OECD_MAX_REQUEST_ATTEMPTS)
        self.assertEqual(
            sleep.call_args_list,
            [call(2), call(4), call(8)],
        )

    def test_inpatient_query_uses_local_structure_and_only_requests_observations(self):
        dataset_id = "TEST.AGENCY,DSD_EVENTS@DF_HOSPITAL,1.0"
        structure = {
            "dataset_id": dataset_id,
            "dataflow": {
                "id": "DF_HOSPITAL",
                "agency_id": "TEST.AGENCY",
                "version": "1.0",
                "name": "Hospital events",
                "description": "Inpatient events",
            },
            "data_structure": {
                "id": "DSD_EVENTS",
                "agency_id": "TEST.AGENCY",
                "version": "1.0",
                "dimensions": [
                    {
                        "id": "REF_AREA",
                        "name": "Reference area",
                        "type": "Dimension",
                        "position": 1,
                        "codelist_id": "CL_AREA",
                        "codes": [
                            {"id": "MEX", "label": "Mexico"},
                            {"id": "USA", "label": "United States"},
                        ],
                    },
                    {
                        "id": "MEASURE",
                        "name": "Measure",
                        "type": "MeasureDimension",
                        "position": 2,
                        "codelist_id": "CL_MEASURE",
                        "codes": [
                            {"id": "INPATIENT", "label": "Inpatient events"},
                        ],
                    },
                    {
                        "id": "FREQ",
                        "name": "Frequency",
                        "type": "Dimension",
                        "position": 3,
                        "codelist_id": "CL_FREQ",
                        "codes": [{"id": "A", "label": "Annual"}],
                    },
                    {
                        "id": "TIME_PERIOD",
                        "name": "Time period",
                        "type": "TimeDimension",
                        "position": 4,
                        "codelist_id": None,
                        "codes": [],
                    },
                ],
            },
            "codelists": [],
        }
        concepts = build_dataset_concepts(structure)
        for concept in concepts:
            if concept["concept_role"] == "semantic":
                concept["semantic_text"] = (
                    "OECD dataset: Hospital events\n"
                    "Dimension: Measure\n"
                    "Code: INPATIENT\n"
                    "Label: Inpatient events"
                )
                concept["embedding"] = [1.0, 0.0]
                concept["embedding_model"] = "nomic-local"

        query = "How many inpatient events happen in Mexico last year"
        previous_year = str(date.today().year - 1)
        csv_content = (
            "REF_AREA,MEASURE,FREQ,TIME_PERIOD,OBS_VALUE,OBS_STATUS\n"
            f"MEX,INPATIENT,A,{previous_year},42,A\n"
        ).encode("utf-8")
        observation_response = Mock()
        observation_response.content = csv_content
        observation_response.url = "https://sdmx.oecd.org/public/rest/v1/data/test"
        observation_response.raise_for_status.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(database, "DB", f"{temp_dir}\\agent.db"):
                database.save_dataset_structure(structure)
                database.save_dataset_concepts(concepts)

                with (
                    patch.object(embedding, "embed_text", return_value=[1.0, 0.0]),
                    patch.object(
                        api_data.requests,
                        "get",
                        return_value=observation_response,
                    ) as oecd_get,
                ):
                    agent = Agent()
                    self.assertEqual(
                        agent.route_request(query),
                        {"route": "oecd_analysis"},
                    )
                    result = agent.analyze_oecd_question(query)

        self.assertEqual(len(result["observations"]), 1)
        self.assertEqual(result["observations"][0]["REF_AREA"], "MEX")
        self.assertEqual(result["observations"][0]["OBS_VALUE"], 42.0)
        self.assertEqual(result["statistics"]["first_period"], previous_year)
        self.assertEqual(oecd_get.call_count, 1)
        request = oecd_get.call_args
        self.assertIn("/v1/data/", request.args[0])
        self.assertNotIn("dataflow", request.args[0].lower())
        self.assertEqual(
            request.kwargs["params"],
            {"startPeriod": previous_year, "endPeriod": previous_year},
        )
        self.assertTrue(request.args[0].endswith("/MEX.INPATIENT.A"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
