import tempfile
import unittest
from unittest.mock import Mock, patch

import database
import embedding
from analysis import analyze_data_source_time_series
from data_sources import CMSDataSource, OECDDataSource, WHODataSource
from data_sources.retrieval import DataSourceRetrieval
from data_sources.who import WHO_MAX_PAGE_SIZE


WHO_INDICATOR_CODE = "WHO_TEST_LIFE_EXPECTANCY"
WHO_API_BASE = "https://who.example.test/api"


def _indicator(code=WHO_INDICATOR_CODE):
    return {
        "IndicatorCode": code,
        "IndicatorName": "Life expectancy at birth",
        "Definition": "Average number of years a person may expect to live.",
        "Rationale": "Summarizes population health and mortality.",
        "URL": "https://www.who.int/data/gho/data/indicators/indicator-details/GHO/life-expectancy-at-birth",
        "SpatialDimType": "COUNTRY",
        "TimeDimType": "YEAR",
        "Dim1Type": "SEX",
        "Unit": "Years",
        "DataSource": "World Health Organization",
    }


def _response(payload, url):
    response = Mock()
    response.status_code = 200
    response.headers = {}
    response.url = url
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


def _observation(year, value):
    return {
        "IndicatorCode": WHO_INDICATOR_CODE,
        "IndicatorName": "Life expectancy at birth",
        "SpatialDimType": "COUNTRY",
        "SpatialDim": "CAN",
        "TimeDimType": "YEAR",
        "TimeDim": str(year),
        "Dim1Type": "SEX",
        "Dim1": "MLE",
        "Dim1ValueCode": "MLE",
        "Dim1Value": "Male",
        "NumericValue": value,
        "Value": str(value),
        "TimeDimensionBegin": f"{year}-01-01",
        "TimeDimensionEnd": f"{year}-12-31",
        "ValueType": "numeric",
        "Comments": "Fixture observation status",
    }


def _api_response(url, params=None, headers=None, timeout=60):
    if url.endswith("/Indicator"):
        indicators = [
            _indicator(),
            {
                **_indicator("WHO_TEST_MORTALITY"),
                "IndicatorName": "Under-five mortality rate",
                "Definition": "Probability of dying before age five.",
                "Unit": "Deaths per 1,000 live births",
            },
        ]
        if params and "$filter" in params:
            target = params["$filter"].split("'")[1]
            indicators = [
                item for item in indicators
                if item["IndicatorCode"] == target
            ]
        return _response({"value": indicators}, url)
    if "/Dimension/" in url and url.endswith("/DimensionValues"):
        dimension = url.split("/Dimension/")[1].split("/")[0]
        values = {
            "COUNTRY": [
                {"Code": "CAN", "Title": "Canada"},
                {"Code": "USA", "Title": "United States"},
            ],
            "SEX": [
                {"Code": "MLE", "Title": "Male"},
                {"Code": "FMLE", "Title": "Female"},
            ],
        }
        return _response({
            "value": values.get(dimension, []),
        }, url)
    if url.endswith(f"/{WHO_INDICATOR_CODE}"):
        return _response({
            "value": [
                _observation(2020, 79.1),
                _observation(2021, 79.4),
                _observation(2022, 79.8),
            ],
        }, url)
    raise AssertionError(f"Unexpected WHO endpoint: {url}")


class WHODataSourceTests(unittest.TestCase):
    def test_who_is_registered_as_a_third_source(self):
        sources = DataSourceRetrieval().data_sources
        self.assertEqual(
            set(sources),
            {"OECD", "CMS", "WHO"},
        )
        self.assertIsInstance(sources["WHO"], WHODataSource)
        self.assertIsInstance(sources["CMS"], CMSDataSource)
        self.assertIsInstance(sources["OECD"], OECDDataSource)

    def test_discovery_structure_metadata_and_filter_concepts(self):
        source = WHODataSource(api_base_url=WHO_API_BASE)
        with patch(
            "data_sources.who._get_who_response",
            side_effect=_api_response,
        ) as who_get:
            discovered = source.discover_datasets(limit=2)
            self.assertEqual(len(discovered), 2)
            self.assertEqual(discovered[0]["source"], "WHO")
            self.assertEqual(
                discovered[0]["dataset_id"],
                WHO_INDICATOR_CODE,
            )
            self.assertIn("Life expectancy", discovered[0]["title"])
            self.assertEqual(
                discovered[0]["raw"]["Definition"],
                _indicator()["Definition"],
            )

            metadata = source.get_metadata(WHO_INDICATOR_CODE)
            structure = source.get_structure(
                WHO_INDICATOR_CODE,
                metadata,
            )
            dimensions = {
                item["id"]: item
                for item in structure["data_structure"]["dimensions"]
            }
            self.assertEqual(dimensions["SpatialDim"]["name"], "Country")
            self.assertEqual(
                dimensions["SpatialDim"]["codes"],
                [
                    {"id": "CAN", "label": "Canada"},
                    {"id": "USA", "label": "United States"},
                ],
            )
            self.assertEqual(dimensions["Dim1"]["name"], "Sex")
            self.assertEqual(dimensions["TimeDim"]["type"], "TimeDimension")
            self.assertEqual(structure["unit"], "Years")
            self.assertTrue(any(
                "/Dimension/COUNTRY/DimensionValues" in call.args[0]
                for call in who_get.call_args_list
            ))
            self.assertFalse(any(
                "/Dimension/YEAR/DimensionValues" in call.args[0]
                for call in who_get.call_args_list
            ))

            concepts = source.build_concepts(
                WHO_INDICATOR_CODE,
                metadata,
                structure,
            )
            semantic = [
                concept
                for concept in concepts
                if concept["concept_role"] == "semantic"
            ]
            filters = [
                concept
                for concept in concepts
                if concept["concept_role"] == "filter"
            ]
            self.assertEqual(len(semantic), 1)
            self.assertIn("Country", semantic[0]["semantic_text"])
            self.assertIn("Sex", semantic[0]["semantic_text"])
            self.assertFalse(any(
                code in semantic[0]["semantic_text"]
                for code in ("CAN", "USA", "MLE", "FMLE")
            ))
            self.assertEqual(len(filters), 4)
            self.assertTrue(all(
                concept["concept_role"] == "filter"
                and not concept.get("embedding")
                for concept in filters
            ))
            self.assertIn(
                "https://www.who.int/data/gho/data/indicators/"
                "indicator-details/GHO/life-expectancy-at-birth",
                metadata["source_url"],
            )

    def test_bounded_index_filters_observations_dataframe_and_analysis(self):
        source = WHODataSource(api_base_url=WHO_API_BASE)
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(database, "DB", f"{temp_dir}\\who.db"):
                with (
                    patch(
                        "data_sources.who._get_who_response",
                        side_effect=_api_response,
                    ),
                    patch.object(
                        embedding,
                        "embed_texts",
                        return_value=[[1.0, 0.0]],
                    ) as embed_texts,
                    patch.object(
                        embedding,
                        "embed_text",
                        return_value=[1.0, 0.0],
                    ),
                ):
                    report = source.index_datasets(limit=1)
                    self.assertEqual(report["totals"]["datasets_indexed"], 1)
                    self.assertEqual(report["totals"]["semantic_concepts"], 1)
                    self.assertEqual(report["totals"]["filter_concepts"], 4)
                    self.assertEqual(report["totals"]["embeddings_created"], 1)
                    self.assertEqual(report["failures"], [])
                    self.assertEqual(embed_texts.call_count, 1)

                    filters = source.find_filters(
                        WHO_INDICATOR_CODE,
                        dimension_id="SpatialDim",
                    )
                    self.assertEqual(
                        [(item["code"], item["label"]) for item in filters],
                        [("CAN", "Canada"), ("USA", "United States")],
                    )
                    with patch(
                        "data_sources.who._get_who_response",
                        side_effect=_api_response,
                    ) as who_get:
                        result = analyze_data_source_time_series(
                            source,
                            WHO_INDICATOR_CODE,
                            {"SpatialDim": "CAN", "Dim1": "MLE"},
                            start_period="2020",
                            end_period="2022",
                            limit=5,
                        )

                    self.assertEqual(
                        result["measure"]["code"],
                        WHO_INDICATOR_CODE,
                    )
                    self.assertEqual(result["statistics"]["last_value"], 79.8)
                    self.assertEqual(
                        result["observations"][0]["SpatialDim"],
                        "CAN",
                    )
                    self.assertEqual(
                        result["observations"][0]["SpatialDim_LABEL"],
                        "Canada",
                    )
                    self.assertEqual(
                        result["observations"][0]["Dim1Value"],
                        "Male",
                    )
                    self.assertEqual(
                        result["observations"][0]["Comments"],
                        "Fixture observation status",
                    )
                    self.assertEqual(result["provenance"]["source"], "WHO")
                    data_call = next(
                        call
                        for call in who_get.call_args_list
                        if call.args[0].endswith(WHO_INDICATOR_CODE)
                    )
                    self.assertEqual(
                        data_call.kwargs["params"]["$top"],
                        5,
                    )
                    self.assertIn(
                        "SpatialDim eq 'CAN'",
                        data_call.kwargs["params"]["$filter"],
                    )
                    self.assertIn(
                        "Dim1 eq 'MLE'",
                        data_call.kwargs["params"]["$filter"],
                    )
                    self.assertIn(
                        "date(TimeDimensionBegin) ge 2020-01-01",
                        data_call.kwargs["params"]["$filter"],
                    )
                    self.assertIn(
                        "date(TimeDimensionBegin) lt 2023-01-01",
                        data_call.kwargs["params"]["$filter"],
                    )

                    trace = DataSourceRetrieval(
                        data_sources=(source,),
                    ).retrieve("life expectancy in Canada")
                    self.assertEqual(trace["selected_source"], "WHO")
                    self.assertEqual(
                        trace["selected_concept"]["label"],
                        "Life expectancy at birth",
                    )
                    database.save_dataset_concepts([{
                        "source": "OECD",
                        "dataset_id": "OECD.TEST,DSD@HEALTH,1.0",
                        "dimension_id": "MEASURE",
                        "dimension_name": "Measure",
                        "code": "LIFE_EXPECTANCY",
                        "label": "Life expectancy",
                        "concept_role": "semantic",
                        "semantic_text": "OECD life expectancy",
                        "embedding": [0.0, 1.0],
                        "embedding_model": "nomic-local",
                    }])
                    with patch.object(
                        embedding,
                        "embed_text",
                        return_value=[1.0, 0.0],
                    ):
                        cross_source = embedding.search_dataset_concepts(
                            "life expectancy",
                            top_k=10,
                        )
                    self.assertEqual(
                        {item["source"] for item in cross_source},
                        {"OECD", "WHO"},
                    )

    def test_observation_request_rejects_unknown_fields_codes_and_unbounded_limit(self):
        source = WHODataSource(api_base_url=WHO_API_BASE)
        source._discovered[WHO_INDICATOR_CODE] = (
            source._normalize_metadata(WHO_INDICATOR_CODE, _indicator())
        )
        with patch(
            "data_sources.who._get_who_response",
            side_effect=_api_response,
        ):
            with self.assertRaisesRegex(ValueError, "Unknown or non-filterable"):
                source.retrieve_data(
                    WHO_INDICATOR_CODE,
                    filters={"UnknownDim": "x"},
                    limit=10,
                )
            with self.assertRaisesRegex(ValueError, "not an indexed code"):
                source.retrieve_data(
                    WHO_INDICATOR_CODE,
                    filters={"SpatialDim": "MISSING"},
                    limit=10,
                )
            with self.assertRaises(ValueError):
                source.retrieve_data(
                    WHO_INDICATOR_CODE,
                    start_period="2023",
                    end_period="2020",
                    limit=10,
                )
            with self.assertRaises(ValueError):
                source.retrieve_data(
                    WHO_INDICATOR_CODE,
                    limit=WHO_MAX_PAGE_SIZE + 1,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
