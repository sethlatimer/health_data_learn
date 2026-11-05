import os
import unittest

import requests

from data_sources.who import WHODataSource


LIVE_CONCEPTS = {
    "life_expectancy": ("life expectancy",),
    "mortality": ("mortality",),
    "immunization": ("immuniz", "vaccin"),
}
LIVE_CATALOG_LIMIT = 5000


@unittest.skipUnless(
    os.environ.get("RUN_WHO_LIVE_TESTS") == "1",
    "Set RUN_WHO_LIVE_TESTS=1 to call the official WHO GHO API",
)
class WHOLiveSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = WHODataSource()
        try:
            cls.datasets = cls.source.discover_datasets(
                limit=LIVE_CATALOG_LIMIT,
            )
        except requests.RequestException as error:
            raise unittest.SkipTest(
                f"WHO GHO indicator catalog unavailable: {error}"
            ) from error
        cls.matches = {}
        for case, terms in LIVE_CONCEPTS.items():
            match = next(
                (
                    item
                    for item in cls.datasets
                    if all(
                        term in (
                            item["title"] + " " + item["description"]
                        ).casefold()
                        for term in terms
                    )
                ),
                None,
            )
            if match is None:
                raise unittest.SkipTest(
                    f"WHO catalog page did not include a {case} indicator"
                )
            cls.matches[case] = match
        if len({
            item["dataset_id"] for item in cls.matches.values()
        }) != len(LIVE_CONCEPTS):
            raise unittest.SkipTest(
                "WHO catalog did not expose three distinct live concepts"
            )

    def _smoke_indicator(self, concept):
        metadata = self.matches[concept]
        structure = self.source.get_structure(
            metadata["dataset_id"],
            metadata=metadata,
        )
        result = self.source.retrieve_data(
            metadata["dataset_id"],
            limit=1,
        )
        dataframe = self.source.normalize_dataframe(result, structure)
        self.assertEqual(result["source"], "WHO")
        self.assertLessEqual(len(result["rows"]), 1)
        self.assertIn("dataset_id", dataframe.columns)
        self.assertIn("source", dataframe.columns)
        self.assertIn("OBS_VALUE", dataframe.columns)
        self.assertIn("TIME_PERIOD", dataframe.columns)

    def test_live_life_expectancy_indicator(self):
        self._smoke_indicator("life_expectancy")

    def test_live_mortality_indicator(self):
        self._smoke_indicator("mortality")

    def test_live_immunization_indicator(self):
        self._smoke_indicator("immunization")


if __name__ == "__main__":
    unittest.main(verbosity=2)
