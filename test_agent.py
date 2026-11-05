import unittest

from agent import Agent


class OECDAnalysisWorkflowTests(unittest.TestCase):
    def test_natural_language_question_resolves_to_validated_analysis(self):
        question = (
            "How have asthma hospital admissions changed in Australia "
            "since 2016?"
        )
        result = Agent().analyze_oecd_question(question)
        statistics = result["statistics"]

        self.assertEqual(statistics["first_period"], "2016")
        self.assertEqual(statistics["last_period"], "2023")
        self.assertEqual(statistics["first_value"], 70.4)
        self.assertEqual(statistics["last_value"], 52.0)
        self.assertAlmostEqual(statistics["absolute_change"], -18.4)
        self.assertAlmostEqual(statistics["percent_change"], -26.1363636364)
        self.assertAlmostEqual(statistics["mean"], 55.5)
        self.assertEqual(
            result["measure"]["label"],
            "Asthma hospital admission",
        )
        self.assertEqual(
            result["filters"]["REF_AREA"]["label"],
            "Australia",
        )
        self.assertEqual(len(result["observations"]), 8)


if __name__ == "__main__":
    unittest.main()
