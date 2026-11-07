import tempfile
import unittest
from unittest.mock import patch

import database
from data_sources.base import DataSource
from data_sources.compatibility import (
    classify_measure,
    check_concept_compatibility,
    check_dataset_compatibility,
)
from data_sources.retrieval import (
    DataSourceRetrieval,
    parse_question_requirements,
)


class BenchmarkDataSource(DataSource):
    source = "BENCH"

    def discover_datasets(self, limit=None):
        return []

    def get_metadata(self, dataset_id):
        return {}

    def get_structure(self, dataset_id, metadata=None):
        raise NotImplementedError

    def build_concepts(self, dataset_id, metadata, structure):
        return []

    def find_filters(self, dataset_id, dimension_id=None, search=None):
        return []

    def retrieve_data(
        self,
        dataset_id,
        filters=None,
        start_period=None,
        end_period=None,
        limit=1000,
    ):
        return {}

    def normalize_dataframe(self, result, structure):
        raise NotImplementedError


def _profile(dataset_id, title, description, concept_label, concept_text,
             unit, latest=True, adjustment=None):
    dimensions = [
        {
            "id": "COUNTRY",
            "name": "Country",
            "type": "Dimension",
            "position": 1,
            "codelist_id": "CL_COUNTRY",
            "codes": [
                {"id": "CAN", "label": "Canada"},
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
                {"id": dataset_id, "label": concept_label},
            ],
        },
        {
            "id": "UNIT_MEASURE",
            "name": "Unit of measure",
            "type": "Dimension",
            "position": 3,
            "codelist_id": "CL_UNIT",
            "codes": [{"id": "UNIT", "label": unit}],
        },
        {
            "id": "TIME_PERIOD",
            "name": "Time period",
            "type": "TimeDimension",
            "position": 4,
            "codelist_id": None,
            "codes": [],
        },
    ]
    semantic_text = concept_text
    if adjustment:
        semantic_text += f"\nAdjustment: {adjustment}"
    structure = {
        "source": "BENCH",
        "dataset_id": dataset_id,
        "dataflow": {
            "id": dataset_id,
            "agency_id": "BENCH",
            "version": "2026",
            "name": title,
            "description": description,
        },
        "data_structure": {
            "id": f"DSD_{dataset_id}",
            "agency_id": "BENCH",
            "version": "2026",
            "dimensions": dimensions,
        },
        "codelists": [],
    }
    concept = {
        "source": "BENCH",
        "dataset_id": dataset_id,
        "dimension_id": "MEASURE",
        "dimension_name": "Measure",
        "code": dataset_id,
        "label": concept_label,
        "concept_role": "semantic",
        "semantic_text": semantic_text,
        "embedding": [1.0, 0.0],
        "embedding_model": "nomic-local",
    }
    return {
        "source": "BENCH",
        "dataset_id": dataset_id,
        "title": title,
        "description": description,
        "source_url": f"https://example.test/{dataset_id}",
        "metadata": {
            "temporal": [{
                "startDate": "2016-01-01",
                "endDate": "2025-12-31",
            }],
            "is_latest_release": latest,
        },
        "is_latest_release": latest,
        "structure": structure,
        "concepts": [concept],
    }


BENCHMARK_PROFILES = (
    _profile(
        "utilization-count",
        "Hospital inpatient utilization",
        "Original Medicare inpatient hospital services and use.",
        "Total inpatient discharges",
        "The number of inpatient hospital discharges.",
        "Count",
    ),
    _profile(
        "hospital-spending",
        "Hospital inpatient spending",
        "Original Medicare hospital payments and costs.",
        "Total Medicare payment amount",
        "Total payment amount in dollars for inpatient services.",
        "US dollars",
    ),
    _profile(
        "utilization-rate",
        "Hospital admission rate",
        "Inpatient utilization rate per 1,000 persons.",
        "Inpatient admissions per 1,000 persons",
        "Rate of hospital admissions per 1,000 persons.",
        "Admissions per 1,000 persons",
    ),
    _profile(
        "standardized-rate",
        "Age-standardized hospital admission rate",
        "Age standardized inpatient hospital rate per 1,000 persons.",
        "Age-standardized inpatient admission rate per 1,000 persons",
        "Age-standardized hospital admission rate per 1,000 persons.",
        "Admissions per 1,000 persons",
        adjustment="standardized",
    ),
    _profile(
        "crude-rate",
        "Crude hospital admission rate",
        "Crude inpatient hospital rate per 1,000 persons.",
        "Crude inpatient admission rate per 1,000 persons",
        "Crude hospital admission rate per 1,000 persons.",
        "Admissions per 1,000 persons",
        adjustment="crude",
    ),
    _profile(
        "archived-utilization",
        "Archived hospital inpatient utilization",
        "Archived Original Medicare inpatient hospital discharges.",
        "Total inpatient discharges",
        "The number of inpatient hospital discharges.",
        "Count",
        latest=False,
    ),
)


class RetrievalBenchmarkTests(unittest.TestCase):
    def test_deterministic_requirement_parser_benchmark_cases(self):
        utilization = parse_question_requirements(
            "How many inpatient hospital discharges in Canada in 2024?"
        )
        self.assertEqual(utilization["topic"], "utilization")
        self.assertEqual(utilization["measure_type"], "count")
        self.assertEqual(utilization["geography"]["name"], "Canada")
        self.assertEqual(utilization["end_period"], "2024")

        spending = parse_question_requirements(
            "What were hospital spending and payments in the United States?"
        )
        self.assertEqual(spending["topic"], "spending")
        self.assertEqual(spending["measure_type"], "amount")

        rate = parse_question_requirements(
            "What was the inpatient admission rate per 1,000 persons?"
        )
        self.assertEqual(rate["topic"], "utilization")
        self.assertEqual(rate["measure_type"], "rate")
        self.assertEqual(rate["measure_family"], "utilization")
        self.assertEqual(rate["quantity_type"], "per_capita")
        self.assertEqual(rate["unit"], "per 1,000 persons")
        self.assertEqual(rate["care_setting"], "inpatient")

        standardized = parse_question_requirements(
            "Compare the age-standardized admission rate since 2020"
        )
        self.assertEqual(standardized["adjustment"], "standardized")
        self.assertEqual(standardized["start_period"], "2020")

        crude = parse_question_requirements(
            "What is the crude hospitalization rate?"
        )
        self.assertEqual(crude["adjustment"], "crude")

    def test_measure_intent_parses_family_quantity_and_domain(self):
        provider_count = parse_question_requirements(
            "How many providers and physicians are available?"
        )
        self.assertEqual(
            provider_count["measure_family"],
            "provider_capacity",
        )
        self.assertEqual(provider_count["quantity_type"], "count")
        physician_supply = parse_question_requirements(
            "Which dataset measures physician supply across countries?"
        )
        self.assertEqual(physician_supply["provider_type"], "physician")

        provider_payment = parse_question_requirements(
            "What were Medicare provider payments?"
        )
        self.assertEqual(
            provider_payment["measure_family"],
            "spending_payment",
        )
        self.assertEqual(provider_payment["quantity_type"], "currency")
        self.assertEqual(
            provider_payment["financial_measure"],
            "payment",
        )

        mortality = parse_question_requirements(
            "What was the age-standardized mortality rate?"
        )
        self.assertEqual(mortality["measure_family"], "mortality")
        self.assertEqual(
            mortality["quantity_type"],
            "standardized_rate",
        )

        behavioral = parse_question_requirements(
            "Which measures describe behavioral health service utilization?"
        )
        self.assertEqual(behavioral["measure_family"], "utilization")
        self.assertEqual(behavioral["measure_domain"], "behavioral_health")
        self.assertIsNone(behavioral["specific_condition"])

    def test_measure_classifier_and_compatibility_reject_paired_substitutions(self):
        utilization_request = parse_question_requirements(
            "Which measures describe inpatient utilization?"
        )
        count_profile = BENCHMARK_PROFILES[0]
        spending_profile = BENCHMARK_PROFILES[1]
        self.assertEqual(
            classify_measure(count_profile["concepts"][0], count_profile)[
                "family"
            ],
            "utilization",
        )
        spending_check = check_concept_compatibility(
            spending_profile["concepts"][0],
            utilization_request,
            profile=spending_profile,
        )
        self.assertFalse(spending_check["compatible"])
        self.assertTrue(any(
            "spending_payment is incompatible" in reason
            for reason in spending_check["rejection_reasons"]
        ))

        count_request = parse_question_requirements(
            "How many inpatient admissions occurred?"
        )
        rate_profile = BENCHMARK_PROFILES[2]
        rate_check = check_concept_compatibility(
            rate_profile["concepts"][0],
            count_request,
            profile=rate_profile,
        )
        self.assertFalse(rate_check["compatible"])
        self.assertTrue(any(
            "Quantity type" in reason
            for reason in rate_check["rejection_reasons"]
        ))

        provider_count = _profile(
            "provider-count",
            "Provider supply",
            "Counts of healthcare providers.",
            "Number of providers",
            "Number of active healthcare providers.",
            "Count",
        )
        provider_payment = _profile(
            "provider-payment",
            "Provider payment",
            "Medicare payment amounts to providers.",
            "Total provider payment amount",
            "Total Medicare payment amount to providers in dollars.",
            "US dollars",
        )
        provider_count["concepts"][0]["label"] = "Number of providers"
        provider_count["concepts"][0]["metadata"] = {
            "description": "Number of active healthcare providers."
        }
        provider_payment["concepts"][0]["label"] = (
            "Total provider payment amount"
        )
        provider_payment["concepts"][0]["metadata"] = {
            "description": "Total Medicare payment amount to providers in dollars."
        }
        provider_payment_request = parse_question_requirements(
            "What were Medicare provider payments?"
        )
        provider_mismatch = check_concept_compatibility(
            provider_count["concepts"][0],
            provider_payment_request,
            profile=provider_count,
        )
        self.assertFalse(provider_mismatch["compatible"])
        self.assertIn(
            "financial_measure",
            [check["characteristic"] for check in provider_mismatch["checks"]],
        )
        self.assertTrue(
            check_concept_compatibility(
                provider_payment["concepts"][0],
                provider_payment_request,
                profile=provider_payment,
            )["compatible"]
        )
        beds = _profile(
            "hospital-beds",
            "Hospital capacity",
            "Hospital beds.",
            "Hospital beds",
            "Number of hospital beds.",
            "Count",
        )
        physician_supply = parse_question_requirements(
            "Which dataset measures physician supply across countries?"
        )
        beds_check = check_concept_compatibility(
            beds["concepts"][0],
            physician_supply,
            profile=beds,
        )
        self.assertFalse(beds_check["compatible"])
        self.assertIn(
            "provider_type",
            [check["characteristic"] for check in beds_check["checks"]],
        )

        mortality_count = _profile(
            "mortality-count",
            "Mortality",
            "Mortality events.",
            "Number of deaths",
            "Number of deaths.",
            "Count",
        )
        mortality_rate = _profile(
            "mortality-rate",
            "Mortality rates",
            "Mortality rate measures.",
            "Crude mortality rate",
            "Crude mortality rate per 100,000 persons.",
            "Rate per 100,000 persons",
            adjustment="crude",
        )
        mortality_count_request = parse_question_requirements(
            "How many deaths occurred?"
        )
        mortality_count_mismatch = check_concept_compatibility(
            mortality_rate["concepts"][0],
            mortality_count_request,
            profile=mortality_rate,
        )
        self.assertFalse(mortality_count_mismatch["compatible"])
        mortality_rate_request = parse_question_requirements(
            "What was the mortality rate?"
        )
        mortality_rate_mismatch = check_concept_compatibility(
            mortality_count["concepts"][0],
            mortality_rate_request,
            profile=mortality_count,
        )
        self.assertFalse(mortality_rate_mismatch["compatible"])

        standardized_request = parse_question_requirements(
            "What was the age-standardized mortality rate?"
        )
        crude_check = check_concept_compatibility(
            mortality_rate["concepts"][0],
            standardized_request,
            profile=mortality_rate,
        )
        self.assertFalse(crude_check["compatible"])
        self.assertIn(
            "standardized",
            next(
                check
                for check in crude_check["checks"]
                if check["characteristic"] == "adjustment"
            )["requested"],
        )

        behavioral = _profile(
            "adhd-utilization",
            "Behavioral health utilization",
            "Behavioral health service use.",
            "Percent of beneficiaries with ADHD or conduct disorders",
            "Percent of beneficiaries with ADHD or conduct disorders.",
            "Percent",
        )
        behavioral["concepts"][0]["metadata"] = {
            "description": (
                "Percentage of beneficiaries diagnosed with ADHD and "
                "other conduct disorders."
            )
        }
        behavioral_check = check_concept_compatibility(
            behavioral["concepts"][0],
            parse_question_requirements(
                "Which measures describe behavioral health service utilization?"
            ),
            profile=behavioral,
        )
        self.assertFalse(behavioral_check["compatible"])
        self.assertTrue(any(
            "Condition-specific adhd" in reason
            for reason in behavioral_check["rejection_reasons"]
        ))
        self.assertEqual(
            behavioral_check["measure_classification"]["condition"],
            "adhd",
        )

    def test_uncertain_measure_classification_remains_unknown(self):
        concept = {
            "dimension_id": "MEASURE",
            "label": "Composite indicator",
            "code": "X1",
            "metadata": {},
        }
        classification = classify_measure(concept)
        self.assertEqual(classification["quantity_type"], "unknown")
        self.assertEqual(classification["confidence"], "unknown")

        current = parse_question_requirements(
            "Show the latest current inpatient utilization"
        )
        self.assertEqual(current["currentness"], "current")
        archived = parse_question_requirements(
            "Show the archived historical utilization series"
        )
        self.assertEqual(archived["currentness"], "archived")

    def test_concept_checks_reject_count_rate_and_adjustment_substitution(self):
        rate_request = parse_question_requirements(
            "What was the inpatient admission rate?"
        )
        count_concept = BENCHMARK_PROFILES[0]["concepts"][0]
        self.assertFalse(
            check_concept_compatibility(
                count_concept,
                rate_request,
                profile=BENCHMARK_PROFILES[0],
            )["compatible"]
        )

        count_request = parse_question_requirements(
            "How many inpatient hospital discharges?"
        )
        readmission_concept = {
            **count_concept,
            "label": (
                "Congestive heart failure - All-cause hospital "
                "readmissions within 365 days after discharge"
            ),
            "semantic_text": (
                "Congestive heart failure - All-cause hospital "
                "readmissions within 365 days after discharge"
            ),
        }
        check = check_concept_compatibility(
            readmission_concept,
            count_request,
            profile=BENCHMARK_PROFILES[0],
        )
        self.assertFalse(check["compatible"])

        standardized_request = parse_question_requirements(
            "What was the standardized admission rate?"
        )
        crude_concept = BENCHMARK_PROFILES[4]["concepts"][0]
        check = check_concept_compatibility(
            crude_concept,
            standardized_request,
            profile=BENCHMARK_PROFILES[4],
        )
        self.assertFalse(check["compatible"])
        self.assertTrue(check["rejection_reasons"])

    def test_requested_rate_unit_must_match_indexed_concept(self):
        request = parse_question_requirements(
            "What was the inpatient admission rate per 1,000 persons?"
        )
        matching = check_concept_compatibility(
            BENCHMARK_PROFILES[2]["concepts"][0],
            request,
            profile=BENCHMARK_PROFILES[2],
        )
        self.assertTrue(matching["compatible"])
        self.assertTrue(
            next(
                check
                for check in matching["checks"]
                if check["characteristic"] == "unit"
            )["supported"]
        )

        incompatible = check_concept_compatibility(
            BENCHMARK_PROFILES[0]["concepts"][0],
            request,
            profile=BENCHMARK_PROFILES[0],
        )
        self.assertFalse(incompatible["compatible"])
        self.assertIn(
            "unit",
            [
                check["characteristic"]
                for check in incompatible["checks"]
                if not check["supported"]
            ],
        )

    def test_release_period_supports_snapshot_but_not_trend_without_time_field(self):
        profile = {
            "source": "CMS",
            "dataset_id": "cms-inpatient-2024",
            "title": "Medicare Inpatient Hospitals - by Provider : 2024",
            "description": "Provider-level inpatient hospital data.",
            "metadata": {
                "temporal": [{
                    "startDate": "2024-01-01",
                    "endDate": "2024-12-31",
                }],
            },
            "structure": {
                "data_structure": {
                    "dimensions": [{
                        "id": "Tot_Dschrgs",
                        "name": "Total Discharges",
                        "type": "MeasureDimension",
                        "description": "Total number of inpatient discharges.",
                        "metadata": {
                            "name": "Tot_Dschrgs",
                            "label": "Total Discharges",
                        },
                        "codes": [],
                    }],
                },
            },
            "concepts": [{
                "dimension_id": "Tot_Dschrgs",
                "dimension_name": "Total Discharges",
                "label": "Total Discharges",
                "concept_role": "semantic",
                "metadata": {
                    "description": "Total number of inpatient discharges.",
                },
            }],
        }
        snapshot = check_dataset_compatibility(
            profile,
            parse_question_requirements(
                "How many inpatient hospital discharges during 2024?"
            ),
        )
        self.assertTrue(
            next(
                check
                for check in snapshot["checks"]
                if check["characteristic"] == "time_range"
            )["supported"]
        )
        self.assertTrue(snapshot["compatible"])

        trend = check_dataset_compatibility(
            profile,
            parse_question_requirements(
                "How did inpatient hospital discharges change from 2023 to 2024?"
            ),
        )
        self.assertFalse(trend["compatible"])
        self.assertFalse(
            next(
                check
                for check in trend["checks"]
                if check["characteristic"] == "time_range"
            )["supported"]
        )

    def test_exact_dataset_title_wins_when_semantic_scores_are_tied(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(database, "DB", f"{temp_dir}\\agent.db"):
                discharge = _profile(
                    "hospital-discharge-specific",
                    "Hospital discharges by diagnostic categories",
                    "Hospital discharge counts by diagnostic category.",
                    "Hospital discharges",
                    "Hospital discharges",
                    "Count",
                )
                average_stay = _profile(
                    "hospital-average-stay",
                    "Hospital average length of stay by diagnostic categories",
                    "Average hospital length of stay by diagnosis.",
                    "Hospital discharges",
                    "Hospital discharges",
                    "Count",
                )
                profiles = (discharge, average_stay)
                for profile in profiles:
                    database.save_dataset_record(
                        source="BENCH",
                        dataset_id=profile["dataset_id"],
                        title=profile["title"],
                        description=profile["description"],
                        source_url=profile["source_url"],
                        metadata=profile["metadata"],
                        is_latest_release=profile["is_latest_release"],
                    )
                    database.save_dataset_structure(
                        profile["structure"],
                        source="BENCH",
                    )
                    database.save_dataset_concepts(profile["concepts"])

                with (
                    patch(
                        "data_sources.retrieval.embedding.rank_dataset_candidates",
                        return_value=[
                            {
                                "source": "BENCH",
                                "dataset_id": profile["dataset_id"],
                                "score": 0.8,
                                "best_concept_score": 0.8,
                                "semantic_concepts_scored": 1,
                            }
                            for profile in profiles
                        ],
                    ),
                    patch(
                        "data_sources.retrieval.embedding.search_dataset_concepts",
                        side_effect=lambda query, top_k, source, dataset_ids,
                        instrumentation=None: [
                            {
                                **profile["concepts"][0],
                                "score": 0.8,
                            }
                            for profile in profiles
                            if profile["dataset_id"] in dataset_ids
                        ],
                    ) as scoped_concepts,
                ):
                    trace = DataSourceRetrieval(
                        [BenchmarkDataSource()]
                    ).retrieve(
                        "How many inpatient hospital discharges in Canada "
                        "during 2024?",
                        top_dataset_candidates=1,
                        concept_dataset_count=1,
                    )

                self.assertEqual(
                    trace["selected_dataset"]["dataset_id"],
                    "hospital-discharge-specific",
                )
                self.assertEqual(
                    len(trace["stage_1_dataset_candidates"]),
                    1,
                )
                self.assertEqual(
                    scoped_concepts.call_args.kwargs["dataset_ids"],
                    ["hospital-discharge-specific"],
                )

    def test_spending_retrieval_rejects_unrelated_survey_measures(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(database, "DB", f"{temp_dir}\\agent.db"):
                hospital = BENCHMARK_PROFILES[1]
                survey = _profile(
                    "health-expenditure-survey",
                    "National health expenditure survey",
                    "Survey estimates of current healthcare expenditure.",
                    "Current health expenditure",
                    "Current health expenditure per capita.",
                    "US dollars",
                )
                profiles = (hospital, survey)
                for profile in profiles:
                    database.save_dataset_record(
                        source="BENCH",
                        dataset_id=profile["dataset_id"],
                        title=profile["title"],
                        description=profile["description"],
                        source_url=profile["source_url"],
                        metadata=profile["metadata"],
                        is_latest_release=profile["is_latest_release"],
                    )
                    database.save_dataset_structure(
                        profile["structure"],
                        source="BENCH",
                    )
                    database.save_dataset_concepts(profile["concepts"])

                with (
                    patch(
                        "data_sources.retrieval.embedding.rank_dataset_candidates",
                        return_value=[
                            {
                                "source": "BENCH",
                                "dataset_id": "health-expenditure-survey",
                                "score": 0.99,
                                "best_concept_score": 0.99,
                                "semantic_concepts_scored": 1,
                            },
                            {
                                "source": "BENCH",
                                "dataset_id": "hospital-spending",
                                "score": 0.8,
                                "best_concept_score": 0.8,
                                "semantic_concepts_scored": 1,
                            },
                        ],
                    ),
                    patch(
                        "data_sources.retrieval.embedding.search_dataset_concepts",
                        side_effect=lambda query, top_k, source, dataset_ids,
                        instrumentation=None: [
                            {
                                **profile["concepts"][0],
                                "score": 0.9,
                            }
                            for profile in profiles
                            if profile["dataset_id"] in dataset_ids
                        ],
                    ) as scoped_concepts,
                ):
                    trace = DataSourceRetrieval(
                        [BenchmarkDataSource()]
                    ).retrieve(
                        "What were hospital spending and payments in the "
                        "United States?",
                        top_dataset_candidates=2,
                        concept_dataset_count=2,
                    )

                self.assertEqual(
                    trace["selected_dataset"]["dataset_id"],
                    "hospital-spending",
                )
                survey_check = next(
                    candidate
                    for candidate in trace["compatibility_evaluations"]
                    if candidate["dataset_id"] == "health-expenditure-survey"
                )
                self.assertFalse(survey_check["compatible"])
                self.assertTrue(any(
                    "hospital care" in reason
                    for reason in survey_check["rejection_reasons"]
                ))
                self.assertNotIn(
                    "health-expenditure-survey",
                    scoped_concepts.call_args.kwargs["dataset_ids"],
                )

    def test_two_stage_retrieval_selects_only_compatible_candidate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(database, "DB", f"{temp_dir}\\agent.db"):
                for profile in BENCHMARK_PROFILES:
                    database.save_dataset_record(
                        source="BENCH",
                        dataset_id=profile["dataset_id"],
                        title=profile["title"],
                        description=profile["description"],
                        source_url=profile["source_url"],
                        metadata=profile["metadata"],
                        is_latest_release=profile["is_latest_release"],
                    )
                    database.save_dataset_structure(
                        profile["structure"],
                        source="BENCH",
                    )
                    database.save_dataset_concepts(profile["concepts"])

                retrieval = DataSourceRetrieval(
                    [BenchmarkDataSource()]
                )
                query = (
                    "How many inpatient hospital discharges in Canada "
                    "during 2024?"
                )
                with (
                    patch(
                        "data_sources.retrieval.embedding.rank_dataset_candidates",
                        return_value=[
                            {
                                "source": "BENCH",
                                "dataset_id": profile["dataset_id"],
                                "score": 0.9 - index * 0.01,
                                "best_concept_score": 0.9 - index * 0.01,
                                "semantic_concepts_scored": 1,
                            }
                            for index, profile in enumerate(
                                BENCHMARK_PROFILES
                            )
                        ],
                    ),
                    patch(
                        "data_sources.retrieval.embedding.search_dataset_concepts",
                        side_effect=lambda query, top_k, source, dataset_ids,
                        instrumentation=None: [
                            {
                                **profile["concepts"][0],
                                "score": 0.9,
                            }
                            for profile in BENCHMARK_PROFILES
                            if profile["dataset_id"] in dataset_ids
                        ],
                    ) as scoped_concepts,
                ):
                    trace = retrieval.retrieve(
                        query,
                        top_dataset_candidates=10,
                        concept_dataset_count=6,
                    )

                self.assertEqual(
                    trace["selected_dataset"]["dataset_id"],
                    "utilization-count",
                )
                self.assertEqual(trace["selected_source"], "BENCH")
                self.assertEqual(
                    trace["selected_concept"]["label"],
                    "Total inpatient discharges",
                )
                self.assertGreaterEqual(
                    len(trace["stage_1_dataset_candidates"]),
                    5,
                )
                self.assertTrue(any(
                    candidate["dataset_id"] == "hospital-spending"
                    and not candidate["compatible"]
                    for candidate in trace["stage_1_dataset_candidates"]
                ))
                scoped_ids = scoped_concepts.call_args.kwargs["dataset_ids"]
                self.assertNotIn("hospital-spending", scoped_ids)

    def test_currentness_rejects_archive_and_archive_intent_rejects_current(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(database, "DB", f"{temp_dir}\\agent.db"):
                profiles = (
                    BENCHMARK_PROFILES[0],
                    BENCHMARK_PROFILES[-1],
                )
                for profile in profiles:
                    database.save_dataset_record(
                        source="BENCH",
                        dataset_id=profile["dataset_id"],
                        title=profile["title"],
                        description=profile["description"],
                        source_url=profile["source_url"],
                        metadata=profile["metadata"],
                        is_latest_release=profile["is_latest_release"],
                    )
                    database.save_dataset_structure(
                        profile["structure"],
                        source="BENCH",
                    )
                    database.save_dataset_concepts(profile["concepts"])

                retrieval = DataSourceRetrieval(
                    [BenchmarkDataSource()]
                )
                for question, selected_id, rejected_id in (
                    (
                        "Show current inpatient utilization in Canada",
                        "utilization-count",
                        "archived-utilization",
                    ),
                    (
                        "Show archived inpatient utilization in Canada",
                        "archived-utilization",
                        "utilization-count",
                    ),
                ):
                    with (
                        patch(
                            "data_sources.retrieval.embedding.rank_dataset_candidates",
                            return_value=[
                                {
                                    "source": "BENCH",
                                    "dataset_id": profile["dataset_id"],
                                    "score": 0.9,
                                    "best_concept_score": 0.9,
                                    "semantic_concepts_scored": 1,
                                }
                                for profile in profiles
                            ],
                        ),
                        patch(
                            "data_sources.retrieval.embedding.search_dataset_concepts",
                            side_effect=lambda query, top_k, source, dataset_ids,
                            instrumentation=None: [
                                {
                                    **profile["concepts"][0],
                                    "score": 0.9,
                                }
                                for profile in profiles
                                if profile["dataset_id"] in dataset_ids
                            ],
                        ),
                    ):
                        trace = retrieval.retrieve(
                            question,
                            top_dataset_candidates=5,
                            concept_dataset_count=5,
                        )
                    self.assertEqual(
                        trace["selected_dataset"]["dataset_id"],
                        selected_id,
                    )
                    self.assertTrue(any(
                        candidate["dataset_id"] == rejected_id
                        and not candidate["compatible"]
                        for candidate in trace["stage_1_dataset_candidates"]
                    ))

    def test_dataset_ranking_accepts_legacy_local_embedding_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(database, "DB", f"{temp_dir}/agent.db"):
                for profile in BENCHMARK_PROFILES[:2]:
                    database.save_dataset_concepts(profile["concepts"])

                metrics = {}
                with patch(
                    "embedding.embed_text",
                    return_value=[1.0, 0.0],
                ):
                    ranked = embedding.rank_dataset_candidates(
                        "inpatient hospital utilization",
                        source="BENCH",
                        instrumentation=metrics,
                    )

        self.assertEqual(len(ranked), 2)
        self.assertEqual(metrics["vectors_compared"], 2)

    def test_trace_reports_retrieval_timings_and_vector_counts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(database, "DB", f"{temp_dir}\\agent.db"):
                profiles = BENCHMARK_PROFILES[:2]
                for profile in profiles:
                    database.save_dataset_record(
                        source="BENCH",
                        dataset_id=profile["dataset_id"],
                        title=profile["title"],
                        description=profile["description"],
                        source_url=profile["source_url"],
                        metadata=profile["metadata"],
                        is_latest_release=profile["is_latest_release"],
                    )
                    database.save_dataset_structure(
                        profile["structure"],
                        source="BENCH",
                    )
                    database.save_dataset_concepts(profile["concepts"])

                with patch(
                    "embedding.embed_text",
                    return_value=[1.0, 0.0],
                ):
                    trace = DataSourceRetrieval(
                        [BenchmarkDataSource()]
                    ).retrieve(
                        "How many inpatient hospital discharges during 2024?",
                        top_dataset_candidates=2,
                        concept_dataset_count=2,
                    )

        instrumentation = trace["instrumentation"]
        self.assertIn(
            "total_elapsed",
            instrumentation["stage_timings_seconds"],
        )
        self.assertIn(
            "question_parsing",
            instrumentation["stage_timings_seconds"],
        )
        self.assertIn(
            "dataset_ranking",
            instrumentation["stage_timings_seconds"],
        )
        self.assertIn(
            "compatibility_checks",
            instrumentation["stage_timings_seconds"],
        )
        self.assertGreater(
            instrumentation["query_embedding_seconds"],
            0,
        )
        self.assertEqual(
            instrumentation["counts"]["datasets_considered"],
            2,
        )
        self.assertEqual(
            instrumentation["counts"]["concept_vectors_compared"],
            3,
        )
        self.assertEqual(
            instrumentation["counts"]["concepts_returned"],
            1,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
