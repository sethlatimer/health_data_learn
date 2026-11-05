import argparse
import json
import math
import os
import re
import sys
import statistics
import tempfile
import time
from unittest.mock import patch

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

import database
import embedding
from data_sources.retrieval import DataSourceRetrieval


DEFAULT_DEFINITIONS = os.path.join(
    os.path.dirname(__file__),
    "retrieval_benchmark.jsonl",
)


def _read_definitions(path):
    with open(path, encoding="utf-8") as definitions_file:
        cases = [
            json.loads(line)
            for line in definitions_file
            if line.strip()
        ]
    if not cases:
        raise ValueError(f"No benchmark cases found in {path}")
    return cases


def _one_hot(index, length):
    return [1.0 if position == index else 0.0 for position in range(length)]


def _fixture_profile(case, index, dimension_count):
    expected = case.get("expected", {})
    source = (expected.get("sources") or ["OECD"])[0].upper()
    dataset_id = f"fixture-{case['id']}"
    title_terms = expected.get("dataset_title_terms") or [
        case["id"].replace("-", " ")
    ]
    title = " ".join(title_terms).title()
    question = case["question"]
    measure_type = expected.get("measure_type")
    topic = expected.get("topic")
    concept_terms = expected.get("concept_terms") or [case["id"]]
    label = " ".join(concept_terms).title()
    if topic == "utilization":
        label += " Hospital Inpatient Utilization"
    elif topic == "spending":
        label += " Hospital Payment Amount"
    if measure_type == "count":
        label += " Count"
    elif measure_type == "rate":
        label += " Rate Per 1,000 Persons"
    elif measure_type == "amount":
        label += " Amount In Dollars"
    adjustment = expected.get("adjustment")
    if adjustment:
        label += f" {adjustment.title()}"
    semantic_text = f"{label}. {question}"
    latest = expected.get("currentness") != "archived"
    unit = (
        "Admissions per 1,000 persons"
        if measure_type == "rate"
        else "US dollars"
        if measure_type == "amount"
        else "Count"
    )
    structure = {
        "source": source,
        "dataset_id": dataset_id,
        "dataflow": {
            "id": dataset_id,
            "name": title,
            "description": f"{title}. {question}",
            "metadata": {},
        },
        "data_structure": {
            "id": f"DSD_{dataset_id}",
            "dimensions": [
                {
                    "id": "COUNTRY",
                    "name": "Country",
                    "type": "Dimension",
                    "position": 1,
                    "codes": [
                        {"id": "USA", "label": "United States"},
                        {"id": "CAN", "label": "Canada"},
                    ],
                },
                {
                    "id": "MEASURE",
                    "name": "Measure",
                    "type": "MeasureDimension",
                    "position": 2,
                    "codes": [{"id": "VALUE", "label": label}],
                },
                {
                    "id": "UNIT_MEASURE",
                    "name": "Unit of measure",
                    "type": "Dimension",
                    "position": 3,
                    "codes": [{"id": "UNIT", "label": unit}],
                },
                {
                    "id": "TIME_PERIOD",
                    "name": "Time period",
                    "type": "TimeDimension",
                    "position": 4,
                    "codes": [],
                },
            ],
        },
        "codelists": [],
    }
    concept = {
        "source": source,
        "dataset_id": dataset_id,
        "dimension_id": "MEASURE",
        "dimension_name": "Measure",
        "code": "VALUE",
        "label": label,
        "concept_role": "semantic",
        "semantic_text": semantic_text,
        "embedding": _one_hot(index, dimension_count),
        "embedding_model": "nomic-local",
        "metadata": {"description": semantic_text},
    }
    return {
        "source": source,
        "dataset_id": dataset_id,
        "title": title,
        "description": f"{title}. {question}",
        "source_url": f"https://fixture.invalid/{dataset_id}",
        "metadata": {
            "temporal": [{
                "startDate": "2018-01-01",
                "endDate": "2024-12-31",
            }],
            "is_latest_release": latest,
        },
        "is_latest_release": latest,
        "structure": structure,
        "concept": concept,
        "vector": _one_hot(index, dimension_count),
    }


def _fixture_runner(cases):
    temp_dir = tempfile.TemporaryDirectory(prefix="retrieval-benchmark-")
    database_path = os.path.join(temp_dir.name, "fixture.db")
    try:
        with patch.object(database, "DB", database_path):
            fixtures = [
                _fixture_profile(case, index, len(cases))
                for index, case in enumerate(cases)
            ]
            for fixture in fixtures:
                database.save_dataset_record(
                    source=fixture["source"],
                    dataset_id=fixture["dataset_id"],
                    title=fixture["title"],
                    description=fixture["description"],
                    source_url=fixture["source_url"],
                    metadata=fixture["metadata"],
                    is_latest_release=fixture["is_latest_release"],
                )
                database.save_dataset_structure(
                    fixture["structure"],
                    source=fixture["source"],
                )
                database.save_dataset_concepts([fixture["concept"]])

            for case, fixture in zip(cases, fixtures):
                with patch.object(
                    embedding,
                    "embed_text",
                    return_value=fixture["vector"],
                ):
                    trace = DataSourceRetrieval().retrieve(case["question"])
                yield trace
    finally:
        temp_dir.cleanup()


def _contains_any(text, terms):
    normalized = str(text or "").casefold()
    return any(str(term).casefold() in normalized for term in terms)


def _measure_types_in_text(text):
    markers = {
        "count": (
            "count",
            "number of",
            "number",
            "discharge",
            "discharges",
            "admission",
            "admissions",
            "visit",
            "visits",
            "beneficiaries",
            "events",
        ),
        "rate": (
            "rate",
            "rates",
            "per capita",
            "per person",
            "per 100",
            "per 1,000",
            "per 1000",
            "percent",
            "percentage",
            "proportion",
            "ratio",
        ),
        "amount": (
            "amount",
            "amounts",
            "spending",
            "payment",
            "payments",
            "cost",
            "costs",
            "charge",
            "charges",
            "expenditure",
            "dollar",
            "dollars",
        ),
    }
    detected = [
        measure_type
        for measure_type, terms in markers.items()
        if any(
            re.search(
                r"(?<![a-z0-9])"
                + r"\s+".join(re.escape(part) for part in term.split())
                + r"(?![a-z0-9])",
                str(text or "").casefold(),
            )
            for term in terms
        )
    ]
    explicit_count = any(
        re.search(
            r"(?<![a-z0-9])"
            + r"\s+".join(re.escape(part) for part in term.split())
            + r"(?![a-z0-9])",
            str(text or "").casefold(),
        )
        for term in ("count", "counts", "number of", "total number")
    )
    if "rate" in detected and not explicit_count:
        detected = [
            measure_type
            for measure_type in detected
            if measure_type != "count"
        ]
    return detected


def _check_case(case, trace):
    expected = case.get("expected", {})
    forbidden = case.get("forbidden", {})
    selected_dataset = trace.get("selected_dataset") or {}
    selected_concept = trace.get("selected_concept") or {}
    selected_text = " ".join((
        selected_concept.get("label", ""),
        selected_concept.get("semantic_text", ""),
    ))
    selected_title = selected_dataset.get("title", "")
    requirements = trace.get("requirements", {})
    checks = []

    def add(name, passed, actual=None, wanted=None):
        checks.append({
            "check": name,
            "pass": bool(passed),
            "actual": actual,
            "expected": wanted,
        })

    selected_checks = selected_concept.get("compatibility_checks", [])
    selected_dataset_candidate = next(
        (
            item
            for item in trace.get("stage_1_dataset_candidates", [])
            if item.get("source") == selected_dataset.get("source")
            and item.get("dataset_id") == selected_dataset.get("dataset_id")
        ),
        {},
    )
    dataset_checks = selected_dataset_candidate.get(
        "compatibility_checks",
        [],
    )

    if "sources" in expected:
        add(
            "source",
            trace.get("selected_source") in expected["sources"],
            trace.get("selected_source"),
            expected["sources"],
        )
    if "dataset_title_terms" in expected:
        terms = expected["dataset_title_terms"]
        add(
            "dataset",
            _contains_any(selected_title, terms),
            selected_title or None,
            terms,
        )
    for field in (
        "topic",
        "measure_type",
        "measure_family",
        "quantity_type",
        "measure_domain",
        "financial_measure",
        "provider_type",
        "adjustment",
        "currentness",
    ):
        if field in expected:
            if field == "currentness":
                evidence = next(
                    (
                        check
                        for check in dataset_checks
                        if check["characteristic"] == "currentness"
                    ),
                    {},
                )
            elif field in {
                "measure_family",
                "quantity_type",
                "measure_domain",
                "financial_measure",
                "provider_type",
            }:
                characteristic = {
                    "measure_domain": "measure_domain",
                }.get(field, field)
                evidence = next(
                    (
                        check
                        for check in selected_checks
                        if check["characteristic"] == characteristic
                    ),
                    {},
                )
            else:
                evidence = next(
                    (
                        check
                        for check in selected_checks
                        if check["characteristic"] == field
                    ),
                    {},
                )
            supported = (
                evidence.get("supported") is True
                if                 field in {
                    "topic",
                    "measure_type",
                    "measure_family",
                    "quantity_type",
                    "measure_domain",
                    "financial_measure",
                    "provider_type",
                    "adjustment",
                    "currentness",
                }
                else False
            )
            add(
                field,
                requirements.get(field) == expected[field] and supported,
                {
                    "parsed": requirements.get(field),
                    "selected_measure_supported": supported,
                },
                expected[field],
            )
    if "concept_terms" in expected:
        terms = expected["concept_terms"]
        add(
            "concept",
            bool(selected_concept)
            and _contains_any(selected_text, terms),
            selected_concept.get("label") or None,
            terms,
        )

    candidates = trace.get("stage_1_dataset_candidates", [])
    concepts = trace.get("stage_2_concept_candidates", [])
    if "sources" in expected:
        add(
            "source_recall_at_5",
            any(
                item.get("source") in expected["sources"]
                for item in candidates[:5]
            ),
            [item.get("source") for item in candidates[:5]],
            expected["sources"],
        )
    if "dataset_title_terms" in expected:
        terms = expected["dataset_title_terms"]
        add(
            "dataset_recall_at_5",
            any(
                _contains_any(item.get("title"), terms)
                for item in candidates[:5]
            ),
            [
                item.get("title")
                for item in candidates[:5]
            ],
            terms,
        )
    if "concept_terms" in expected:
        terms = expected["concept_terms"]
        add(
            "concept_recall_at_5",
            any(
                _contains_any(
                    f"{item.get('label', '')} {item.get('semantic_text', '')}",
                    terms,
                )
                for item in concepts[:5]
            ),
            [item.get("label") for item in concepts[:5]],
            terms,
        )

    if "topic" in forbidden:
        add(
            "forbidden_topic",
            requirements.get("topic") != forbidden["topic"],
            requirements.get("topic"),
            f"not {forbidden['topic']}",
        )
    if forbidden.get("concept_terms"):
        terms = forbidden["concept_terms"]
        add(
            "forbidden_concept",
            not _contains_any(selected_text, terms),
            selected_concept.get("label") or None,
            f"not any of {terms}",
        )
    if forbidden.get("dataset_title_terms"):
        terms = forbidden["dataset_title_terms"]
        add(
            "forbidden_dataset",
            not _contains_any(selected_title, terms),
            selected_title or None,
            f"not any of {terms}",
        )
    for field in ("currentness", "adjustments"):
        values = forbidden.get(field)
        if values is None:
            continue
        if isinstance(values, str):
            values = [values]
        if field == "currentness":
            evidence = next(
                (
                    check
                    for check in dataset_checks
                    if check["characteristic"] == "currentness"
                ),
                {},
            )
            selected_value = (
                evidence.get("requested")
                if evidence.get("supported") is True
                else "unverified or rejected"
            )
        else:
            selected_value = _measure_types_in_text(selected_text)
        add(
            f"forbidden_{field}",
            (
                selected_value not in values
                if field == "currentness"
                else not any(value in selected_value for value in values)
            ),
            selected_value,
            f"not any of {values}",
        )
    if forbidden.get("measure_types"):
        detected = _measure_types_in_text(selected_text)
        add(
            "forbidden_measure_type",
            not any(
                value in detected
                for value in forbidden["measure_types"]
            ),
            detected,
            f"not any of {forbidden['measure_types']}",
        )

    passed = all(check["pass"] for check in checks)
    return checks, passed


def _percentile(values, percentile):
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _run(cases, mode, confirm_record=False, version=None, output=None):
    if mode == "fixture":
        traces = iter(_fixture_runner(cases))
        source_label = "fixture"
    else:
        traces = None
        source_label = "live"

    results = []
    for case in cases:
        case_started = time.perf_counter()
        try:
            trace = (
                next(traces)
                if mode == "fixture"
                else DataSourceRetrieval().retrieve(case["question"])
            )
            checks, passed = _check_case(case, trace)
            error = None
        except Exception as exception:
            trace = {}
            checks = [{
                "check": "retrieval_exception",
                "pass": False,
                "actual": f"{type(exception).__name__}: {exception}",
                "expected": "retrieval completes",
            }]
            passed = False
            error = checks[0]["actual"]
        instrumentation = trace.get("instrumentation", {})
        row = {
            "test_id": case["id"],
            "question": case["question"],
            "top_dataset_candidates": [
                {
                    "source": item["source"],
                    "dataset_id": item["dataset_id"],
                    "title": item["title"],
                    "score": item["candidate_score"],
                    "semantic_score": item["semantic_score"],
                    "compatibility": item["compatible"],
                    "rejection_reasons": item["rejection_reasons"],
                }
                for item in trace.get("stage_1_dataset_candidates", [])[:10]
            ],
            "top_concept_candidates": [
                {
                    "source": item["source"],
                    "dataset_id": item["dataset_id"],
                    "label": item["label"],
                    "score": item["score"],
                    "compatible": item["compatible"],
                    "rejection_reasons": item["rejection_reasons"],
                    "measure_classification": item.get(
                        "measure_classification"
                    ),
                }
                for item in trace.get("stage_2_concept_candidates", [])
            ],
            "compatibility_decisions": [
                {
                    "source": item["source"],
                    "dataset_id": item["dataset_id"],
                    "compatible": item["compatible"],
                    "checks": [
                        {
                            "characteristic": check["characteristic"],
                            "supported": check["supported"],
                            "detail": check.get("detail"),
                        }
                        for check in item["compatibility_checks"]
                    ],
                    "rejection_reasons": item["rejection_reasons"],
                }
                for item in trace.get("stage_1_dataset_candidates", [])[:10]
            ],
            "selected": {
                "source": trace.get("selected_source"),
                "dataset": trace.get("selected_dataset"),
                "concept": trace.get("selected_concept"),
            },
            "expected_result_checks": checks,
            "stage_timings_seconds": instrumentation.get(
                "stage_timings_seconds",
                {},
            ),
            "total_latency_seconds": instrumentation.get(
                "stage_timings_seconds",
                {},
            ).get("total_elapsed"),
            "database_rows_examined": {
                "dataset_structures": instrumentation.get(
                    "database", {}
                ).get("dataset_structures_loaded", 0),
                "semantic_concepts": instrumentation.get(
                    "database", {}
                ).get("semantic_concept_rows_loaded", 0),
                "vectors_loaded": instrumentation.get(
                    "counts", {}
                ).get("concept_vectors_loaded", 0),
                "vectors_compared": instrumentation.get(
                    "counts", {}
                ).get("concept_vectors_compared", 0),
            },
            "database_timings": instrumentation.get("database", {}),
            "cache_usage": instrumentation.get("cache_usage", {}),
            "retrieval_counts": instrumentation.get("counts", {}),
            "query_embedding_seconds": instrumentation.get(
                "query_embedding_seconds"
            ),
            "sqlite_vector_loading_seconds": instrumentation.get(
                "sqlite_vector_loading_seconds"
            ),
            "cosine_similarity_ranking_seconds": instrumentation.get(
                "cosine_similarity_ranking_seconds"
            ),
            "embedding_calls": instrumentation.get("embedding_calls", []),
            "status": "PASS" if passed else "FAIL",
        }
        if error is not None:
            row["error"] = error
            row["total_latency_seconds"] = time.perf_counter() - case_started
        if mode == "record":
            row["trace"] = trace
        results.append(row)
        print(json.dumps(row, ensure_ascii=False))

    latency = [
        row["total_latency_seconds"]
        for row in results
        if row["total_latency_seconds"] is not None
    ]
    recall_checks = {
        name: [
            check["pass"]
            for row in results
            for check in row["expected_result_checks"]
            if check["check"] == name
        ]
        for name in (
            "source_recall_at_5",
            "dataset_recall_at_5",
            "concept_recall_at_5",
        )
    }
    forbidden_failures = [
        check
        for row in results
        for check in row["expected_result_checks"]
        if check["check"].startswith("forbidden_") and not check["pass"]
    ]
    failures_by_stage = {}
    for row in results:
        for check in row["expected_result_checks"]:
            if not check["pass"]:
                if check["check"] == "retrieval_exception":
                    stage = "retrieval"
                elif check["check"] in {"source", "dataset", "concept"}:
                    stage = "selection"
                elif check["check"].startswith("forbidden_"):
                    stage = "forbidden_substitution"
                elif check["check"] in {
                    "topic",
                    "measure_type",
                    "measure_family",
                    "quantity_type",
                    "measure_domain",
                    "financial_measure",
                    "adjustment",
                    "currentness",
                }:
                    actual = check.get("actual")
                    stage = (
                        "compatibility"
                        if isinstance(actual, dict)
                        and actual.get("parsed") == check["expected"]
                        and not actual.get("selected_measure_supported")
                        else "requirement_parsing"
                    )
                else:
                    stage = "candidate_recall"
                failures_by_stage[stage] = (
                    failures_by_stage.get(stage, 0) + 1
                )
    vector_counts = [
        row["database_rows_examined"]["vectors_compared"]
        for row in results
    ]
    aggregate = {
        "mode": source_label,
        "benchmark_cases": len(results),
        "retrieval_success_rate": (
            sum(row["status"] == "PASS" for row in results)
            / len(results)
            if results else 0.0
        ),
        "expected_source_recall_at_5": (
            sum(recall_checks["source_recall_at_5"])
            / len(recall_checks["source_recall_at_5"])
            if recall_checks["source_recall_at_5"] else None
        ),
        "expected_dataset_recall_at_5": (
            sum(recall_checks["dataset_recall_at_5"])
            / len(recall_checks["dataset_recall_at_5"])
            if recall_checks["dataset_recall_at_5"] else None
        ),
        "expected_concept_recall_at_5": (
            sum(recall_checks["concept_recall_at_5"])
            / len(recall_checks["concept_recall_at_5"])
            if recall_checks["concept_recall_at_5"] else None
        ),
        "forbidden_substitution_failures": len(forbidden_failures),
        "median_latency_seconds": (
            statistics.median(latency) if latency else None
        ),
        "p95_latency_seconds": _percentile(latency, 0.95),
        "slowest_cases": sorted(
            (
                {
                    "test_id": row["test_id"],
                    "latency_seconds": row["total_latency_seconds"],
                    "stage_timings_seconds": row["stage_timings_seconds"],
                }
                for row in results
            ),
            key=lambda row: row["latency_seconds"] or 0,
            reverse=True,
        )[:5],
        "average_vectors_compared": (
            statistics.mean(vector_counts) if vector_counts else 0
        ),
        "failures_grouped_by_stage": failures_by_stage,
    }
    report = {
        "benchmark_version": version,
        "aggregate": aggregate,
        "cases": results,
    }
    print(json.dumps({"aggregate": aggregate}, ensure_ascii=False))
    if output:
        with open(output, "w", encoding="utf-8") as output_file:
            for row in results:
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            output_file.write(
                json.dumps({"aggregate": aggregate}, ensure_ascii=False)
                + "\n"
            )
    if mode == "record":
        if not confirm_record:
            raise ValueError(
                "Record mode requires --confirm-validated after manually "
                "validating the live results"
            )
        if not version:
            raise ValueError("Record mode requires --version")
        fixture_dir = os.path.join(os.path.dirname(__file__), "fixtures")
        os.makedirs(fixture_dir, exist_ok=True)
        fixture_path = os.path.join(
            fixture_dir,
            f"retrieval_fixture_v{version}.jsonl",
        )
        with open(fixture_path, "w", encoding="utf-8") as fixture_file:
            for row in results:
                if row["status"] == "PASS":
                    fixture_file.write(
                        json.dumps({
                            "benchmark_version": version,
                            "validated": True,
                            "test_id": row["test_id"],
                            "question": row["question"],
                            "trace": row["trace"],
                        }, ensure_ascii=False)
                        + "\n"
                    )
        print(json.dumps({
            "validated_fixture": fixture_path,
            "recorded_cases": sum(
                row["status"] == "PASS" for row in results
            ),
            "withheld_failed_cases": sum(
                row["status"] != "PASS" for row in results
            ),
        }))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("fixture", "live", "record"),
        default="fixture",
    )
    parser.add_argument("--definitions", default=DEFAULT_DEFINITIONS)
    parser.add_argument("--output")
    parser.add_argument("--version")
    parser.add_argument(
        "--confirm-validated",
        action="store_true",
        help="Confirm live record results were explicitly validated",
    )
    args = parser.parse_args()
    if args.mode == "record" and (
        not args.confirm_validated or not args.version
    ):
        parser.error(
            "record mode requires --confirm-validated and --version "
            "after manually validating the live benchmark"
        )
    cases = _read_definitions(args.definitions)
    _run(
        cases,
        args.mode,
        confirm_record=args.confirm_validated,
        version=args.version,
        output=args.output,
    )


if __name__ == "__main__":
    main()
