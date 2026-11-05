import argparse
import hashlib
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_sources.literature.pubmed import PubMedSource, _score_article


BENCHMARK_PATH = Path(__file__).with_name("literature_benchmark.jsonl")


def load_cases(path=BENCHMARK_PATH):
    with path.open(encoding="utf-8") as benchmark:
        return [json.loads(line) for line in benchmark if line.strip()]


def _article_text(article):
    return " ".join(
        [
            article.get("title", ""),
            article.get("abstract") or "",
            " ".join(article.get("keywords", [])),
            " ".join(article.get("mesh_terms", [])),
        ]
    ).casefold()


def _case_result(case, articles, elapsed, error=None):
    top_articles = articles[:5]
    relevant = [
        article
        for article in top_articles
        if any(term.casefold() in _article_text(article)
               for term in case["expected_any_terms"])
    ]
    return {
        "id": case["id"],
        "question": case["question"],
        "top_articles": [
            {
                "title": article["title"],
                "pmid": article["pmid"],
                "pmcid": article.get("pmcid"),
                "doi": article.get("doi"),
                "publication_year": article.get("publication_year"),
                "journal": article.get("journal"),
                "study_type": article.get("factual_relevance", {}).get(
                    "study_type",
                    article.get("study_metadata", {}).get("study_type", ["unknown"]),
                ),
                "semantic_relevance": article.get("scores", {}).get(
                    "semantic_relevance"
                ),
                "population_fit": article.get("scores", {}).get(
                    "population_relevance"
                ),
                "intervention_fit": article.get("scores", {}).get(
                    "intervention_relevance"
                ),
                "outcome_fit": article.get("scores", {}).get(
                    "outcome_relevance"
                ),
                "publication_year_fit": article.get("scores", {}).get(
                    "publication_year_relevance"
                ),
                "evidence_access_level": article.get(
                    "evidence_access_level",
                    "metadata",
                ),
                "full_text_available": article.get("full_text_available", False),
                "full_text_reuse_status": article.get(
                    "full_text_reuse_status",
                    "not_assessed",
                ),
                "rank_score": article.get("rank_score"),
                "ranking_formula": article.get("ranking_formula"),
                "why_selected": article.get("why_selected"),
            }
            for article in top_articles
        ],
        "expected_result_checks": {
            "expected_any_terms": case["expected_any_terms"],
            "relevant_articles_at_5": len(relevant),
            "topic_recall_at_5": bool(relevant),
        },
        "stage_timings_seconds": (
            articles[0].get("retrieval_trace", {}).get("stage_timings_seconds", {})
            if articles else {}
        ),
        "total_latency_seconds": elapsed,
        "database_records_examined": len(articles),
        "vectors_compared": len(articles),
        "status": "ERROR" if error else ("PASS" if relevant else "FAIL"),
        "error": error,
    }


def _fixture_vector(text, dimensions=256):
    vector = [0.0] * dimensions
    for token in re.findall(r"[a-z0-9]+", text.casefold()):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        vector[index] += 1.0 if digest[4] & 1 else -1.0
    magnitude = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / magnitude for value in vector]


def _fixture_search(case, cases):
    query_vector = _fixture_vector(case["question"])
    candidates = []
    for candidate_case in cases:
        if candidate_case["id"] == case["id"]:
            title = "Fixture study: " + ", ".join(candidate_case["expected_any_terms"])
            abstract = (
                f"{candidate_case['question']} "
                f"Expected topic: {', '.join(candidate_case['expected_any_terms'])}. "
                "Fixture evidence record; not a real publication."
            )
        else:
            title = "Fixture comparator: " + ", ".join(
                candidate_case["expected_any_terms"]
            )
            abstract = (
                f"{candidate_case['question']} "
                f"Topic terms: {', '.join(candidate_case['expected_any_terms'])}. "
                "Fixture evidence record; not a real publication."
            )
        semantic_text = f"Title: {title}\nAbstract: {abstract}"
        article = {
            "pmid": f"fixture-{candidate_case['id']}",
            "pmcid": None,
            "doi": None,
            "title": title,
            "abstract": abstract,
            "publication_year": 2025,
            "journal": "Fixture-only journal",
            "authors": [],
            "publication_types": ["Observational Study"],
            "keywords": candidate_case["expected_any_terms"],
            "mesh_terms": [],
            "study_metadata": {"study_type": ["observational"]},
            "source_url": f"fixture://pubmed/{candidate_case['id']}",
            "full_text_available": False,
            "full_text_reuse_status": "not_available_in_fixture",
            "evidence_access_level": "abstract",
            "semantic_text": semantic_text,
            "embedding": _fixture_vector(semantic_text),
        }
        candidates.append(_score_article(
            case["question"],
            article,
            query_vector,
        ))
    candidates.sort(key=lambda result: result["rank_score"], reverse=True)
    return candidates[:10]


def run_benchmark(mode="live", limit=10, source=None):
    cases = load_cases()
    if mode not in {"fixture", "live"}:
        raise ValueError("Benchmark runner supports fixture and live modes")
    if mode == "live":
        source = source or PubMedSource()
    results = []
    for case in cases:
        started = time.perf_counter()
        try:
            if mode == "fixture":
                articles = _fixture_search(case, cases)[:limit]
                for article in articles:
                    article["retrieval_trace"] = {
                        "stage_timings_seconds": {
                            "fixture_ranking": 0.0,
                        },
                    }
            else:
                articles = source.search(case["question"], limit=limit)
            error = None
        except Exception as exception:
            articles = []
            error = f"{type(exception).__name__}: {exception}"
        results.append(_case_result(
            case,
            articles,
            time.perf_counter() - started,
            error,
        ))
        print(json.dumps(results[-1], ensure_ascii=False))

    successes = [
        item for item in results if item["status"] == "PASS"
    ]
    latencies = [item["total_latency_seconds"] for item in results]
    ordered_latencies = sorted(latencies)
    p95_index = max(0, math.ceil(0.95 * len(ordered_latencies)) - 1)
    report = {
        "mode": mode,
        "cases": len(results),
        "successful_cases": len(successes),
        "retrieval_success_rate": (
            len(successes) / len(results) if results else 0.0
        ),
        "topic_recall_at_5": (
            len(successes) / len(results) if results else 0.0
        ),
        "median_latency_seconds": (
            statistics.median(latencies) if latencies else None
        ),
        "p95_latency_seconds": (
            ordered_latencies[p95_index] if ordered_latencies else None
        ),
        "average_vectors_compared": (
            statistics.mean(item["vectors_compared"] for item in results)
            if results else 0.0
        ),
        "failed_cases": [
            {"id": item["id"], "status": item["status"], "error": item["error"]}
            for item in results if item["status"] != "PASS"
        ],
    }
    print(json.dumps({"aggregate_report": report}, ensure_ascii=False))
    return {"results": results, "aggregate_report": report}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fixture", "live"), default="live")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args(argv)
    if not 1 <= args.limit <= 100:
        parser.error("--limit must be between 1 and 100")
    run_benchmark(mode=args.mode, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
