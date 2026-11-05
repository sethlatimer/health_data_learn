import time

from data_sources.documents import DocumentIngestionService
from data_sources.literature.pubmed import PubMedSource
from data_sources.retrieval import DataSourceRetrieval


def _article_evidence(article):
    study_type = article.get("factual_relevance", {}).get(
        "study_type",
        article.get("study_metadata", {}).get("study_type", ["unknown"]),
    )
    return {
        "evidence_id": f"pubmed:{article['pmid']}",
        "source": "PUBMED",
        "record_type": "academic_literature",
        "title": article["title"],
        "abstract": article.get("abstract"),
        "reported_findings": article.get("abstract"),
        "study_type": study_type,
        "study_metadata": article.get("study_metadata", {}),
        "identifiers": {
            "pmid": article["pmid"],
            "pmcid": article.get("pmcid"),
            "doi": article.get("doi"),
        },
        "publication_year": article.get("publication_year"),
        "journal": article.get("journal"),
        "evidence_access_level": article.get("evidence_access_level", "metadata"),
        "full_text_available": article.get("full_text_available", False),
        "full_text_url": article.get("full_text_url"),
        "full_text_reuse_status": article.get(
            "full_text_reuse_status",
            "not_assessed",
        ),
        "causal_inference_status": article.get(
            "causal_inference_status",
            "not_assessed",
        ),
        "scores": article.get("scores", {}),
        "provenance": {
            "source_url": article["source_url"],
            "pmid": article["pmid"],
            "evidence_access_level": article.get(
                "evidence_access_level",
                "metadata",
            ),
        },
    }


def build_evidence_package(
    question,
    *,
    structured_retrieval=None,
    document_service=None,
    literature_source=None,
    top_dataset_candidates=20,
    concept_dataset_count=5,
    top_concepts=10,
    top_documents=6,
    top_articles=10,
    embedding_model="nomic-local",
    analysis_result=None,
):
    """Retrieve distinct evidence layers without generating an LLM narrative."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    total_started = time.perf_counter()
    retrieval = structured_retrieval or DataSourceRetrieval()
    documents = document_service or DocumentIngestionService(
        embedding_model=embedding_model
    )
    literature = literature_source or PubMedSource()

    structured_started = time.perf_counter()
    structured_trace = retrieval.retrieve(
        question,
        top_dataset_candidates=top_dataset_candidates,
        concept_dataset_count=concept_dataset_count,
        top_concepts=top_concepts,
    )
    structured_seconds = time.perf_counter() - structured_started

    concept_compatible_candidates = structured_trace.get(
        "compatible_dataset_candidates",
        [],
    )
    dataset_candidates = [
        candidate
        for candidate in structured_trace.get("stage_1_dataset_candidates", [])
        if candidate.get("dataset_compatible") is True
    ]
    if not dataset_candidates:
        dataset_candidates = concept_compatible_candidates
    dataset_ids = list(dict.fromkeys(
        item["dataset_id"]
        for item in dataset_candidates
        if item.get("dataset_id")
    ))
    selected_dataset = structured_trace.get("selected_dataset") or {}
    if selected_dataset.get("dataset_id") and (
        selected_dataset["dataset_id"] not in dataset_ids
    ):
        dataset_ids.insert(0, selected_dataset["dataset_id"])
    dataset_ids = dataset_ids[:top_dataset_candidates]

    document_started = time.perf_counter()
    official_document_evidence = []
    if dataset_ids:
        per_dataset_limit = max(1, top_documents // len(dataset_ids))
        for dataset_id in dataset_ids:
            profile = next(
                (
                    candidate for candidate in dataset_candidates
                    if candidate.get("dataset_id") == dataset_id
                ),
                {},
            )
            hits = documents.search(
                question,
                top_k=per_dataset_limit,
                source=profile.get("source"),
                dataset_id=dataset_id,
            )
            official_document_evidence.extend(hits)
    if not official_document_evidence:
        official_document_evidence = documents.search(
            question,
            top_k=top_documents,
        )
    official_document_evidence.sort(
        key=lambda item: item.get("score", 0.0),
        reverse=True,
    )
    official_document_evidence = official_document_evidence[:top_documents]
    document_seconds = time.perf_counter() - document_started

    literature_started = time.perf_counter()
    article_results = literature.search(
        question,
        limit=top_articles,
        embedding_model=embedding_model,
    )
    literature_seconds = time.perf_counter() - literature_started
    literature_evidence = [
        _article_evidence(article)
        for article in article_results
    ]

    deterministic_evidence = []
    if analysis_result is not None:
        deterministic_evidence.append({
            "evidence_id": "structured_analysis:0",
            "source": analysis_result.get("source"),
            "record_type": "deterministic_analysis",
            "result": analysis_result,
            "provenance": analysis_result.get("provenance"),
            "calculation_method": analysis_result.get("analysis_method"),
        })

    return {
        "question": question,
        "generated_narrative": None,
        "evidence_layers": {
            "computed_structured_findings": deterministic_evidence,
            "structured_data_candidates": {
                "selected_source": structured_trace.get("selected_source"),
                "selected_dataset": selected_dataset,
                "selected_concept": structured_trace.get("selected_concept"),
                "dataset_candidates": dataset_candidates,
                "concept_compatible_candidates": concept_compatible_candidates,
                "compatible_candidates": concept_compatible_candidates,
                "trace": structured_trace,
            },
            "official_documentation": official_document_evidence,
            "academic_literature": literature_evidence,
            "llm_interpretation": [],
            "possible_actions": [],
        },
        "claim_provenance": [
            {
                "evidence_type": "deterministic_analysis",
                "source": item.get("source")
                or (item.get("provenance") or {}).get("source"),
                "dataset_id": (item.get("provenance") or {}).get("dataset_id"),
                "source_url": (item.get("provenance") or {}).get("source_url"),
                "analysis_provenance": item.get("provenance"),
            }
            for item in deterministic_evidence
        ] + [
            item["provenance"] for item in literature_evidence
        ] + [
            {
                "evidence_type": "official_documentation",
                "source": item.get("source"),
                "dataset_id": item.get("dataset_id"),
                "document_id": item.get("document_id"),
                "chunk_id": item.get("chunk_id"),
                "source_url": item.get("source_url"),
                "page_number": item.get("page_number"),
                "section": item.get("section"),
            }
            for item in official_document_evidence
        ],
        "synthesis_constraints": {
            "structured_calculations_are_authoritative_for_numeric_findings": True,
            "literature_does_not_modify_deterministic_calculations": True,
            "observational_evidence_is_association_only": True,
            "contradictory_and_null_results_are_not_filtered": True,
            "full_text_reuse_requires_license_verification": True,
            "llm_narrative_generated": False,
        },
        "retrieval_trace": {
            "stage_timings_seconds": {
                "structured_retrieval": structured_seconds,
                "official_document_retrieval": document_seconds,
                "literature_retrieval": literature_seconds,
                "total_elapsed": time.perf_counter() - total_started,
            },
            "structured_retrieval": structured_trace.get("instrumentation", {}),
            "literature_retrieval": literature.last_trace,
            "counts": {
                "structured_dataset_candidates": len(dataset_candidates),
                "official_documents": len(official_document_evidence),
                "academic_articles": len(literature_evidence),
                "deterministic_findings": len(deterministic_evidence),
            },
        },
    }
