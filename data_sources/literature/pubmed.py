import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
from datetime import date
from urllib.parse import quote

import requests

import database
import embedding
from data_sources.literature.base import LiteratureSource


EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
MAX_SEARCH_RESULTS = 100
REQUEST_TIMEOUT = 30
_POPULATION_TERMS = (
    "older adults", "older people", "adults", "children", "pediatric",
    "medicare", "rural", "urban", "low income", "underserved",
    "patients", "beneficiaries", "women", "veterans",
)
_INTERVENTION_TERMS = (
    "hospital at home", "home based", "home-based", "community health worker",
    "remote monitoring", "telehealth", "care management", "case management",
    "medication management", "integrated behavioral health",
    "food as medicine", "food insecurity", "primary care payment",
    "value based care", "palliative care", "hospice",
)
_OUTCOME_TERMS = (
    "utilization", "admission", "hospitalization", "readmission",
    "emergency department", "mortality", "diabetes", "glycemic",
    "cost", "spending", "access", "quality of life", "health outcome",
)
_SEARCH_EXPANSIONS = {
    "hospital at home": (
        "hospital at home", "home hospital", "acute care at home",
        "hospital-level care at home", "hospital level at home",
    ),
    "home-based": (
        "home-based care", "home based care", "hospital at home",
        "acute care at home", "hospital-level care at home",
    ),
    "home based": (
        "home-based care", "home based care", "hospital at home",
        "acute care at home", "hospital-level care at home",
    ),
    "community health worker": (
        "community health worker", "community health workers",
        "community health aide",
    ),
    "remote monitoring": (
        "remote patient monitoring", "remote monitoring", "telemonitoring",
    ),
    "behavioral health integration": (
        "integrated behavioral health", "behavioral health integration",
        "collaborative care",
    ),
    "food as medicine": (
        "food as medicine", "food prescription", "medically tailored meals",
    ),
    "utilization": (
        "healthcare utilization", "health care utilization",
        "health services use", "hospital use",
    ),
    "admission": (
        "hospital admission", "avoidable admission", "hospitalization",
        "hospital admissions",
    ),
    "hospitalization": (
        "hospitalization", "hospital admissions", "inpatient admission",
    ),
    "readmission": (
        "hospital readmission", "readmission", "readmissions",
    ),
    "primary care": ("primary care", "family practice", "general practice"),
    "rural": ("rural health", "rural healthcare", "rural health services"),
    "telehealth": ("telehealth", "telemedicine", "virtual care"),
    "diabetes": ("diabetes", "diabetic", "glycemic control"),
    "medication management": (
        "medication management", "pharmacist-led care", "medication review",
    ),
    "care management": (
        "care management", "case management", "disease management",
    ),
    "value based care": (
        "value-based care", "value based care", "value-based payment",
    ),
    "palliative care": ("palliative care", "hospice care", "end-of-life care"),
    "systematic review": ("systematic review", "meta-analysis"),
}
_QUERY_NOISE = {
    "a", "an", "and", "are", "as", "be", "could", "did", "do", "does",
    "evidence", "for", "how", "idea", "in", "it", "might", "of", "on",
    "or", "outcomes", "supports", "that", "the", "this", "to", "what",
    "where", "which", "with", "would", "evaluate", "evaluated", "reduce",
    "reduces", "reducing", "contradict", "contradicts",
}
_STUDY_TYPE_PATTERNS = (
    ("systematic_review", ("systematic review", "meta-analysis", "meta analysis")),
    ("randomized_trial", ("randomized controlled trial", "randomised controlled trial", "clinical trial")),
    ("observational", ("cohort study", "observational study", "case-control study", "cross-sectional study")),
    ("economic_evaluation", ("cost-effectiveness", "cost effectiveness", "economic evaluation")),
    ("commentary", ("editorial", "comment", "letter")),
)


def _element_text(element):
    return " ".join("".join(element.itertext()).split()) if element is not None else ""


def _children_text(parent, path):
    element = parent.find(path)
    if element is None:
        return []
    return [
        _element_text(child)
        for child in list(element)
        if _element_text(child)
    ]


def _first_year(element):
    if element is None:
        return None
    year = element.findtext("Year")
    if year and re.fullmatch(r"\d{4}", year.strip()):
        return int(year.strip())
    medline_date = element.findtext("MedlineDate") or ""
    match = re.search(r"\b(?:19|20)\d{2}\b", medline_date)
    return int(match.group()) if match else None


def _semantic_representation(record):
    identifiers = [
        ("PMID", record.get("pmid")),
        ("PMCID", record.get("pmcid")),
        ("DOI", record.get("doi")),
    ]
    return "\n".join(
        part
        for part in (
            f"Title: {record.get('title') or ''}".strip(),
            f"Abstract: {record['abstract']}"
            if record.get("abstract")
            else "",
            f"Keywords: {', '.join(record.get('keywords') or [])}"
            if record.get("keywords")
            else "",
            f"MeSH subject terms: {', '.join(record.get('mesh_terms') or [])}"
            if record.get("mesh_terms")
            else "",
            f"Study type: {', '.join(record.get('publication_types') or [])}"
            if record.get("publication_types")
            else "",
            "Identifiers: " + "; ".join(
                f"{name} {value}" for name, value in identifiers if value
            ),
        )
        if part
    )


def _article_metadata(article):
    citation = article.find("MedlineCitation")
    if citation is None:
        return None
    source_article = citation.find("Article")
    if source_article is None:
        return None
    pmid = _element_text(citation.find("PMID"))
    if not pmid:
        return None
    title = _element_text(source_article.find("ArticleTitle")) or f"PubMed article {pmid}"
    abstract_parts = []
    abstract_sections = {}
    abstract_node = source_article.find("Abstract")
    if abstract_node is not None:
        for item in abstract_node.findall("AbstractText"):
            label = item.attrib.get("Label") or item.attrib.get("NlmCategory")
            text = _element_text(item)
            if text:
                abstract_parts.append(f"{label}: {text}" if label else text)
                if label:
                    abstract_sections[label.casefold()] = text
    abstract = "\n".join(abstract_parts) or None
    authors = []
    for author in source_article.findall("AuthorList/Author"):
        collective = _element_text(author.find("CollectiveName"))
        name = " ".join(
            part for part in (
                _element_text(author.find("ForeName")),
                _element_text(author.find("LastName")),
            ) if part
        ) or collective
        if name:
            authors.append(name)
    journal = _element_text(source_article.find("Journal/Title")) or None
    publication_year = _first_year(
        source_article.find("Journal/JournalIssue/PubDate")
    )
    if publication_year is None:
        publication_year = _first_year(source_article.find("ArticleDate"))
    publication_types = _children_text(source_article, "PublicationTypeList")
    study_metadata = {
        "publication_types": publication_types,
        "study_type": classify_study_type({
            "publication_types": publication_types,
            "mesh_terms": [],
            "title": title,
            "abstract": abstract,
        }),
        "abstract_sections": abstract_sections,
    }
    for field_name, labels in {
        "population": ("population", "participants", "patients", "subjects"),
        "intervention": ("intervention", "treatment", "exposure"),
        "outcomes": ("outcome", "outcomes", "endpoints", "primary outcome"),
    }.items():
        values = [
            text for label, text in abstract_sections.items()
            if any(name in label for name in labels)
        ]
        if values:
            study_metadata[field_name] = values
    mesh_terms = [
        _element_text(item.find("DescriptorName"))
        for item in citation.findall("MeshHeadingList/MeshHeading")
        if _element_text(item.find("DescriptorName"))
    ]
    keywords = [
        _element_text(item)
        for item in citation.findall("KeywordList/Keyword")
        if _element_text(item)
    ]
    identifiers = {}
    article_ids = article.findall("PubmedData/ArticleIdList/ArticleId")
    for identifier in article_ids:
        kind = identifier.attrib.get("IdType", "").casefold()
        value = _element_text(identifier)
        if value and kind in {"doi", "pmc"}:
            identifiers[kind] = value
    for identifier in source_article.findall("ELocationID"):
        if identifier.attrib.get("EIdType", "").casefold() == "doi":
            identifiers.setdefault("doi", _element_text(identifier))
    pmcid = identifiers.get("pmc")
    if pmcid and not pmcid.upper().startswith("PMC"):
        pmcid = f"PMC{pmcid}"
    full_text_available = bool(pmcid)
    metadata = {
        "publication_date": _element_text(source_article.find("Journal/JournalIssue/PubDate")) or None,
        "language": _children_text(source_article, "Language"),
        "publication_types": publication_types,
        "mesh_terms": mesh_terms,
        "keywords": keywords,
        "article_ids": identifiers,
        "raw_pubmed_xml": ET.tostring(article, encoding="unicode"),
    }
    record = {
        "pmid": pmid,
        "pmcid": pmcid,
        "doi": identifiers.get("doi"),
        "title": title,
        "abstract": abstract,
        "publication_year": publication_year,
        "journal": journal,
        "authors": authors,
        "publication_types": publication_types,
        "keywords": keywords,
        "mesh_terms": mesh_terms,
        "metadata": metadata,
        "study_metadata": study_metadata,
        "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{quote(pmid)}/",
        "full_text_url": (
            f"https://pmc.ncbi.nlm.nih.gov/articles/{quote(pmcid)}/"
            if pmcid else None
        ),
        "full_text_available": full_text_available,
        "full_text_access_status": (
            "available_in_pmc"
            if full_text_available
            else "not_indicated_by_pubmed_metadata"
        ),
        "full_text_reuse_status": (
            "license_not_assessed" if full_text_available else "not_assessed"
        ),
        "evidence_access_level": "abstract" if abstract else "metadata",
    }
    record["semantic_text"] = _semantic_representation(record)
    return record


def classify_study_type(article):
    text = " ".join(
        article.get("publication_types", [])
        + article.get("mesh_terms", [])
        + [article.get("title", "")]
        + [article.get("abstract", "") or ""]
    ).casefold()
    types = [
        category
        for category, patterns in _STUDY_TYPE_PATTERNS
        if any(pattern in text for pattern in patterns)
    ]
    return types or ["unknown"]


def _requested_terms(query, vocabulary):
    lowered = query.casefold()
    def phrase_pattern(term):
        phrase = re.escape(term).replace(r"\ ", r"[\s-]+")
        if not term.endswith("s"):
            phrase += "s?"
        return r"(?<!\w)" + phrase + r"(?!\w)"

    return [
        term for term in vocabulary
        if re.search(phrase_pattern(term), lowered)
    ]


def build_pubmed_search_query(question):
    """Translate a question into broad, bounded PubMed title/abstract terms."""
    intervention_terms = _requested_terms(question, _INTERVENTION_TERMS)
    outcome_terms = _requested_terms(question, _OUTCOME_TERMS)
    phrase_groups = []
    for terms in (intervention_terms, outcome_terms):
        phrases = []
        for term in terms:
            phrases.extend(_SEARCH_EXPANSIONS.get(term, (term,)))
        phrases = list(dict.fromkeys(phrases))
        if phrases:
            phrase_groups.append(
                "(" + " OR ".join(
                    f'"{phrase}"[Title/Abstract]'
                    for phrase in phrases
                ) + ")"
            )
    study_group = None
    if re.search(r"\b(systematic review|meta-analysis)\b", question, re.I):
        study_group = (
            '("systematic review"[Title/Abstract] OR '
            '"meta-analysis"[Title/Abstract])'
        )
    negative_group = None
    explicit_null_intent = re.search(
        r"\b(null|conflicting|mixed results?|no effect|no benefit)\b",
        question,
        re.I,
    )
    asks_for_contrary_evidence = re.search(
        r"\b(contradict\w*|oppos\w*|negative findings?|what evidence "
        r"against|against the idea)\b",
        question,
        re.I,
    )
    if explicit_null_intent or asks_for_contrary_evidence:
        negative_group = (
            '("null result"[Title/Abstract] OR "no significant difference"'
            '[Title/Abstract] OR "conflicting evidence"[Title/Abstract] OR '
            '"mixed results"[Title/Abstract] OR "no effect"[Title/Abstract] '
            'OR "no benefit"[Title/Abstract] OR "did not reduce"[Title/Abstract])'
        )
    if phrase_groups:
        core_groups = phrase_groups + ([study_group] if study_group else [])
        core_query = " AND ".join(core_groups)
        if (
            asks_for_contrary_evidence
            and negative_group
            and intervention_terms
        ):
            alternative_groups = [
                phrase_groups[0],
                negative_group,
            ]
            if study_group:
                alternative_groups.append(study_group)
            alternative_query = " AND ".join(alternative_groups)
            return f"({core_query}) OR ({alternative_query})"
        if explicit_null_intent and negative_group:
            core_groups.append(negative_group)
        return " AND ".join(core_groups)

    terms = [
        token for token in re.findall(r"[a-z0-9-]+", question.casefold())
        if token not in _QUERY_NOISE and len(token) > 1
    ]
    return " ".join(dict.fromkeys(terms)) or question


def _term_fit(query, article_text, vocabulary):
    required = _requested_terms(query, vocabulary)
    if not required:
        return {"score": None, "matched_terms": [], "requested_terms": []}
    lowered = article_text.casefold()
    matched = [term for term in required if term in lowered]
    return {
        "score": len(matched) / len(required),
        "matched_terms": matched,
        "requested_terms": required,
    }


def _query_study_types(query):
    lowered = query.casefold()
    return [
        category
        for category, patterns in _STUDY_TYPE_PATTERNS
        if any(pattern in lowered for pattern in patterns)
    ]


def _factual_relevance(query, article, current_year=None):
    current_year = current_year or date.today().year
    combined = " ".join(
        [
            article.get("title", ""),
            article.get("abstract", "") or "",
            " ".join(article.get("keywords", [])),
            " ".join(article.get("mesh_terms", [])),
            " ".join(article.get("publication_types", [])),
        ]
    )
    study_types = classify_study_type(article)
    requested_study_types = _query_study_types(query)
    study_type_relevance = None
    if requested_study_types:
        study_type_relevance = (
            1.0 if set(study_types) & set(requested_study_types) else 0.0
        )
    year = article.get("publication_year")
    year_relevance = None
    requested_years = [
        int(value)
        for value in re.findall(r"\b(?:19|20)\d{2}\b", query)
    ]
    if year and requested_years:
        if len(requested_years) > 1:
            year_relevance = float(min(requested_years) <= year <= max(requested_years))
        else:
            year_relevance = float(year >= requested_years[0])
    elif year and re.search(r"\b(recent|latest|current)\b", query, re.I):
        year_relevance = max(0.0, 1.0 - max(0, current_year - year) / 20)
    elif year:
        year_relevance = max(0.0, 1.0 - max(0, current_year - year) / 40)
    return {
        "study_type": study_types,
        "study_type_relevance": study_type_relevance,
        "publication_year_relevance": year_relevance,
        "population_fit": _term_fit(query, combined, _POPULATION_TERMS),
        "intervention_fit": _term_fit(query, combined, _INTERVENTION_TERMS),
        "outcome_fit": _term_fit(query, combined, _OUTCOME_TERMS),
    }


def _score_article(query, article, query_vector):
    vector = article.get("embedding")
    if vector is None:
        raise ValueError(f"Literature record {article['pmid']} has no embedding")
    semantic_score = embedding._cosine_similarity(query_vector, vector)
    factual = _factual_relevance(query, article)
    components = {
        "semantic_relevance": semantic_score,
        "study_type_relevance": factual["study_type_relevance"],
        "publication_year_relevance": factual["publication_year_relevance"],
        "population_relevance": factual["population_fit"]["score"],
        "intervention_relevance": factual["intervention_fit"]["score"],
        "outcome_relevance": factual["outcome_fit"]["score"],
    }
    weights = {
        "semantic_relevance": 0.60,
        "study_type_relevance": 0.10,
        "publication_year_relevance": 0.10,
        "population_relevance": 0.07,
        "intervention_relevance": 0.07,
        "outcome_relevance": 0.06,
    }
    available = [
        name for name, value in components.items()
        if value is not None
    ]
    denominator = sum(weights[name] for name in available)
    rank_score = (
        sum(components[name] * weights[name] for name in available) / denominator
        if denominator
        else semantic_score
    )
    return {
        **article,
        "scores": components,
        "factual_relevance": factual,
        "rank_score": rank_score,
        "ranking_formula": (
            "weighted mean of non-null components; " + json.dumps(weights, sort_keys=True)
        ),
        "why_selected": _selection_reason(components, factual),
        "evidence_access_level": article.get("evidence_access_level", "metadata"),
        "causal_inference_status": (
            "association_only"
            if "observational" in factual["study_type"]
            else "not_assessed"
        ),
    }


def _selection_reason(components, factual):
    reasons = [
        f"semantic relevance {components['semantic_relevance']:.3f}"
    ]
    for key, label in (
        ("study_type_relevance", "requested study type"),
        ("publication_year_relevance", "publication year"),
        ("population_relevance", "population fit"),
        ("intervention_relevance", "intervention fit"),
        ("outcome_relevance", "outcome fit"),
    ):
        value = components[key]
        if value is not None:
            reasons.append(f"{label} {value:.3f}")
    if factual["study_type"] == ["unknown"]:
        reasons.append("study design not classified from PubMed metadata")
    return "; ".join(reasons)


class PubMedSource(LiteratureSource):
    source = "PUBMED"

    def __init__(
        self,
        email=None,
        api_key=None,
        session=None,
        timeout=REQUEST_TIMEOUT,
    ):
        self.email = email
        self.api_key = api_key
        self.session = session or requests.Session()
        self.timeout = timeout
        self.last_trace = {}
        self._last_request_at = None

    def _params(self, params):
        result = dict(params)
        if self.email:
            result["email"] = self.email
        if self.api_key:
            result["api_key"] = self.api_key
        return result

    def _request(self, endpoint, params):
        requests_per_second = 10 if self.api_key else 3
        interval = 1 / requests_per_second
        if self._last_request_at is not None:
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < interval:
                time.sleep(interval - elapsed)
        self._last_request_at = time.monotonic()
        response = self.session.get(
            f"{EUTILS}/{endpoint}",
            params=self._params(params),
            timeout=self.timeout,
            headers={"User-Agent": "LocalVBCResearch/1.0 (PubMed E-utilities)"},
        )
        response.raise_for_status()
        return response

    @staticmethod
    def _query_key(query):
        normalized = "pubmed-query-v2:" + " ".join(query.casefold().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _search_ids(self, query, limit):
        search_query = build_pubmed_search_query(query)
        key = self._query_key(search_query)
        cached = database.get_literature_search_cache(key, limit)
        if cached is not None:
            return cached, True, search_query
        response = self._request(
            "esearch.fcgi",
            {
                "db": "pubmed",
                "term": search_query,
                "retmax": limit,
                "retstart": 0,
                "sort": "relevance",
                "retmode": "json",
                "tool": "local_vbc_literature",
            },
        )
        payload = response.json()
        search_result = payload.get("esearchresult", {})
        if search_result.get("error"):
            raise ValueError(
                "PubMed ESearch rejected query: "
                + str(search_result["error"])
            )
        pmids = list(dict.fromkeys(search_result.get("idlist", [])))[:limit]
        database.save_literature_search_cache(
            key,
            search_query,
            pmids,
            limit,
        )
        return pmids, False, search_query

    def _fetch_articles(self, pmids):
        if not pmids:
            return []
        response = self._request(
            "efetch.fcgi",
            {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml",
                "tool": "local_vbc_literature",
            },
        )
        root = ET.fromstring(response.content)
        return [
            parsed
            for article in root.findall(".//PubmedArticle")
            if (parsed := _article_metadata(article)) is not None
        ]

    def _records_for_pmids(self, pmids, embedding_model):
        sqlite_started = time.perf_counter()
        cached = database.get_literature_records(pmids)
        cache_load_seconds = time.perf_counter() - sqlite_started
        missing = [pmid for pmid in pmids if pmid not in cached]
        fetch_started = time.perf_counter()
        fetched = self._fetch_articles(missing)
        fetch_seconds = time.perf_counter() - fetch_started
        records_by_id = {record["pmid"]: record for record in fetched}
        records_by_id.update(cached)
        records = [
            records_by_id[pmid]
            for pmid in pmids if pmid in records_by_id
        ]
        pending = []
        for record in records:
            if not record.get("full_text_url") and record.get("pmcid"):
                record["full_text_url"] = (
                    "https://pmc.ncbi.nlm.nih.gov/articles/"
                    f"{record['pmcid']}/"
                )
            if not record.get("study_metadata"):
                record["study_metadata"] = {
                    "study_type": classify_study_type(record),
                }
            semantic_text = _semantic_representation(record)
            semantic_text_changed = record.get("semantic_text") != semantic_text
            record["semantic_text"] = semantic_text
            vector = record.get("embedding")
            if (
                record.get("embedding_model") == embedding_model
                and vector is not None
                and not semantic_text_changed
            ):
                continue
            pending.append(record)
        embed_started = time.perf_counter()
        vectors = (
            embedding.embed_texts(
                (record["semantic_text"] for record in pending),
                embedding_model,
            )
            if pending
            else []
        )
        if len(vectors) != len(pending):
            raise ValueError(
                "Embedding response count does not match pending articles"
            )
        for record, vector in zip(pending, vectors):
            record["embedding"] = vector
            record["embedding_model"] = embedding_model
        embed_seconds = time.perf_counter() - embed_started
        persist_counts = database.upsert_literature_records(records)
        return (
            records,
            {
                "sqlite_cache_load_seconds": cache_load_seconds,
                "metadata_fetch_seconds": fetch_seconds,
                "embedding_seconds": embed_seconds,
                "records_fetched": len(fetched),
                "metadata_cache_hits": len(cached),
                "embedding_cache_hits": persist_counts["embedding_reused"],
                "records_embedded": len(pending),
                "persist_counts": persist_counts,
            },
        )

    def search(self, query, limit=20, embedding_model="nomic-local"):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not 1 <= limit <= MAX_SEARCH_RESULTS:
            raise ValueError(
                f"limit must be between 1 and {MAX_SEARCH_RESULTS}"
            )
        started = time.perf_counter()
        search_started = time.perf_counter()
        pmids, search_cache_hit, pubmed_query = self._search_ids(
            query.strip(),
            limit,
        )
        search_seconds = time.perf_counter() - search_started
        records, record_metrics = self._records_for_pmids(
            pmids,
            embedding_model,
        )
        query_started = time.perf_counter()
        query_vector = embedding.embed_text(
            "search_query: " + query.strip(),
            embedding_model,
        ) if records else None
        query_embedding_seconds = time.perf_counter() - query_started
        rank_started = time.perf_counter()
        ranked = [
            _score_article(query, record, query_vector)
            for record in records
        ] if records else []
        ranked.sort(key=lambda result: result["rank_score"], reverse=True)
        ranking_seconds = time.perf_counter() - rank_started
        results = ranked[:limit]
        self.last_trace = {
            "source": self.source,
            "candidate_stage": {
                "api": "NCBI E-utilities ESearch",
                "pubmed_query": pubmed_query,
                "candidate_pmids": pmids,
                "candidate_count": len(pmids),
                "cache_hit": search_cache_hit,
                "elapsed_seconds": search_seconds,
            },
            "metadata_stage": {
                "api": "NCBI E-utilities EFetch",
                "records_fetched": record_metrics["records_fetched"],
                "metadata_cache_hits": record_metrics["metadata_cache_hits"],
                "elapsed_seconds": record_metrics["metadata_fetch_seconds"],
            },
            "ranking_stage": {
                "query_embedding_seconds": query_embedding_seconds,
                "sqlite_cache_load_seconds": record_metrics["sqlite_cache_load_seconds"],
                "article_vectors_compared": len(records),
                "cosine_and_factual_ranking_seconds": ranking_seconds,
                "returned": len(results),
            },
            "embedding": {
                "elapsed_seconds": record_metrics["embedding_seconds"],
                "records_embedded": record_metrics["records_embedded"],
                "cache_hits": record_metrics["embedding_cache_hits"],
                "persist": record_metrics["persist_counts"],
            },
            "total_elapsed_seconds": time.perf_counter() - started,
            "selected_article": results[0]["pmid"] if results else None,
        }
        for result in results:
            result["retrieval_trace"] = self.last_trace
        return results

    def get_article(self, identifier, embedding_model="nomic-local"):
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("identifier must be a non-empty PMID, PMCID, or DOI")
        identifier = identifier.strip()
        if re.fullmatch(r"\d+", identifier):
            pmid = identifier
        else:
            if identifier.upper().startswith("PMC"):
                query = f"{identifier.upper()}[pmc]"
            else:
                doi = identifier.removeprefix("https://doi.org/").removeprefix("http://doi.org/")
                query = f'"{doi}"[aid]'
            response = self._request(
                "esearch.fcgi",
                {
                    "db": "pubmed",
                    "term": query,
                    "retmax": 2,
                    "retmode": "json",
                    "tool": "local_vbc_literature",
                },
            )
            payload = response.json().get("esearchresult", {})
            pmids = payload.get("idlist", [])
            if not pmids:
                raise LookupError(f"No PubMed article found for {identifier}")
            pmid = pmids[0]
        records, _ = self._records_for_pmids([pmid], embedding_model)
        if not records:
            raise LookupError(f"No PubMed article found for {identifier}")
        return records[0]
