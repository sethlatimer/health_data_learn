import re
import time
from datetime import date

import embedding
from data_sources.cms import CMSDataSource
from data_sources.compatibility import (
    classify_measure,
    check_concept_compatibility,
)
from data_sources.oecd import OECDDataSource
from data_sources.who import WHODataSource
from database import get_indexed_dataset_profiles


_COUNTRIES = {
    "australia": ("Australia", ("Australia", "AUS")),
    "austria": ("Austria", ("Austria", "AUT")),
    "belgium": ("Belgium", ("Belgium", "BEL")),
    "canada": ("Canada", ("Canada", "CAN")),
    "chile": ("Chile", ("Chile", "CHL")),
    "colombia": ("Colombia", ("Colombia", "COL")),
    "costa rica": ("Costa Rica", ("Costa Rica", "CRI")),
    "czech republic": ("Czech Republic", ("Czech Republic", "CZE")),
    "denmark": ("Denmark", ("Denmark", "DNK")),
    "estonia": ("Estonia", ("Estonia", "EST")),
    "finland": ("Finland", ("Finland", "FIN")),
    "france": ("France", ("France", "FRA")),
    "germany": ("Germany", ("Germany", "DEU")),
    "greece": ("Greece", ("Greece", "GRC")),
    "hungary": ("Hungary", ("Hungary", "HUN")),
    "iceland": ("Iceland", ("Iceland", "ISL")),
    "ireland": ("Ireland", ("Ireland", "IRL")),
    "israel": ("Israel", ("Israel", "ISR")),
    "italy": ("Italy", ("Italy", "ITA")),
    "japan": ("Japan", ("Japan", "JPN")),
    "korea": ("Korea", ("Korea", "KOR")),
    "latvia": ("Latvia", ("Latvia", "LVA")),
    "lithuania": ("Lithuania", ("Lithuania", "LTU")),
    "luxembourg": ("Luxembourg", ("Luxembourg", "LUX")),
    "mexico": ("Mexico", ("Mexico", "MEX")),
    "netherlands": ("Netherlands", ("Netherlands", "NLD")),
    "new zealand": ("New Zealand", ("New Zealand", "NZL")),
    "norway": ("Norway", ("Norway", "NOR")),
    "poland": ("Poland", ("Poland", "POL")),
    "portugal": ("Portugal", ("Portugal", "PRT")),
    "slovak republic": ("Slovak Republic", ("Slovak Republic", "SVK")),
    "slovakia": ("Slovakia", ("Slovakia", "SVK")),
    "slovenia": ("Slovenia", ("Slovenia", "SVN")),
    "spain": ("Spain", ("Spain", "ESP")),
    "sweden": ("Sweden", ("Sweden", "SWE")),
    "switzerland": ("Switzerland", ("Switzerland", "CHE")),
    "turkey": ("Turkey", ("Turkey", "TUR")),
    "türkiye": ("Türkiye", ("Türkiye", "TUR")),
    "united kingdom": ("United Kingdom", ("United Kingdom", "GBR")),
    "uk": ("United Kingdom", ("United Kingdom", "GBR")),
    "united states": ("United States", ("United States", "USA", "US")),
    "usa": ("United States", ("United States", "USA", "US")),
    "us": ("United States", ("United States", "USA", "US")),
}
_US_STATES = (
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho",
    "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana",
    "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "New York",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
    "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
    "West Virginia", "Wisconsin", "Wyoming",
)
_VALID_INTENT_VALUES = {
    "topic": {"utilization", "spending", None},
    "measure_type": {"count", "rate", "amount", None},
    "desired_analysis": {
        "trend",
        "comparison",
        "ranking",
        "snapshot",
        None,
    },
}


def _match_phrase(text, phrase):
    return re.search(
        r"(?<![a-z0-9])"
        + re.escape(phrase)
        + r"(?![a-z0-9])",
        text,
        flags=re.IGNORECASE,
    ) is not None


def parse_question_requirements(question):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    text = question.strip()
    lowered = text.casefold()
    requirements = {
        "question": text,
        "topic": None,
        "measure_type": None,
        "geography": None,
        "population": None,
        "start_period": None,
        "end_period": None,
        "unit": None,
        "measure_family": None,
        "quantity_type": None,
        "measure_domain": None,
        "specific_condition": None,
        "financial_measure": None,
        "provider_type": None,
        "desired_analysis": None,
        "adjustment": None,
        "care_setting": None,
        "currentness": None,
        "filters": [],
        "ambiguities": [],
        "parser": "deterministic",
    }

    geography_matches = []
    for alias in sorted(_COUNTRIES, key=len, reverse=True):
        if _match_phrase(lowered, alias):
            canonical, aliases = _COUNTRIES[alias]
            geography_matches.append({
                "name": canonical,
                "level": "country",
                "aliases": list(aliases),
                "start": lowered.find(alias),
            })
    for state in _US_STATES:
        if _match_phrase(lowered, state.casefold()):
            geography_matches.append({
                "name": state,
                "level": "state",
                "aliases": [state],
                "start": lowered.find(state.casefold()),
            })
    if geography_matches:
        geography_matches.sort(
            key=lambda item: (
                -len(item["name"]),
                item["start"],
            )
        )
        unique = {
            (item["name"], item["level"]): item
            for item in geography_matches
        }
        if len(unique) == 1:
            requirements["geography"] = next(iter(unique.values()))
        else:
            requirements["ambiguities"].append("geography")

    population_phrases = (
        "original medicare part a",
        "original medicare",
        "medicare advantage",
        "medicaid beneficiaries",
        "dual eligible beneficiaries",
        "medicare beneficiaries",
        "older adults",
        "adults",
        "children",
        "women",
        "men",
    )
    population_matches = [
        phrase
        for phrase in population_phrases
        if _match_phrase(lowered, phrase)
    ]
    if population_matches:
        requirements["population"] = max(population_matches, key=len)
    if re.search(r"\b(?:women|females?)\b", lowered):
        requirements["filters"].append({
            "name": "sex",
            "value": "female",
        })
    elif re.search(r"\b(?:men|males?)\b", lowered):
        requirements["filters"].append({
            "name": "sex",
            "value": "male",
        })
    age_match = re.search(
        r"\b(?:age[d]?\s*)?(under\s+\d+|\d{1,3}\s*(?:\+|and over|or older))\b",
        lowered,
    )
    if age_match:
        requirements["filters"].append({
            "name": "age",
            "value": re.sub(r"\s+", " ", age_match.group(1)),
        })

    period = re.search(
        r"\b(?:from|between)\s+((?:19|20)\d{2})\s+"
        r"(?:to|through|and)\s+((?:19|20)\d{2})\b",
        lowered,
    )
    since = re.search(
        r"\b(?:since|from|starting(?:\s+in)?)\s+((?:19|20)\d{2})\b",
        lowered,
    )
    years = re.findall(r"\b(?:19|20)\d{2}\b", lowered)
    if period:
        requirements["start_period"] = period.group(1)
        requirements["end_period"] = period.group(2)
    elif since:
        requirements["start_period"] = since.group(1)
    elif re.search(r"\b(last|previous|past)\s+year\b", lowered):
        prior_year = str(date.today().year - 1)
        requirements["start_period"] = prior_year
        requirements["end_period"] = prior_year
    elif years:
        requirements["start_period"] = years[0]
        requirements["end_period"] = years[-1]

    amount_intent = bool(re.search(
        r"\b(spend(?:ing)?|payment|payments|costs?|charges?|"
        r"expenditure|dollars?|amounts?)\b",
        lowered,
    ))
    utilization_intent = bool(re.search(
        r"\b(utili[sz]ation|admissions?|discharges?|in[\s-]?patient|"
        r"out[\s-]?patient|hospital use|visits?|stays?)\b",
        lowered,
    ))
    if amount_intent and utilization_intent:
        requirements["ambiguities"].append("topic")
    elif amount_intent:
        requirements["topic"] = "spending"
    elif utilization_intent:
        requirements["topic"] = "utilization"
    if requirements["topic"] == "spending":
        requirements["measure_family"] = "spending_payment"
        if re.search(
            r"\b(payment|payments|paid|pay|received|receive|"
            r"medicare spending)\b",
            lowered,
        ):
            requirements["financial_measure"] = "payment"
    elif requirements["topic"] == "utilization":
        requirements["measure_family"] = "utilization"

    if re.search(r"\b(mortalit(?:y|ies)|death rate|deaths?)\b", lowered):
        requirements["measure_family"] = "mortality"
    elif re.search(r"\b(prevalence|prevalent)\b", lowered):
        requirements["measure_family"] = "prevalence"
    elif re.search(r"\b(incidence|incident cases|new cases)\b", lowered):
        requirements["measure_family"] = "incidence"
    elif re.search(
        r"\b(quality|readmission|patient safety|patient experience)\b",
        lowered,
    ):
        requirements["measure_family"] = "quality"
    elif (
        not amount_intent
        and re.search(
            r"\b(physicians?|physician supply|physician density|provider count|"
            r"number of providers|hospital beds|bed capacity|capacity)\b",
            lowered,
        )
    ):
        requirements["measure_family"] = "provider_capacity"
    if re.search(r"\b(physicians?|medical doctors?)\b", lowered):
        requirements["provider_type"] = "physician"
    elif re.search(r"\b(hospital beds?|bed capacity)\b", lowered):
        requirements["provider_type"] = "hospital_beds"
    elif re.search(r"\b(providers?|healthcare workforce)\b", lowered):
        requirements["provider_type"] = "provider"

    if re.search(
        r"\b(behavioral health|behavioural health|mental health|"
        r"psychiatric)\b",
        lowered,
    ):
        requirements["measure_domain"] = "behavioral_health"
    condition_phrases = (
        ("heart_failure", r"\b(congestive heart failure|heart failure)\b"),
        ("schizophrenia", r"\bschizophrenia\b"),
        ("depression", r"\b(depression|depressive)\b"),
        ("anxiety", r"\banxiety\b"),
        ("bipolar_disorder", r"\bbipolar\b"),
        ("adhd", r"\b(adhd|attention deficit)\b"),
        ("asthma", r"\basthma\b"),
        ("diabetes", r"\b(diabetes|diabetic)\b"),
        ("copd", r"\b(copd|chronic obstructive pulmonary)\b"),
    )
    requirements["specific_condition"] = next(
        (
            condition
            for condition, pattern in condition_phrases
            if re.search(pattern, lowered)
        ),
        None,
    )

    if re.search(r"\bin[\s-]?patient\b", lowered):
        requirements["care_setting"] = "inpatient"
    elif re.search(r"\bout[\s-]?patient\b", lowered):
        requirements["care_setting"] = "outpatient"
    elif re.search(r"\bhospital(?:ization|isation)?s?\b", lowered):
        requirements["care_setting"] = "hospital"

    rate_intent = bool(re.search(
        r"\b(rate|rates|per capita|per person|per inhabitant|"
        r"per head|per\s+\d[\d,]*|percent(?:age)?|proportion|ratio)\b",
        lowered,
    ))
    count_intent = bool(re.search(
        r"\b(how many|number of|count|counts|total number)\b",
        lowered,
    ))
    if rate_intent:
        requirements["measure_type"] = "rate"
    elif count_intent:
        requirements["measure_type"] = "count"
    elif amount_intent:
        requirements["measure_type"] = "amount"

    unit_match = re.search(
        r"\b(per capita|per person|per inhabitant|per head|"
        r"per\s+[\d,]+\s*(?:people|persons?|beneficiaries|patients)?)\b",
        lowered,
    )
    if unit_match:
        requirements["unit"] = re.sub(r"\s+", " ", unit_match.group(1))
    elif re.search(r"\b(?:dollars?|usd)\b", lowered):
        requirements["unit"] = "dollars"
    elif re.search(r"\bpercent(?:age)?\b", lowered):
        requirements["unit"] = "percent"

    if re.search(r"\bstandardi[sz]ed\b|\brisk[- ]adjusted\b|\bage[- ]adjusted\b", lowered):
        requirements["adjustment"] = "standardized"
    elif re.search(r"\bcrude\b|\bunadjusted\b", lowered):
        requirements["adjustment"] = "crude"

    if requirements["adjustment"] == "standardized" and rate_intent:
        requirements["quantity_type"] = "standardized_rate"
    elif re.search(
        r"\b(per capita|per person|per inhabitant|per head|"
        r"per\s+[\d,]+\s*(?:people|persons?|beneficiaries|patients)?)\b",
        lowered,
    ):
        requirements["quantity_type"] = "per_capita"
    elif re.search(r"\b(percent(?:age)?|proportion)\b", lowered):
        requirements["quantity_type"] = "percent"
    elif rate_intent:
        requirements["quantity_type"] = "rate"
    elif amount_intent:
        requirements["quantity_type"] = "currency"
    elif count_intent:
        requirements["quantity_type"] = "count"

    if re.search(r"\b(latest|current|most recent|up[- ]to[- ]date)\b", lowered):
        requirements["currentness"] = "current"
    elif re.search(r"\b(archived|historical|older release)\b", lowered):
        requirements["currentness"] = "archived"

    if re.search(r"\b(compare|comparison|versus|vs\.?)\b", lowered):
        requirements["desired_analysis"] = "comparison"
    elif re.search(r"\b(rank|ranking|highest|lowest|top)\b", lowered):
        requirements["desired_analysis"] = "ranking"
    elif re.search(r"\b(changed?|trend|increased?|decreased?|over time|since)\b", lowered):
        requirements["desired_analysis"] = "trend"
    elif re.search(r"\b(how many|how much|what is|what was|what were)\b", lowered):
        requirements["desired_analysis"] = "snapshot"

    if (
        amount_intent
        and count_intent
        and requirements["measure_type"] == "count"
        and requirements["topic"] == "spending"
    ):
        requirements["ambiguities"].append("measure_type")
    requirements["ambiguities"] = list(dict.fromkeys(
        requirements["ambiguities"]
    ))
    return requirements


def _resolve_ambiguous_intent(question, requirements, intent_resolver):
    if not requirements["ambiguities"] or intent_resolver is None:
        return requirements
    resolvable = [
        field
        for field in requirements["ambiguities"]
        if field in _VALID_INTENT_VALUES
    ]
    if not resolvable:
        return requirements
    resolution = intent_resolver(
        question,
        tuple(resolvable),
    )
    if not isinstance(resolution, dict):
        raise ValueError("Intent resolver must return a JSON-style object")
    for field in resolvable:
        if field not in resolution:
            continue
        value = resolution[field]
        if (
            not isinstance(value, (str, type(None)))
            or value not in _VALID_INTENT_VALUES[field]
        ):
            raise ValueError(
                f"Intent resolver returned invalid {field}: {value!r}"
            )
        requirements[field] = value
    unresolved = [
        field
        for field in requirements["ambiguities"]
        if field not in resolvable or field not in resolution
    ]
    requirements["ambiguities"] = unresolved
    if resolution:
        requirements["parser"] = "deterministic+llm-disambiguation"
    return requirements


def _lexical_score(question, profile, requirements=None):
    stop_words = {
        "a", "an", "and", "are", "by", "during", "for", "from",
        "how", "in", "is", "last", "many", "me", "of", "on", "the",
        "there", "this", "to", "was", "were", "what", "when", "which",
        "with",
    }
    words = {
        word
        for word in re.findall(r"[a-z0-9]+", question.casefold())
        if len(word) > 2
        and word not in stop_words
        and not re.fullmatch(r"(?:19|20)\d{2}", word)
    }
    geography = (requirements or {}).get("geography")
    if geography:
        geography_words = {
            word
            for alias in geography.get("aliases", [])
            for word in re.findall(r"[a-z0-9]+", alias.casefold())
        }
        words.difference_update(geography_words)
    if not words:
        return 0.0
    title_words = set(re.findall(
        r"[a-z0-9]+",
        str(profile.get("title", "")).casefold(),
    ))
    description_words = set(re.findall(
        r"[a-z0-9]+",
        str(profile.get("description", "")).casefold(),
    ))
    title_overlap = len(words & title_words) / len(words)
    description_overlap = len(words & description_words) / len(words)
    return 0.7 * title_overlap + 0.3 * description_overlap


def _candidate_score(semantic_score, lexical_score):
    if semantic_score is None:
        return lexical_score
    return 0.65 * semantic_score + 0.35 * lexical_score


class DataSourceRetrieval:
    def __init__(self, data_sources=None):
        if data_sources is None:
            data_sources = (
                OECDDataSource(),
                CMSDataSource(),
                WHODataSource(),
            )
        if isinstance(data_sources, dict):
            items = data_sources.values()
        else:
            items = data_sources
        self.data_sources = {
            source.source.upper(): source
            for source in items
        }

    def retrieve(
        self,
        question,
        source=None,
        top_dataset_candidates=20,
        concept_dataset_count=5,
        top_concepts=20,
        intent_resolver=None,
    ):
        total_started = time.perf_counter()
        instrumentation = {
            "stage_timings_seconds": {
                "question_parsing": 0.0,
                "dataset_ranking": 0.0,
                "compatibility_checks": 0.0,
                "concept_search": 0.0,
            },
            "database": {},
            "embedding_calls": [],
            "counts": {},
            "cache_usage": {
                "query_embedding_cache": False,
                "indexed_vectors_loaded_from_sqlite": True,
            },
        }

        def finish(trace):
            instrumentation["stage_timings_seconds"][
                "total_elapsed"
            ] = time.perf_counter() - total_started
            instrumentation["query_embedding_seconds"] = sum(
                call.get("query_embedding_seconds", 0.0)
                for call in instrumentation["embedding_calls"]
            )
            instrumentation["sqlite_vector_loading_seconds"] = sum(
                call.get("vector_loading_seconds", 0.0)
                for call in instrumentation["embedding_calls"]
            )
            instrumentation["cosine_similarity_ranking_seconds"] = sum(
                call.get("cosine_ranking_seconds", 0.0)
                for call in instrumentation["embedding_calls"]
            )
            instrumentation["counts"].setdefault(
                "concept_vectors_loaded",
                sum(
                    call.get("vector_rows_loaded", 0)
                    for call in instrumentation["embedding_calls"]
                ),
            )
            instrumentation["counts"].setdefault(
                "concept_vectors_compared",
                sum(
                    call.get("vectors_compared", 0)
                    for call in instrumentation["embedding_calls"]
                ),
            )
            instrumentation["counts"].setdefault(
                "concepts_returned",
                sum(
                    call.get("concepts_returned", 0)
                    for call in instrumentation["embedding_calls"]
                ),
            )
            instrumentation["cache_usage"][
                "persisted_embedding_vectors_read"
            ] = instrumentation["counts"]["concept_vectors_loaded"]
            trace["instrumentation"] = instrumentation
            return trace

        if top_dataset_candidates < 1:
            raise ValueError("top_dataset_candidates must be positive")
        if concept_dataset_count < 1:
            raise ValueError("concept_dataset_count must be positive")
        if top_concepts < 1:
            raise ValueError("top_concepts must be positive")

        parsing_started = time.perf_counter()
        requirements = _resolve_ambiguous_intent(
            question,
            parse_question_requirements(question),
            intent_resolver,
        )
        instrumentation["stage_timings_seconds"][
            "question_parsing"
        ] = time.perf_counter() - parsing_started
        requested_source = source.upper() if source else None
        if (
            requested_source is not None
            and requested_source not in self.data_sources
        ):
            raise ValueError(f"Unknown data source {source!r}")

        profile_metrics = {}
        profiles = get_indexed_dataset_profiles(
            source=requested_source,
            include_concepts=False,
            instrumentation=profile_metrics,
        )
        instrumentation["database"].update(profile_metrics)
        instrumentation["stage_timings_seconds"][
            "sqlite_profile_loading"
        ] = profile_metrics.get("profile_load_seconds", 0.0)
        if not profiles:
            instrumentation["counts"].update({
                "datasets_considered": 0,
                "dataset_candidates_retained": 0,
            })
            return finish({
                "question": question,
                "requirements": requirements,
                "stage_1_dataset_candidates": [],
                "compatibility_evaluations": [],
                "stage_2_concept_candidates": [],
                "selected_dataset": None,
                "selected_concept": None,
                "selected_source": None,
                "rejection_reasons": [
                    "No locally indexed datasets are available for this source"
                ],
                "compatible_dataset_candidates": [],
            })

        ranking_metrics = {}
        ranking_started = time.perf_counter()
        ranked_embeddings = embedding.rank_dataset_candidates(
            question,
            source=requested_source,
            instrumentation=ranking_metrics,
        )
        scores_by_key = {
            (candidate["source"], candidate["dataset_id"]): candidate
            for candidate in ranked_embeddings
        }
        profiles_by_key = {
            (profile["source"], profile["dataset_id"]): profile
            for profile in profiles
        }
        ranked_profiles = []
        for key, profile in profiles_by_key.items():
            semantic = scores_by_key.get(key)
            semantic_score = semantic["score"] if semantic else None
            lexical = _lexical_score(
                question,
                profile,
                requirements=requirements,
            )
            ranked_profiles.append({
                **profile,
                "semantic_score": semantic_score,
                "best_concept_score": (
                    semantic["best_concept_score"]
                    if semantic
                    else None
                ),
                "candidate_score": _candidate_score(
                    semantic_score,
                    lexical,
                ),
                "lexical_score": lexical,
            })
        ranked_profiles.sort(
            key=lambda profile: profile["candidate_score"],
            reverse=True,
        )
        instrumentation["stage_timings_seconds"][
            "dataset_ranking"
        ] = time.perf_counter() - ranking_started
        instrumentation["embedding_calls"].append({
            "purpose": "dataset_ranking",
            **ranking_metrics,
        })
        instrumentation["counts"].update({
            "datasets_considered": len(profiles),
            "dataset_candidates_retained": min(
                len(ranked_profiles),
                top_dataset_candidates,
            ),
        })
        checked_candidates = []
        compatible_profiles = []
        compatibility_started = time.perf_counter()
        for profile in ranked_profiles[:top_dataset_candidates]:
            adapter = self.data_sources.get(profile["source"].upper())
            if adapter is None:
                compatibility = {
                    "compatible": False,
                    "checks": [],
                    "rejection_reasons": [
                        f"No adapter is registered for {profile['source']}"
                    ],
                }
            else:
                compatibility = adapter.check_compatibility(
                    profile,
                    requirements,
                )
            checked_candidates.append({
                "source": profile["source"],
                "dataset_id": profile["dataset_id"],
                "title": profile["title"],
                "semantic_score": profile["semantic_score"],
                "best_concept_score": profile["best_concept_score"],
                "lexical_score": profile["lexical_score"],
                "candidate_score": profile["candidate_score"],
                "compatibility_checks": compatibility["checks"],
                "dataset_compatible": compatibility["compatible"],
                "compatible": (
                    None if compatibility["compatible"] else False
                ),
                "rejection_reasons": compatibility["rejection_reasons"],
            })
            if compatibility["compatible"]:
                compatible_profiles.append(profile)
        instrumentation["stage_timings_seconds"][
            "compatibility_checks"
        ] += time.perf_counter() - compatibility_started

        best_profiles = compatible_profiles[:concept_dataset_count]
        concept_candidates = []
        if best_profiles:
            concept_metrics = {}
            concept_started = time.perf_counter()
            concept_results = embedding.search_dataset_concepts(
                question,
                top_k=top_concepts * len(best_profiles),
                source=requested_source,
                dataset_ids=[
                    profile["dataset_id"]
                    for profile in best_profiles
                ],
                instrumentation=concept_metrics,
            )
            instrumentation["stage_timings_seconds"][
                "concept_search"
            ] = time.perf_counter() - concept_started
            instrumentation["embedding_calls"].append({
                "purpose": "concept_search",
                **concept_metrics,
            })
            profile_map = {
                (profile["source"], profile["dataset_id"]): profile
                for profile in best_profiles
            }
            for result in concept_results:
                key = (result["source"], result["dataset_id"])
                profile = profile_map.get(key)
                if profile is None:
                    continue
                compatibility_started = time.perf_counter()
                compatibility = check_concept_compatibility(
                    result,
                    requirements,
                    profile=profile,
                )
                instrumentation["stage_timings_seconds"][
                    "compatibility_checks"
                ] += time.perf_counter() - compatibility_started
                concept_candidates.append({
                    **result,
                    "compatibility_checks": compatibility["checks"],
                    "compatible": compatibility["compatible"],
                    "rejection_reasons": compatibility[
                        "rejection_reasons"
                    ],
                    "measure_classification": compatibility[
                        "measure_classification"
                    ],
                })

        for profile in best_profiles:
            candidate = next(
                item
                for item in checked_candidates
                if item["source"] == profile["source"]
                and item["dataset_id"] == profile["dataset_id"]
            )
            matching_concepts = [
                concept
                for concept in concept_candidates
                if concept["source"] == profile["source"]
                and concept["dataset_id"] == profile["dataset_id"]
            ]
            has_compatible_concept = any(
                concept["compatible"]
                for concept in matching_concepts
            )
            candidate["compatible"] = has_compatible_concept
            candidate["compatibility_checks"].append({
                "characteristic": "requested_measure_concept",
                "requested": {
                    "topic": requirements.get("topic"),
                    "measure_type": requirements.get("measure_type"),
                    "care_setting": requirements.get("care_setting"),
                    "adjustment": requirements.get("adjustment"),
                    "unit": requirements.get("unit"),
                    "measure_family": requirements.get("measure_family"),
                    "quantity_type": requirements.get("quantity_type"),
                    "measure_domain": requirements.get("measure_domain"),
                    "specific_condition": requirements.get(
                        "specific_condition"
                    ),
                    "financial_measure": requirements.get(
                        "financial_measure"
                    ),
                    "provider_type": requirements.get("provider_type"),
                },
                "supported": has_compatible_concept,
                "evidence": (
                    "At least one indexed semantic concept"
                    if has_compatible_concept
                    else "No indexed semantic concept satisfies the requested measure"
                ),
            })
            if not has_compatible_concept:
                concept_rejections = list(dict.fromkeys(
                    reason
                    for concept in matching_concepts
                    for reason in concept["rejection_reasons"]
                ))
                candidate["rejection_reasons"].extend(
                    concept_rejections[:5]
                )
                candidate["rejection_reasons"].append(
                    "No indexed semantic concept satisfies the requested measure"
                )
        instrumentation["counts"].update({
            "dataset_candidates_compatible": len(compatible_profiles),
            "dataset_candidates_searched_for_concepts": len(best_profiles),
            "concept_candidates_returned": len(concept_candidates),
        })

        compatibility_evaluations = checked_candidates

        compatible_concepts = [
            concept
            for concept in concept_candidates
            if concept["compatible"]
        ]
        selected_concept = (
            compatible_concepts[0] if compatible_concepts else None
        )
        selected_dataset = None
        selected_source = None
        if selected_concept is not None:
            selected_dataset = next(
                (
                    {
                        "source": candidate["source"],
                        "dataset_id": candidate["dataset_id"],
                        "title": candidate["title"],
                    }
                    for candidate in checked_candidates
                    if candidate["source"] == selected_concept["source"]
                    and candidate["dataset_id"]
                    == selected_concept["dataset_id"]
                ),
                None,
            )
            selected_source = selected_concept["source"]

        rejection_reasons = []
        if selected_dataset is None:
            rejection_reasons = [
                {
                    "source": candidate["source"],
                    "dataset_id": candidate["dataset_id"],
                    "reasons": candidate["rejection_reasons"],
                }
                for candidate in checked_candidates
                if candidate["rejection_reasons"]
            ]
            if not rejection_reasons:
                rejection_reasons.append({
                    "reason": "No candidate semantic concept passed compatibility checks"
                })

        instrumentation["counts"]["selected_source"] = selected_source
        instrumentation["counts"]["selected_dataset_id"] = (
            selected_dataset["dataset_id"]
            if selected_dataset
            else None
        )
        instrumentation["counts"]["selected_concept"] = (
            selected_concept.get("label") if selected_concept else None
        )
        instrumentation["counts"]["selected_concept_code"] = (
            selected_concept.get("code") if selected_concept else None
        )
        instrumentation["counts"]["selected_concept_dimension"] = (
            selected_concept.get("dimension_id")
            if selected_concept
            else None
        )
        return finish({
            "question": question,
            "requirements": requirements,
            "stage_1_dataset_candidates": checked_candidates[
                :top_dataset_candidates
            ],
            "compatibility_evaluations": compatibility_evaluations,
            "compatible_dataset_candidates": [
                candidate
                for candidate in checked_candidates
                if candidate["compatible"] is True
            ],
            "stage_2_concept_candidates": concept_candidates,
            "selected_dataset": selected_dataset,
            "selected_concept": selected_concept,
            "selected_source": selected_source,
            "rejection_reasons": rejection_reasons,
        })


def retrieve_question(question, **kwargs):
    return DataSourceRetrieval().retrieve(question, **kwargs)
