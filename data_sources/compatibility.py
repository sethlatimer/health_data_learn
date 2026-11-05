import re
import unicodedata


_GEOGRAPHY_TERMS = (
    "area",
    "country",
    "geography",
    "location",
    "region",
    "state",
    "territory",
    "zip",
    "postal",
    "county",
)
_TIME_TERMS = (
    "period",
    "time",
    "year",
    "date",
    "quarter",
)


def _normalized(value):
    value = unicodedata.normalize("NFKD", str(value or ""))
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _contains(text, phrase):
    normalized_text = f" {_normalized(text)} "
    normalized_phrase = f" {_normalized(phrase)} "
    return normalized_phrase in normalized_text


def _supports_measure_type(text, measure_type, type_terms):
    terms = type_terms.get(measure_type, (measure_type,))
    if not any(_contains(text, term) for term in terms):
        return False
    if measure_type == "count":
        rate_terms = (
            "rate",
            "per capita",
            "per person",
            "per inhabitant",
            "per 100",
            "per 1 000",
            "per 1000",
            "percent",
            "percentage",
            "proportion",
            "ratio",
        )
        return not any(_contains(text, term) for term in rate_terms)
    return True


def classify_measure(concept, profile=None):
    """Classify a measure from its indexed source metadata, not query wording."""
    profile = profile or {}
    structure = profile.get("structure", {})
    dimensions = structure.get("data_structure", {}).get("dimensions", [])
    dimension = next(
        (
            item
            for item in dimensions
            if item.get("id") == concept.get("dimension_id")
        ),
        {},
    )
    metadata = concept.get("metadata", {})
    evidence_parts = [
        concept.get("code", ""),
        concept.get("dimension_id", ""),
        concept.get("dimension_name", ""),
        concept.get("label", ""),
        dimension.get("description", ""),
        json_safe(metadata),
    ]
    evidence = " ".join(str(part or "") for part in evidence_parts)
    normalized = _normalized(evidence)

    condition_terms = (
        ("adhd", ("adhd", "attention deficit", "conduct disorder")),
        ("asthma", ("asthma",)),
        ("diabetes", ("diabetes", "diabetic")),
        ("depression", ("depression", "depressive")),
        ("anxiety", ("anxiety",)),
        ("schizophrenia", ("schizophrenia",)),
        ("bipolar_disorder", ("bipolar",)),
        ("heart_failure", ("heart failure", "congestive heart failure")),
        ("copd", ("copd", "chronic obstructive pulmonary")),
    )
    condition = next(
        (
            name
            for name, terms in condition_terms
            if any(_contains(normalized, term) for term in terms)
        ),
        None,
    )
    behavioral_terms = (
        "behavioral health",
        "behavioural health",
        "mental health",
        "psychiatric",
        "adhd",
        "attention deficit",
        "conduct disorder",
        "depression",
        "anxiety",
        "schizophrenia",
        "bipolar",
    )
    domain = (
        "behavioral_health"
        if any(_contains(normalized, term) for term in behavioral_terms)
        else None
    )

    if any(_contains(normalized, term) for term in (
        "mortality", "death", "deaths",
    )):
        family = "mortality"
    elif any(_contains(normalized, term) for term in (
        "prevalence", "percent with", "percentage with",
        "percent of beneficiaries with",
        "percentage of beneficiaries with",
        "percent of people with",
        "percentage of people with",
        "share of beneficiaries with",
    )):
        family = "prevalence"
    elif any(_contains(normalized, term) for term in (
        "incidence", "incident cases", "new cases",
    )):
        family = "incidence"
    elif any(_contains(normalized, term) for term in (
        "payment", "payments", "spending", "expenditure", "cost",
        "costs", "charge", "charges", "dollar", "amount",
    )):
        family = "spending_payment"
    elif any(_contains(normalized, term) for term in (
        "physician", "medical doctor", "medical doctors", "health workforce", "provider count",
        "number of providers", "hospital beds", "bed capacity", "capacity",
    )):
        family = "provider_capacity"
    elif any(_contains(normalized, term) for term in (
        "quality", "readmission", "patient experience",
        "patient safety",
    )):
        family = "quality"
    elif any(_contains(normalized, term) for term in (
        "utilization", "utilisation", "admission", "discharge",
        "admissions", "discharges", "visit", "visits",
        "hospitalization", "hospitalisation",
        "encounter", "length of stay", "service use",
    )):
        family = "utilization"
    else:
        family = "other_outcome"

    if any(_contains(normalized, term) for term in (
        "standardized rate", "standardised rate", "age standardized",
        "age standardised",
    )):
        quantity_type = "standardized_rate"
    elif any(_contains(normalized, term) for term in (
        "per capita", "per person", "per inhabitant", "per head",
        "per 1000 persons", "per 1000 people", "per 100000 persons",
        "per 100000 people", "per 1 000 persons", "per 1 000 people",
        "per 100 000 persons", "per 100 000 people",
    )):
        quantity_type = "per_capita"
    elif any(_contains(normalized, term) for term in (
        "percent", "percentage", "proportion",
    )):
        quantity_type = "percent"
    elif any(_contains(normalized, term) for term in (
        "rate", "ratio", "per 1000", "per 100000",
    )):
        quantity_type = "rate"
    elif any(_contains(normalized, term) for term in (
        "payment", "payments", "spending", "expenditure", "cost",
        "costs", "charge", "charges", "dollar", "currency", "amount",
        "usd",
    )):
        quantity_type = "currency"
    elif any(_contains(normalized, term) for term in (
        "count", "number of", "total discharges", "admissions",
        "discharges", "visits", "events", "deaths",
        "number of providers", "provider count", "physician count",
    )):
        quantity_type = "count"
    else:
        quantity_type = "unknown"

    measure_type = {
        "count": "count",
        "rate": "rate",
        "per_capita": "rate",
        "standardized_rate": "rate",
        "percent": "rate",
        "currency": "amount",
    }.get(quantity_type)
    if quantity_type == "unknown":
        measure_type = None

    if any(_contains(normalized, term) for term in (
        "standardized", "standardised", "age adjusted", "risk adjusted",
    )):
        adjustment = "standardized"
    elif any(_contains(normalized, term) for term in (
        "crude", "unadjusted", "not adjusted",
    )):
        adjustment = "crude"
    else:
        adjustment = None

    if any(_contains(normalized, term) for term in (
        "medicare payment", "medicare payments", "payment amount",
        "payments to providers", "provider payment", "allowed amount",
    )):
        financial_measure = "payment"
    elif any(_contains(normalized, term) for term in (
        "charge", "charges",
    )):
        financial_measure = "charge"
    else:
        financial_measure = None

    if any(_contains(normalized, term) for term in (
        "physician", "medical doctor",
    )):
        provider_type = "physician"
    elif any(_contains(normalized, term) for term in (
        "hospital bed", "hospital beds", "bed capacity",
    )):
        provider_type = "hospital_beds"
    elif any(_contains(normalized, term) for term in (
        "provider", "providers",
    )):
        provider_type = "provider"
    else:
        provider_type = None

    if family == "other_outcome" and quantity_type == "unknown":
        confidence = "unknown"
    elif family in ("utilization", "spending_payment", "mortality",
                    "provider_capacity", "quality", "prevalence", "incidence"):
        confidence = "high"
    else:
        confidence = "medium"
    return {
        "family": family,
        "quantity_type": quantity_type,
        "measure_type": measure_type,
        "adjustment": adjustment,
        "domain": domain,
        "condition": condition,
        "financial_measure": financial_measure,
        "provider_type": provider_type,
        "confidence": confidence,
        "evidence": evidence[:1000],
    }


def _all_text(profile, structure):
    parts = [
        profile.get("title", ""),
        profile.get("description", ""),
    ]
    metadata = profile.get("metadata", {})
    if isinstance(metadata, dict):
        parts.extend(
            str(value)
            for key, value in metadata.items()
            if key not in {"distribution", "inSeries"}
        )
    dataflow = structure.get("dataflow", {})
    parts.extend((
        dataflow.get("name", ""),
        dataflow.get("description", ""),
    ))
    data_structure = structure.get("data_structure", {})
    for dimension in data_structure.get("dimensions", []):
        parts.extend((
            dimension.get("id", ""),
            dimension.get("name", ""),
            dimension.get("description", ""),
        ))
        for code in dimension.get("codes", []):
            parts.extend((code.get("id", ""), code.get("label", "")))
    for concept in profile.get("concepts", []):
        parts.extend((
            concept.get("label", ""),
        ))
    return " ".join(str(part or "") for part in parts)


def _dimension_matches(dimensions, terms):
    return [
        dimension
        for dimension in dimensions
        if any(
            term in _normalized(
                f"{dimension.get('id', '')} "
                f"{dimension.get('name', '')} "
                f"{dimension.get('description', '')}"
            ).split()
            for term in terms
        )
    ]


def _periods_within(metadata, start_period, end_period):
    temporal = metadata.get("temporal") or []
    if not temporal:
        return None, "Catalog metadata does not declare a bounded period"
    first = temporal[0]
    start = str(first.get("startDate", ""))[:4]
    end = str(first.get("endDate", ""))[:4]
    if not (start.isdigit() and end.isdigit()):
        return None, "Catalog period metadata is not a year range"
    requested_start = int(start_period) if start_period else int(start)
    requested_end = int(end_period) if end_period else int(end)
    if requested_start < int(start) or requested_end > int(end):
        return False, f"Available catalog period is {start}–{end}"
    return True, f"Requested years fall within catalog period {start}–{end}"


def check_dataset_compatibility(profile, requirements):
    structure = profile.get("structure", {})
    data_structure = structure.get("data_structure", {})
    dimensions = data_structure.get("dimensions", [])
    metadata = profile.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    dataflow_metadata = structure.get("dataflow", {}).get("metadata", {})
    if isinstance(dataflow_metadata, dict):
        metadata = {**dataflow_metadata, **metadata}

    checks = []
    rejections = []
    evidence_text = _all_text(profile, structure)
    evidence_normalized = _normalized(evidence_text)

    def add_check(characteristic, requested, supported, evidence, detail=None):
        check = {
            "characteristic": characteristic,
            "requested": requested,
            "supported": supported,
            "evidence": evidence,
        }
        if detail is not None:
            check["detail"] = detail
        checks.append(check)
        if supported is not True:
            rejections.append(
                f"{characteristic}: "
                f"{detail or 'not verified by indexed metadata'}"
            )

    geography = requirements.get("geography")
    if geography:
        geography_name = geography["name"]
        geography_aliases = {
            _normalized(value)
            for value in geography.get("aliases", [])
        }
        geography_aliases.add(_normalized(geography_name))
        geography_level = geography.get("level", "unknown")
        matching_dimensions = _dimension_matches(
            dimensions,
            _GEOGRAPHY_TERMS,
        )
        code_matches = [
            code
            for dimension in matching_dimensions
            for code in dimension.get("codes", [])
            if _contains(code.get("label"), geography_name)
            or _normalized(code.get("id")) in geography_aliases
        ]
        if code_matches:
            add_check(
                "geography",
                geography,
                True,
                [
                    {
                        "dimension_id": dimension.get("id"),
                        "code": code.get("id"),
                        "label": code.get("label"),
                    }
                    for dimension in matching_dimensions
                    for code in dimension.get("codes", [])
                    if code in code_matches
                ],
            )
        elif not matching_dimensions:
            add_check(
                "geography",
                geography,
                False,
                [],
                "No geographic dimension or field is indexed",
            )
        elif geography_level == "country":
            scope = (
                profile.get("title", "")
                + " "
                + profile.get("description", "")
            )
            if _contains(scope, geography_name):
                add_check(
                    "geography",
                    geography,
                    True,
                    "Dataset description explicitly names the requested country",
                    "Country values are not enumerated in the cached structure",
                )
            else:
                add_check(
                    "geography",
                    geography,
                    False,
                    [
                        dimension.get("id")
                        for dimension in matching_dimensions
                    ],
                    "Geographic field exists but the requested country is not "
                    "enumerated in cached metadata; support cannot be verified",
                )
        else:
            add_check(
                "geography",
                geography,
                False,
                [
                    dimension.get("id")
                    for dimension in matching_dimensions
                ],
                "Geographic field exists but valid values are not enumerated; "
                "support cannot be verified",
            )

    care_setting = requirements.get("care_setting")
    if care_setting:
        setting_terms = {
            "hospital": (
                "hospital",
                "inpatient",
                "outpatient",
                "hospitalization",
                "hospitalisation",
            ),
            "inpatient": (
                "inpatient",
                "in patient",
                "hospital admission",
                "hospital discharge",
                "hospitalization",
                "hospitalisation",
            ),
            "outpatient": (
                "outpatient",
                "out patient",
                "ambulatory",
            ),
        }
        matching_terms = setting_terms.get(care_setting, (care_setting,))
        supported = any(
            _contains(evidence_text, term)
            for term in matching_terms
        )
        add_check(
            "care_setting",
            care_setting,
            supported,
            [
                term
                for term in matching_terms
                if _contains(evidence_text, term)
            ],
            (
                f"Indexed metadata does not support {care_setting} care"
                if not supported
                else None
            ),
        )

    population = requirements.get("population")
    if population:
        if _contains(evidence_text, population):
            add_check(
                "population",
                population,
                True,
                "Population phrase appears in indexed metadata or field descriptions",
            )
        else:
            add_check(
                "population",
                population,
                False,
                "",
                "Requested population is not described by indexed metadata",
            )

    measure_type = requirements.get("measure_type")
    if measure_type:
        measure_dimensions = [
            dimension
            for dimension in dimensions
            if dimension.get("type") == "MeasureDimension"
            or any(
                term in _normalized(
                    f"{dimension.get('id', '')} "
                    f"{dimension.get('name', '')}"
                ).split()
                for term in ("measure", "indicator", "metric")
            )
        ]
        measure_evidence = []
        for dimension in measure_dimensions:
            codes = dimension.get("codes", [])
            if codes:
                measure_evidence.extend(
                    " ".join((
                        dimension.get("id", ""),
                        dimension.get("name", ""),
                        code.get("id", ""),
                        code.get("label", ""),
                    ))
                    for code in codes
                )
            else:
                measure_evidence.append(" ".join((
                    dimension.get("id", ""),
                    dimension.get("name", ""),
                    dimension.get("description", ""),
                    json_safe(dimension.get("metadata", {})),
                )))
        dimensions_by_id = {
            dimension.get("id"): dimension
            for dimension in dimensions
        }
        measure_evidence.extend(
            " ".join((
                concept.get("label", ""),
                json_safe(concept.get("metadata", {})),
                json_safe(
                    dimensions_by_id.get(
                        concept.get("dimension_id"),
                        {},
                    ).get("metadata", {})
                ),
            ))
            for concept in profile.get("concepts", [])
            if concept.get("dimension_id") != "__dataset__"
        )
        type_terms = {
            "count": (
                "count",
                "number of",
                "number",
                "discharges",
                "admissions",
                "visits",
                "beneficiaries",
                "events",
            ),
            "rate": (
                "rate",
                "per capita",
                "per person",
                "per inhabitant",
                "per head",
                "per 100",
                "per 1 000",
                "per 1000",
                "percent",
                "percentage",
                "proportion",
                "ratio",
                "_pct",
            ),
            "amount": (
                "amount",
                "spending",
                "payment",
                "cost",
                "charge",
                "expenditure",
                "dollar",
            ),
        }
        matching_units = [
            code
            for dimension in dimensions
            if any(
                term in _normalized(
                    f"{dimension.get('id', '')} "
                    f"{dimension.get('name', '')}"
                ).split()
                for term in ("unit", "measure")
            )
            for code in dimension.get("codes", [])
            if any(
                _contains(
                    f"{code.get('id', '')} {code.get('label', '')}",
                    term,
                )
                for term in type_terms.get(measure_type, (measure_type,))
            )
        ]
        type_supported = any(
            _supports_measure_type(text, measure_type, type_terms)
            for text in measure_evidence
        ) or bool(matching_units)
        add_check(
            "measure_type",
            measure_type,
            type_supported,
            [
                {
                    "code": code.get("id"),
                    "label": code.get("label"),
                }
                for code in matching_units
            ] or next(
                (
                    text[:500]
                    for text in measure_evidence
                    if _supports_measure_type(
                        text,
                        measure_type,
                        type_terms,
                    )
                ),
                " ".join(measure_evidence)[:500],
            ),
            (
                f"No indexed measure or unit supports {measure_type}"
                if not type_supported
                else None
            ),
        )

    start_period = requirements.get("start_period")
    end_period = requirements.get("end_period")
    if start_period or end_period:
        time_dimensions = [
            dimension
            for dimension in dimensions
            if dimension.get("type") == "TimeDimension"
            or any(
                term in _normalized(
                    f"{dimension.get('id', '')} "
                    f"{dimension.get('name', '')}"
                ).split()
                for term in _TIME_TERMS
            )
        ]
        if not time_dimensions:
            period_result, period_evidence = _periods_within(
                metadata,
                start_period,
                end_period,
            )
            single_year_snapshot = (
                start_period
                and end_period
                and str(start_period) == str(end_period)
                and requirements.get("desired_analysis") != "trend"
            )
            supported = (
                single_year_snapshot
                and period_result is True
            )
            detail = (
                "Release-period metadata supports a single-year snapshot "
                "but there is no time field for a trend"
                if supported
                else "No time dimension or period field is indexed; "
                "release metadata cannot support this requested range"
            )
            add_check(
                "time_range",
                {
                    "start_period": start_period,
                    "end_period": end_period,
                },
                supported,
                period_evidence,
                detail,
            )
        else:
            period_result, period_evidence = _periods_within(
                metadata,
                start_period,
                end_period,
            )
            if period_result is None:
                period_result = True
                period_evidence += (
                    "; indexed time dimension supports a bounded request, "
                    "but specific observation years require API validation"
                )
            add_check(
                "time_range",
                {
                    "start_period": start_period,
                    "end_period": end_period,
                },
                period_result,
                period_evidence,
            )

    unit = requirements.get("unit")
    if unit:
        unit_synonyms = {
            "per capita": (
                "per capita",
                "per person",
                "per inhabitant",
                "per head",
                "per beneficiary",
                "per 1000 persons",
                "per 100 000 persons",
            ),
            "per person": (
                "per capita",
                "per person",
                "per inhabitant",
                "per head",
                "per beneficiary",
                "per 1000 persons",
                "per 100 000 persons",
            ),
            "dollars": ("dollar", "usd", "us dollars"),
            "percent": ("percent", "percentage", "%"),
        }
        unit_terms = unit_synonyms.get(unit, (unit,))
        unit_matches = [
            code
            for dimension in dimensions
            if any(
                term in _normalized(
                    f"{dimension.get('id', '')} "
                    f"{dimension.get('name', '')}"
                ).split()
                for term in ("unit", "measure")
            )
            for code in dimension.get("codes", [])
            if any(
                _contains(
                    f"{code.get('id', '')} {code.get('label', '')}",
                    term,
                )
                for term in unit_terms
            )
        ]
        if unit_matches or any(
            _contains(evidence_text, term)
            for term in unit_terms
        ):
            add_check(
                "unit",
                unit,
                True,
                [
                    {
                        "code": code.get("id"),
                        "label": code.get("label"),
                    }
                    for code in unit_matches
                ] or "Unit phrase appears in indexed metadata or concepts",
            )
        else:
            add_check(
                "unit",
                unit,
                False,
                "",
                "Requested unit is not described by indexed concepts or codelists",
            )

    currentness = requirements.get("currentness")
    if currentness:
        is_latest = metadata.get("is_latest_release")
        if is_latest is None:
            add_check(
                "currentness",
                currentness,
                False,
                "Latest-release status is not recorded in the indexed metadata",
                "Cannot verify requested current/archive status",
            )
        else:
            supported = (
                bool(is_latest)
                if currentness == "current"
                else not bool(is_latest)
            )
            add_check(
                "currentness",
                currentness,
                supported,
                f"is_latest_release={bool(is_latest)}",
                "Dataset release does not match the requested current/archive status",
            )

    for requested_filter in requirements.get("filters", []):
        name = requested_filter["name"]
        value = requested_filter["value"]
        dimensions_for_filter = [
            dimension
            for dimension in dimensions
            if _contains(dimension.get("id"), name)
            or _contains(dimension.get("name"), name)
        ]
        if not dimensions_for_filter:
            add_check(
                f"filter:{name}",
                value,
                False,
                [],
                f"No indexed field supports requested filter {name!r}",
            )
            continue
        value_matches = [
            code
            for dimension in dimensions_for_filter
            for code in dimension.get("codes", [])
            if _contains(code.get("label"), value)
            or _normalized(code.get("id")) == _normalized(value)
        ]
        if value_matches:
            add_check(
                f"filter:{name}",
                value,
                True,
                [
                    {
                        "dimension_id": dimension.get("id"),
                        "code": code.get("id"),
                        "label": code.get("label"),
                    }
                    for dimension in dimensions_for_filter
                    for code in dimension.get("codes", [])
                    if code in value_matches
                ],
            )
        else:
            add_check(
                f"filter:{name}",
                value,
                None,
                [
                    dimension.get("id")
                    for dimension in dimensions_for_filter
                ],
                "Matching field exists; allowed values are not enumerated",
            )

    return {
        "compatible": not rejections,
        "checks": checks,
        "rejection_reasons": rejections,
        "evidence_text": evidence_normalized,
    }


def check_concept_compatibility(concept, requirements, profile=None):
    profile = profile or {}
    dimension = next(
        (
            item
            for item in profile.get("structure", {})
            .get("data_structure", {})
            .get("dimensions", [])
            if item.get("id") == concept.get("dimension_id")
        ),
        {},
    )
    text = " ".join((
        str(concept.get("label", "")),
        str(concept.get("semantic_text", "")),
        json_safe(concept.get("metadata", {})),
        str(concept.get("dimension_name", "")),
        str(dimension.get("name", "")),
        str(dimension.get("description", "")),
        json_safe(dimension.get("metadata", {})),
    ))
    normalized = _normalized(text)
    checks = []
    rejections = []
    if concept.get("dimension_id") == "__dataset__":
        return {
            "compatible": False,
            "checks": [{
                "characteristic": "concept_kind",
                "requested": "measure or semantic field",
                "supported": False,
                "evidence": "Dataset-level summary is not a measure/field",
            }],
            "rejection_reasons": [
                "Dataset-level summary is not a measure or semantic field"
            ],
            "normalized_text": normalized,
            "measure_classification": classify_measure(concept, profile),
        }

    measure_classification = classify_measure(concept, profile)
    requested_family = requirements.get("measure_family")
    if requested_family:
        observed_family = measure_classification["family"]
        supported = observed_family == requested_family
        checks.append({
            "characteristic": "measure_family",
            "requested": requested_family,
            "observed": observed_family,
            "supported": supported,
            "evidence": measure_classification["evidence"],
        })
        if not supported:
            if observed_family == "other_outcome":
                rejections.append(
                    f"Measure family is unknown or unclassified; "
                    f"cannot verify requested {requested_family}"
                )
            else:
                rejections.append(
                    f"Measure family {observed_family} is incompatible "
                    f"with requested {requested_family}"
                )

    requested_quantity = requirements.get("quantity_type")
    if requested_quantity:
        observed_quantity = measure_classification["quantity_type"]
        quantity_compatible = (
            observed_quantity == requested_quantity
            or (
                requested_quantity == "rate"
                and observed_quantity in (
                    "per_capita",
                    "standardized_rate",
                )
            )
        )
        checks.append({
            "characteristic": "quantity_type",
            "requested": requested_quantity,
            "observed": observed_quantity,
            "supported": quantity_compatible,
            "evidence": measure_classification["evidence"],
        })
        if not quantity_compatible:
            rejections.append(
                f"Quantity type {observed_quantity} is incompatible "
                f"with requested {requested_quantity}"
            )

    requested_financial_measure = requirements.get("financial_measure")
    if requested_financial_measure:
        observed_financial_measure = measure_classification[
            "financial_measure"
        ]
        supported = observed_financial_measure == requested_financial_measure
        checks.append({
            "characteristic": "financial_measure",
            "requested": requested_financial_measure,
            "observed": observed_financial_measure,
            "supported": supported,
            "evidence": measure_classification["evidence"],
        })
        if not supported:
            rejections.append(
                f"Financial measure {observed_financial_measure or 'unknown'} "
                f"does not match requested {requested_financial_measure}"
            )

    requested_provider_type = requirements.get("provider_type")
    if requested_provider_type:
        observed_provider_type = measure_classification["provider_type"]
        supported = observed_provider_type == requested_provider_type
        checks.append({
            "characteristic": "provider_type",
            "requested": requested_provider_type,
            "observed": observed_provider_type,
            "supported": supported,
            "evidence": measure_classification["evidence"],
        })
        if not supported:
            rejections.append(
                f"Provider measure type "
                f"{observed_provider_type or 'unknown'} does not match "
                f"requested {requested_provider_type}"
            )

    requested_domain = requirements.get("measure_domain")
    if requested_domain:
        observed_domain = measure_classification["domain"]
        supported = observed_domain == requested_domain
        checks.append({
            "characteristic": "measure_domain",
            "requested": requested_domain,
            "observed": observed_domain,
            "supported": supported,
            "evidence": measure_classification["evidence"],
        })
        if not supported:
            rejections.append(
                f"Measure does not verify the requested {requested_domain} domain"
            )

    requested_condition = requirements.get("specific_condition")
    observed_condition = measure_classification["condition"]
    if requested_condition:
        supported = observed_condition == requested_condition
        checks.append({
            "characteristic": "specific_condition",
            "requested": requested_condition,
            "observed": observed_condition,
            "supported": supported,
            "evidence": measure_classification["evidence"],
        })
        if not supported:
            rejections.append(
                f"Measure condition {observed_condition or 'unknown'} "
                f"does not match requested {requested_condition}"
            )
    elif requested_domain == "behavioral_health" and observed_condition:
        checks.append({
            "characteristic": "condition_scope",
            "requested": "broad behavioral-health measure",
            "observed": observed_condition,
            "supported": False,
            "evidence": measure_classification["evidence"],
        })
        rejections.append(
            f"Condition-specific {observed_condition} measure cannot "
            "substitute for broad behavioral-health utilization"
        )

    care_setting = requirements.get("care_setting")
    if care_setting:
        setting_terms = {
            "hospital": (
                "hospital",
                "inpatient",
                "outpatient",
                "hospitalization",
                "hospitalisation",
            ),
            "inpatient": (
                "inpatient",
                "in patient",
                "hospital admission",
                "hospital discharge",
                "hospitalization",
                "hospitalisation",
            ),
            "outpatient": (
                "outpatient",
                "out patient",
                "ambulatory",
            ),
        }
        context = " ".join((
            profile.get("title", ""),
            profile.get("description", ""),
            text,
        ))
        supported = any(
            _contains(context, term)
            for term in setting_terms.get(care_setting, (care_setting,))
        )
        checks.append({
            "characteristic": "care_setting",
            "requested": care_setting,
            "supported": supported,
            "evidence": context[:500],
        })
        if not supported:
            rejections.append(
                f"Selected dataset or concept does not support "
                f"{care_setting} care"
            )

    topic = requirements.get("topic")
    topic_terms = {
        "utilization": (
            "utilization",
            "admission",
            "admissions",
            "discharge",
            "discharges",
            "visit",
            "visits",
            "stay",
            "stays",
            "service use",
            "hospital use",
        ),
        "spending": (
            "spending",
            "payment",
            "cost",
            "charge",
            "expenditure",
            "dollar",
        ),
    }
    if topic:
        supported = any(
            _contains(text, term)
            for term in topic_terms.get(topic, (topic,))
        )
        checks.append({
            "characteristic": "topic",
            "requested": topic,
            "supported": supported,
            "evidence": concept.get("label"),
        })
        if not supported:
            rejections.append(
                f"Selected concept does not describe {topic}"
            )

    measure_type = requirements.get("measure_type")
    type_terms = {
        "count": (
            "count",
            "number of",
            "discharges",
            "admissions",
            "visits",
            "beneficiaries",
            "events",
        ),
        "rate": (
            "rate",
            "per capita",
            "per person",
            "per 100",
            "per 1 000",
            "per 1000",
            "percent",
            "percentage",
            "proportion",
            "ratio",
        ),
        "amount": (
            "amount",
            "spending",
            "payment",
            "cost",
            "charge",
            "expenditure",
            "dollar",
        ),
    }
    if measure_type:
        dimension_name = _normalized(
            f"{concept.get('dimension_id', '')} "
            f"{concept.get('dimension_name', '')} "
            f"{dimension.get('name', '')}"
        )
        concept_text = " ".join((
            text,
            dimension_name,
        ))
        supported = _supports_measure_type(
            concept_text,
            measure_type,
            type_terms,
        )
        checks.append({
            "characteristic": "measure_type",
            "requested": measure_type,
            "supported": supported,
            "evidence": concept.get("label"),
        })
        if not supported:
            rejections.append(
                f"Selected concept does not support measure type {measure_type}"
            )

    adjustment = requirements.get("adjustment")
    if adjustment:
        if adjustment == "standardized":
            terms = ("standardized", "standardised", "risk adjusted", "age adjusted")
        else:
            terms = ("crude", "unadjusted", "not adjusted")
        supported = any(_contains(text, term) for term in terms)
        checks.append({
            "characteristic": "adjustment",
            "requested": adjustment,
            "supported": supported,
            "evidence": concept.get("label"),
        })
        if not supported:
            rejections.append(
                f"Selected concept does not explicitly identify a {adjustment} measure"
            )

    unit = requirements.get("unit")
    if unit:
        unit_synonyms = {
            "per capita": (
                "per capita",
                "per person",
                "per inhabitant",
                "per head",
                "per beneficiary",
                "per 1000 persons",
                "per 100 000 persons",
            ),
            "per person": (
                "per capita",
                "per person",
                "per inhabitant",
                "per head",
                "per beneficiary",
            ),
            "dollars": ("dollar", "usd", "us dollars"),
            "percent": ("percent", "percentage", "%"),
        }
        terms = unit_synonyms.get(unit, (unit,))
        units_dimension = any(
            any(
                term in _normalized(
                    f"{item.get('id', '')} {item.get('name', '')}"
                ).split()
                for term in ("unit", "measure")
            )
            for item in profile.get("structure", {})
            .get("data_structure", {})
            .get("dimensions", [])
        )
        supported = any(_contains(text, term) for term in terms)
        if units_dimension:
            supported = supported or any(
                _contains(code.get("label"), term)
                for item in profile.get("structure", {})
                .get("data_structure", {})
                .get("dimensions", [])
                for code in item.get("codes", [])
                for term in terms
            )
        checks.append({
            "characteristic": "unit",
            "requested": unit,
            "supported": supported,
            "evidence": concept.get("label"),
        })
        if not supported:
            rejections.append(
                f"Selected concept does not explicitly identify unit {unit!r}"
            )

    return {
        "compatible": not rejections,
        "checks": checks,
        "rejection_reasons": rejections,
        "normalized_text": normalized,
        "measure_classification": measure_classification,
    }


def json_safe(value):
    if isinstance(value, dict):
        return " ".join(
            f"{key} {json_safe(item)}"
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return " ".join(json_safe(item) for item in value)
    return str(value or "")
