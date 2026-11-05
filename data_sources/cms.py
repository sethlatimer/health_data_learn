import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

import pandas as pd
import requests

import embedding
from data_sources.base import DataSource
from database import (
    get_dataset_record,
    get_dataset_structure,
    get_source_catalog_cache,
    save_source_catalog_cache,
)


CMS_BASE_URL = "https://data.cms.gov"
CMS_CATALOG_URL = f"{CMS_BASE_URL}/data.json"
CMS_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
CMS_MAX_REQUEST_ATTEMPTS = 4
CMS_CATALOG_CACHE_TTL_SECONDS = 24 * 60 * 60
CMS_MAX_PAGE_SIZE = 5000
CMS_HEALTHCARE_TERMS = (
    "health",
    "healthcare",
    "hospital",
    "inpatient",
    "outpatient",
    "medicare",
    "provider",
    "physician",
    "patient",
    "clinical",
    "medical",
    "beneficiary",
    "claim",
    "payment",
    "spending",
    "utilization",
    "nursing",
    "care",
)
CMS_MEASURE_TERMS = (
    "amount",
    "amt",
    "payment",
    "payments",
    "spend",
    "spending",
    "cost",
    "rate",
    "count",
    "cnt",
    "utilization",
    "visits",
    "admissions",
    "discharges",
    "beneficiaries",
    "services",
    "volume",
    "total",
    "measure",
    "score",
)
CMS_PRIMARY_FILTER_TERMS = (
    "id",
    "identifier",
    "code",
    "ccn",
    "state",
    "county",
    "zip",
    "geography",
    "region",
    "date",
    "year",
    "quarter",
    "period",
    "category",
    "type",
    "status",
    "specialty",
)


def _retry_delay(response, attempt):
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return max(0, float(retry_after))
        except ValueError:
            try:
                retry_time = parsedate_to_datetime(retry_after)
                if retry_time.tzinfo is None:
                    retry_time = retry_time.replace(tzinfo=timezone.utc)
                return max(
                    0,
                    (retry_time - datetime.now(timezone.utc)).total_seconds(),
                )
            except (TypeError, ValueError, OverflowError):
                pass
    return 2 ** (attempt + 1)


def _get_cms_response(url, params=None, headers=None, timeout=60):
    for attempt in range(CMS_MAX_REQUEST_ATTEMPTS):
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=timeout,
        )
        if response.status_code not in CMS_RETRYABLE_STATUSES:
            response.raise_for_status()
            return response
        if attempt == CMS_MAX_REQUEST_ATTEMPTS - 1:
            response.raise_for_status()
        time.sleep(_retry_delay(response, attempt))
    raise RuntimeError("CMS request retries exhausted")


def _dataset_id_from_api_url(url):
    path = urlparse(url).path
    match = re.search(
        r"/data-api/v1/dataset/([0-9a-fA-F-]{36})/"
        r"(?:data|data-viewer)$",
        path,
    )
    return match.group(1).lower() if match else None


def _api_distribution(dataset):
    distributions = dataset.get("distribution")
    if not isinstance(distributions, list):
        return None

    matching = [
        item
        for item in distributions
        if isinstance(item, dict)
        and item.get("format") == "API"
        and _dataset_id_from_api_url(item.get("accessURL", ""))
    ]
    if not matching:
        return None

    dataset_id = _dataset_id_from_api_url(dataset.get("identifier", ""))
    matching.sort(
        key=lambda item: (
            _dataset_id_from_api_url(item.get("accessURL", "")) != dataset_id,
            item.get("description") != "latest",
        )
    )
    return matching[0]


def _catalog_timestamp():
    return datetime.now(timezone.utc).isoformat()


def _is_recent(timestamp, now=None):
    try:
        fetched_at = datetime.fromisoformat(timestamp)
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return False
    current_time = now or datetime.now(timezone.utc)
    return (
        current_time - fetched_at
    ).total_seconds() < CMS_CATALOG_CACHE_TTL_SECONDS


def _healthcare_score(dataset):
    title = str(dataset.get("title", "")).casefold()
    description = str(dataset.get("description", "")).casefold()
    keywords = " ".join(
        str(value)
        for key in ("keyword", "theme")
        for value in (dataset.get(key) or [])
    ).casefold()
    searchable = f"{title} {description} {keywords}"
    if not any(term in searchable for term in CMS_HEALTHCARE_TERMS):
        return 0

    return (
        4 * sum(term in title for term in CMS_HEALTHCARE_TERMS)
        + 2 * sum(term in description for term in CMS_HEALTHCARE_TERMS)
        + sum(term in keywords for term in CMS_HEALTHCARE_TERMS)
    )


def _series_key(dataset):
    in_series = dataset.get("inSeries")
    if isinstance(in_series, list) and in_series:
        series_id = in_series[0].get("@id")
        if series_id:
            return series_id
    return re.sub(
        r"\s*[:\-]\s*(?:19|20)\d{2}[-/]\d{2}[-/]\d{2}\s*$",
        "",
        str(dataset.get("title", "")),
        flags=re.IGNORECASE,
    ).casefold()


def _release_order(dataset):
    temporal = dataset.get("temporal") or []
    end_date = temporal[0].get("endDate", "") if temporal else ""
    return end_date, dataset.get("modified", "")


def _series_latest_ids(catalog):
    latest_ids = {}
    for series in catalog.get("datasetSeries", []):
        if not isinstance(series, dict):
            continue
        series_id = series.get("@id")
        last_entry = series.get("last") or {}
        latest_id = _dataset_id_from_api_url(last_entry.get("@id", ""))
        if series_id and latest_id:
            latest_ids[series_id] = latest_id
    return latest_ids


def _is_filter_field(field):
    name = str(field.get("name", "")).casefold()
    label = str(field.get("label", "")).casefold()
    description = str(field.get("description", "")).casefold()
    searchable = re.sub(r"[^a-z0-9]+", " ", f"{name} {label}")
    words = set(searchable.split())
    if words.intersection(CMS_PRIMARY_FILTER_TERMS):
        return True
    if any(term in description for term in CMS_MEASURE_TERMS):
        return False
    if words.intersection(CMS_MEASURE_TERMS):
        return False
    return True


class CMSDataSource(DataSource):
    source = "CMS"

    def __init__(
        self,
        base_url=CMS_BASE_URL,
        catalog_url=CMS_CATALOG_URL,
    ):
        self.base_url = base_url.rstrip("/")
        self.catalog_url = catalog_url
        self._discovered = {}

    def _catalog(self):
        cached = get_source_catalog_cache(self.source)
        if cached is not None and _is_recent(cached["fetched_at"]):
            return cached["payload"]

        headers = {}
        if cached is not None and cached.get("etag"):
            headers["If-None-Match"] = cached["etag"]
        response = _get_cms_response(
            self.catalog_url,
            headers=headers,
            timeout=120,
        )
        if response.status_code == 304 and cached is not None:
            save_source_catalog_cache(
                self.source,
                cached["payload"],
                fetched_at=_catalog_timestamp(),
                etag=response.headers.get("ETag") or cached.get("etag"),
                last_modified=(
                    response.headers.get("Last-Modified")
                    or cached.get("last_modified")
                ),
            )
            return cached["payload"]
        try:
            catalog = response.json()
        except requests.exceptions.JSONDecodeError as error:
            raise ValueError("CMS catalog did not return valid JSON") from error
        if (
            not isinstance(catalog, dict)
            or not isinstance(catalog.get("dataset"), list)
            or not isinstance(catalog.get("datasetSeries"), list)
        ):
            raise ValueError(
                "CMS DCAT-US catalog must include dataset and datasetSeries arrays"
            )
        save_source_catalog_cache(
            self.source,
            catalog,
            fetched_at=_catalog_timestamp(),
            etag=response.headers.get("ETag"),
            last_modified=response.headers.get("Last-Modified"),
        )
        return catalog

    def discover_datasets(self, limit=None):
        if limit is not None and limit < 1:
            raise ValueError("limit must be greater than zero")

        catalog = self._catalog()
        latest_ids = _series_latest_ids(catalog)
        latest_by_series = {}
        for dataset in catalog["dataset"]:
            if not isinstance(dataset, dict):
                continue
            api_distribution = _api_distribution(dataset)
            if api_distribution is None:
                continue
            score = _healthcare_score(dataset)
            if score == 0:
                continue
            dataset_id = _dataset_id_from_api_url(
                api_distribution["accessURL"]
            )
            if not dataset_id:
                continue
            series_key = _series_key(dataset)
            candidate = {
                "source": self.source,
                "dataset_id": dataset_id,
                "title": dataset.get("title") or dataset_id,
                "description": dataset.get("description", ""),
                "source_url": api_distribution["accessURL"],
                "healthcare_score": score,
                "raw": dataset,
                "api_distribution": api_distribution,
                "is_latest_release": (
                    latest_ids[series_key]
                    == dataset_id
                    if series_key in latest_ids
                    else None
                ),
            }
            existing = latest_by_series.get(series_key)
            if existing is None or _release_order(dataset) > existing[0]:
                latest_by_series[series_key] = (
                    _release_order(dataset),
                    candidate,
                )

        datasets = [
            item[1]
            for item in latest_by_series.values()
        ]
        datasets.sort(
            key=lambda item: (
                -item["healthcare_score"],
                item["title"].casefold(),
            )
        )
        if limit is not None:
            datasets = datasets[:limit]
        self._discovered = {
            dataset["dataset_id"]: dataset
            for dataset in datasets
        }
        return datasets

    def get_metadata(self, dataset_id):
        discovered = self._discovered.get(dataset_id)
        if discovered is not None:
            return discovered
        cached = get_dataset_record(self.source, dataset_id)
        if cached is not None:
            raw_metadata = cached.get("metadata", {})
            return {
                **cached,
                "raw": raw_metadata,
                "api_distribution": _api_distribution(raw_metadata),
            }
        raise LookupError(
            f"CMS dataset {dataset_id!r} is not in the discovered or indexed catalog"
        )

    def get_structure(self, dataset_id, metadata=None):
        metadata = metadata or self.get_metadata(dataset_id)
        cached = get_dataset_structure(dataset_id, source=self.source)
        modified = (metadata.get("raw") or {}).get("modified")
        if (
            cached is not None
            and cached.get("metadata_modified") == modified
        ):
            return cached

        distribution = metadata.get("api_distribution")
        if distribution is None:
            distribution = _api_distribution(metadata.get("raw", {}))
        if distribution is None:
            raise ValueError(
                f"CMS dataset {dataset_id!r} has no API distribution"
            )
        dictionary_ref = distribution.get("describedBy", {})
        dictionary_url = dictionary_ref.get("downloadURL")
        dictionary = None
        if dictionary_url:
            try:
                response = _get_cms_response(dictionary_url, timeout=60)
                dictionary = response.json()
            except requests.HTTPError as error:
                response = error.response
                body = response.text if response is not None else ""
                if (
                    response is None
                    or response.status_code != 404
                    or "does not have an interactive dictionary"
                    not in body.casefold()
                ):
                    raise

        dictionary_data = (
            dictionary.get("data", {})
            if isinstance(dictionary, dict)
            else {}
        )
        fields = dictionary_data.get("fields", [])
        if fields is None:
            fields = []
        if not isinstance(fields, list):
            raise ValueError(
                f"CMS dictionary for {dataset_id!r} has malformed fields"
            )

        dimensions = []
        for position, field in enumerate(fields, start=1):
            if not isinstance(field, dict) or not field.get("name"):
                raise ValueError(
                    f"CMS dictionary for {dataset_id!r} contains a field "
                    "without its original name"
                )
            name = field["name"]
            dimensions.append({
                "id": name,
                "name": field.get("label") or name,
                "type": (
                    "MeasureDimension"
                    if not _is_filter_field(field)
                    else "Dimension"
                ),
                "position": position,
                "codelist_id": None,
                "codes": [],
                "description": field.get("description", ""),
                "metadata": field,
            })

        raw_metadata = metadata.get("raw", {})
        structure = {
            "source": self.source,
            "dataset_id": dataset_id,
            "metadata_modified": modified,
            "dataflow": {
                "id": dataset_id,
                "agency_id": "CMS",
                "version": str(raw_metadata.get("modified", "")),
                "name": metadata.get("title") or dataset_id,
                "description": metadata.get("description", ""),
                "source_url": metadata.get("source_url"),
                "identifier": raw_metadata.get("identifier"),
                "metadata": raw_metadata,
            },
            "data_structure": {
                "id": (
                    dictionary.get("identifier", "")
                    if isinstance(dictionary, dict)
                    else ""
                ),
                "agency_id": "CMS",
                "version": str(raw_metadata.get("modified", "")),
                "dimensions": dimensions,
            },
            "codelists": [],
            "cms_dictionary": dictionary,
        }
        return structure

    def build_concepts(self, dataset_id, metadata, structure):
        raw_metadata = metadata.get("raw", {})
        keywords = raw_metadata.get("keyword") or []
        themes = raw_metadata.get("theme") or []
        dataset_text_parts = [
            f"CMS dataset: {metadata.get('title') or dataset_id}",
        ]
        description = metadata.get("description")
        if description:
            dataset_text_parts.append(f"Description: {description}")
        if keywords:
            dataset_text_parts.append(
                "Keywords: " + "; ".join(str(item) for item in keywords)
            )
        if themes:
            dataset_text_parts.append(
                "Themes: " + "; ".join(str(item) for item in themes)
            )

        concepts = [{
            "source": self.source,
            "dataset_id": dataset_id,
            "dimension_id": "__dataset__",
            "dimension_name": "Dataset",
            "code": dataset_id,
            "label": metadata.get("title") or dataset_id,
            "concept_role": "semantic",
            "semantic_text": "\n".join(dataset_text_parts),
            "metadata": {
                "dataset_identifier": raw_metadata.get("identifier"),
                "source_url": metadata.get("source_url"),
                "keywords": keywords,
                "themes": themes,
                "description": description,
            },
        }]

        for dimension in structure["data_structure"]["dimensions"]:
            field = dimension["metadata"]
            role = "filter" if _is_filter_field(field) else "semantic"
            label = field.get("label") or field["name"]
            semantic_text = None
            if role == "semantic":
                lines = [
                    f"CMS dataset: {metadata.get('title') or dataset_id}",
                    f"Field: {field['name']}",
                    f"Label: {label}",
                ]
                if field.get("description"):
                    lines.append(
                        f"Description: {field['description']}"
                    )
                semantic_text = "\n".join(lines)
            concepts.append({
                "source": self.source,
                "dataset_id": dataset_id,
                "dimension_id": field["name"],
                "dimension_name": field.get("label") or field["name"],
                "code": field["name"],
                "label": label,
                "concept_role": role,
                "semantic_text": semantic_text,
                "metadata": field,
            })
        return concepts

    def find_filters(self, dataset_id, dimension_id=None, search=None):
        return embedding.find_dataset_filters(
            dataset_id,
            dimension_id=dimension_id,
            search=search,
            source=self.source,
        )

    def retrieve_data(
        self,
        dataset_id,
        filters=None,
        start_period=None,
        end_period=None,
        limit=1000,
    ):
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise TypeError("limit must be an integer")
        if not 1 <= limit <= CMS_MAX_PAGE_SIZE:
            raise ValueError(
                f"limit must be between 1 and {CMS_MAX_PAGE_SIZE}"
            )
        if not isinstance(filters or {}, dict):
            raise TypeError("filters must map CMS field names to exact values")
        if start_period is not None or end_period is not None:
            raise NotImplementedError(
                "CMS period ranges must be passed as exact field filters; "
                "the adapter does not assume CMS range operators"
            )

        structure = self.get_structure(dataset_id)
        fields = {
            dimension["id"]
            for dimension in structure["data_structure"]["dimensions"]
        }
        requested_filters = filters or {}
        unknown = set(requested_filters) - fields
        if unknown:
            raise ValueError(
                "Unknown CMS dataset field(s): "
                + ", ".join(sorted(unknown))
            )

        params = {
            "size": limit,
            "offset": 0,
        }
        params.update({
            f"filter[{field}]": value
            for field, value in requested_filters.items()
        })
        metadata = self.get_metadata(dataset_id)
        distribution = metadata.get("api_distribution")
        if distribution is None:
            distribution = _api_distribution(metadata.get("raw", {}))
        if distribution is None:
            raise ValueError(
                f"CMS dataset {dataset_id!r} has no API distribution"
            )
        response = _get_cms_response(
            distribution["accessURL"],
            params=params,
            timeout=60,
        )
        try:
            payload = response.json()
        except requests.exceptions.JSONDecodeError as error:
            raise ValueError(
                f"CMS data endpoint for {dataset_id!r} did not return JSON"
            ) from error
        if not isinstance(payload, list):
            raise ValueError(
                f"CMS data endpoint for {dataset_id!r} did not return a row array"
            )
        if len(payload) > limit:
            raise ValueError(
                f"CMS API returned {len(payload)} rows for a limit of {limit}"
            )
        return {
            "source": self.source,
            "dataset_id": dataset_id,
            "request": {
                "url": response.url,
                "params": params,
                "filters": requested_filters,
                "limit": limit,
                "offset": 0,
            },
            "rows": payload,
        }

    def normalize_dataframe(self, result, structure):
        if not isinstance(result, dict):
            raise TypeError("CMS result must be a mapping")
        if result.get("source") != self.source:
            raise ValueError("CMS result has an unexpected source")
        rows = result.get("rows")
        if not isinstance(rows, list) or any(
            not isinstance(row, dict) for row in rows
        ):
            raise ValueError("CMS result rows must be a list of objects")
        dataframe = pd.DataFrame.from_records(rows)
        dataframe.insert(0, "dataset_id", result["dataset_id"])
        dataframe.insert(1, "source", self.source)
        return dataframe

    def get_provenance(self, dataset_id, structure, result):
        dataflow = structure["dataflow"]
        return {
            "source": self.source,
            "dataset_id": dataset_id,
            "dataset_title": dataflow["name"],
            "source_url": dataflow.get("source_url"),
            "cms_identifier": dataflow.get("identifier"),
            "modified": structure.get("metadata_modified"),
            "data_dictionary_id": structure["data_structure"]["id"],
            "field_count": len(
                structure["data_structure"]["dimensions"]
            ),
            "request": dict(result["request"]),
        }

    def check_compatibility(self, profile, requirements):
        profile = dict(profile)
        metadata = dict(profile.get("metadata") or {})
        if profile.get("is_latest_release") is not None:
            metadata["is_latest_release"] = profile["is_latest_release"]
        profile["metadata"] = metadata

        result = super().check_compatibility(profile, requirements)
        geography = requirements.get("geography")
        if geography and geography.get("level") == "country":
            normalized_country = re.sub(
                r"[^a-z0-9]+",
                "",
                geography["name"].casefold(),
            )
            if normalized_country not in {
                "unitedstates",
                "unitedstatesofamerica",
                "us",
                "usa",
            }:
                reason = (
                    "CMS Medicare provider/hospital catalog records cover "
                    "the United States, not the requested country"
                )
                result["compatible"] = False
                result["checks"].append({
                    "characteristic": "geography_scope",
                    "requested": geography,
                    "supported": False,
                    "evidence": "CMS U.S. federal Medicare dataset",
                    "detail": reason,
                })
                result["rejection_reasons"].append(reason)
            else:
                geography_verified_by_scope = False
                for check in result["checks"]:
                    if (
                        check["characteristic"] == "geography"
                        and check["supported"] is False
                        and str(check.get("detail", "")).startswith(
                            "Geographic field exists"
                        )
                    ):
                        check["supported"] = True
                        check["evidence"] = (
                            "CMS Medicare provider/hospital catalog records "
                            "cover the United States"
                        )
                        check["detail"] = (
                            "U.S. geographic scope is established by the "
                            "official CMS dataset catalog"
                        )
                        geography_verified_by_scope = True
                if geography_verified_by_scope:
                    result["rejection_reasons"] = [
                        reason
                        for reason in result["rejection_reasons"]
                        if not reason.startswith("geography:")
                    ]
                    result["compatible"] = not result["rejection_reasons"]
        return result
