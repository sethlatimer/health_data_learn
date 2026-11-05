import hashlib
import json
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote, urljoin, urlparse

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


WHO_API_BASE_URL = "https://ghoapi.azureedge.net/api"
WHO_CATALOG_TTL_SECONDS = 24 * 60 * 60
WHO_PAGE_SIZE = 1000
WHO_MAX_FILTER_CODES = 500
WHO_MAX_PAGE_SIZE = 5000
WHO_MAX_INDEX_DATASETS = 100
WHO_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
WHO_DIMENSION_TYPES = (
    ("SpatialDimType", "SpatialDim", "geography"),
    ("TimeDimType", "TimeDim", "time"),
    ("Dim1Type", "Dim1", "stratum"),
    ("Dim2Type", "Dim2", "stratum"),
    ("Dim3Type", "Dim3", "stratum"),
    ("DataSourceDimType", "DataSourceDim", "data_source"),
)
WHO_UNIT_FIELDS = (
    "Unit",
    "UnitName",
    "UnitOfMeasure",
    "UnitType",
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


def _get_who_response(url, params=None, headers=None, timeout=60):
    for attempt in range(4):
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=timeout,
        )
        if response.status_code not in WHO_RETRYABLE_STATUSES:
            response.raise_for_status()
            return response
        if attempt == 3:
            response.raise_for_status()
        time.sleep(_retry_delay(response, attempt))
    raise RuntimeError("WHO request retries exhausted")


def _odata_rows(payload):
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise ValueError("WHO OData response must be an object or array")
    rows = payload.get("value")
    if isinstance(rows, list):
        return rows
    legacy = payload.get("d")
    if isinstance(legacy, dict) and isinstance(legacy.get("results"), list):
        return legacy["results"]
    raise ValueError("WHO OData response does not contain a result array")


def _odata_next_link(payload):
    if not isinstance(payload, dict):
        return None
    link = payload.get("@odata.nextLink") or payload.get("odata.nextLink")
    if link:
        return link
    legacy = payload.get("d")
    return legacy.get("__next") if isinstance(legacy, dict) else None


def _json_payload(response, description):
    try:
        return response.json()
    except requests.exceptions.JSONDecodeError as error:
        raise ValueError(f"{description} did not return valid JSON") from error


def _indicator_code(indicator):
    value = indicator.get("IndicatorCode")
    return str(value).strip() if value is not None else ""


def _indicator_name(indicator, dataset_id):
    return str(
        indicator.get("IndicatorName")
        or indicator.get("Title")
        or dataset_id
    ).strip()


def _indicator_description(indicator):
    parts = []
    for key in (
        "Definition",
        "Description",
        "Rationale",
        "MethodOfMeasurement",
        "Methodology",
    ):
        value = indicator.get(key)
        if value and str(value).strip() not in parts:
            parts.append(str(value).strip())
    return "\n\n".join(parts)


def _dimension_label(kind, field_id):
    if field_id == "SpatialDim":
        return {
            "COUNTRY": "Country",
            "REGION": "Region",
            "WORLD_BANK_INCOME_GROUP": "World Bank income group",
        }.get(str(kind or "").upper(), str(kind or "Geography"))
    if field_id == "TimeDim":
        return "Time period"
    if field_id == "DataSourceDim":
        return "Data source"
    labels = {
        "SEX": "Sex",
        "AGEGROUP": "Age group",
        "AGE_GROUP": "Age group",
        "AGE": "Age",
        "POPULATION": "Population group",
        "COVERAGE": "Coverage",
        "WEALTHQUINTILE": "Wealth quintile",
        "WEALTH_QUINTILE": "Wealth quintile",
        "EDUCATION": "Education group",
        "RESIDENCEAREATYPE": "Residence area type",
        "RESIDENCE_AREA_TYPE": "Residence area type",
    }
    return labels.get(
        str(kind or "").upper(),
        str(kind or field_id),
    )


def _dimension_code_label(row):
    code = (
        row.get("Code")
        or row.get("DimensionValue")
        or row.get("Value")
        or row.get("Id")
    )
    label = (
        row.get("Title")
        or row.get("Name")
        or row.get("Label")
        or row.get("Description")
        or code
    )
    return code, label


def _catalog_timestamp():
    return datetime.now(timezone.utc).isoformat()


def _cache_is_fresh(timestamp):
    try:
        fetched_at = datetime.fromisoformat(timestamp)
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return False
    return (
        datetime.now(timezone.utc) - fetched_at
    ).total_seconds() < WHO_CATALOG_TTL_SECONDS


def _metadata_signature(metadata):
    serialized = json.dumps(
        metadata,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class WHODataSource(DataSource):
    source = "WHO"

    def __init__(self, api_base_url=WHO_API_BASE_URL):
        self.api_base_url = api_base_url.rstrip("/")
        self._discovered = {}
        self._dimension_values = {}

    def _catalog(self, limit=None):
        cached = get_source_catalog_cache(self.source)
        if cached is not None and _cache_is_fresh(cached["fetched_at"]):
            payload = cached["payload"]
            indicators = payload.get("indicators")
            if isinstance(indicators, list):
                return indicators[:limit] if limit is not None else indicators

        page_size = min(limit, WHO_PAGE_SIZE) if limit else WHO_PAGE_SIZE
        indicators = []
        seen_codes = set()
        offset = 0
        next_url = None
        while True:
            params = None if next_url else {
                "$format": "json",
                "$top": page_size,
                "$skip": offset,
            }
            response = _get_who_response(
                next_url or f"{self.api_base_url}/Indicator",
                params=params,
                headers={"Accept": "application/json"},
                timeout=120,
            )
            payload = _json_payload(response, "WHO indicator catalog")
            rows = _odata_rows(payload)
            for indicator in rows:
                if not isinstance(indicator, dict):
                    raise ValueError(
                        "WHO indicator catalog contains a non-object record"
                    )
                code = _indicator_code(indicator)
                if not code or code in seen_codes:
                    continue
                seen_codes.add(code)
                indicators.append(indicator)
                if limit is not None and len(indicators) >= limit:
                    break

            if limit is not None and len(indicators) >= limit:
                break
            next_link = _odata_next_link(payload)
            if next_link:
                resolved = urljoin(self.api_base_url + "/", next_link)
                if urlparse(resolved).netloc != urlparse(
                    self.api_base_url
                ).netloc:
                    raise ValueError(
                        "WHO indicator pagination returned an external URL"
                    )
                next_url = resolved
                continue
            next_url = None
            if len(rows) < page_size:
                break
            offset += len(rows)
            if offset >= 50000:
                raise ValueError(
                    "WHO indicator catalog exceeded the 50,000-record safety bound"
                )

        if limit is None:
            save_source_catalog_cache(
                self.source,
                {"indicators": indicators},
                fetched_at=_catalog_timestamp(),
            )
        return indicators

    def discover_datasets(self, limit=None):
        if limit is not None and (
            not isinstance(limit, int) or isinstance(limit, bool) or limit < 1
        ):
            raise ValueError("limit must be a positive integer")
        indicators = self._catalog(limit=limit)
        datasets = []
        self._discovered = {}
        for indicator in indicators:
            dataset_id = _indicator_code(indicator)
            if not dataset_id:
                continue
            dataset = self._normalize_metadata(dataset_id, indicator)
            datasets.append(dataset)
            self._discovered[dataset_id] = dataset
        return datasets

    def _normalize_metadata(self, dataset_id, raw):
        title = _indicator_name(raw, dataset_id)
        api_url = (
            f"{self.api_base_url}/"
            f"{quote(dataset_id, safe='._-')}"
        )
        source_url = raw.get("URL") or api_url
        return {
            "source": self.source,
            "dataset_id": dataset_id,
            "title": title,
            "description": _indicator_description(raw),
            "source_url": source_url,
            "api_url": api_url,
            "raw": dict(raw),
            "is_latest_release": None,
        }

    def get_metadata(self, dataset_id):
        discovered = self._discovered.get(dataset_id)
        if discovered is not None:
            return discovered
        cached = get_dataset_record(self.source, dataset_id)
        if cached is not None:
            raw = cached.get("metadata", {})
            return self._normalize_metadata(dataset_id, raw)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(dataset_id)):
            raise ValueError(f"Invalid WHO indicator code {dataset_id!r}")
        response = _get_who_response(
            f"{self.api_base_url}/Indicator",
            params={
                "$format": "json",
                "$filter": f"IndicatorCode eq '{dataset_id}'",
                "$top": 2,
            },
            headers={"Accept": "application/json"},
            timeout=60,
        )
        rows = _odata_rows(_json_payload(response, "WHO indicator metadata"))
        match = next(
            (
                row
                for row in rows
                if _indicator_code(row) == dataset_id
            ),
            None,
        )
        if match is None:
            raise LookupError(
                f"WHO indicator {dataset_id!r} was not found"
            )
        metadata = self._normalize_metadata(dataset_id, match)
        self._discovered[dataset_id] = metadata
        return metadata

    def _load_dimension_values(self, dimension_type):
        if not dimension_type:
            return []
        dimension_type = str(dimension_type)
        if dimension_type in self._dimension_values:
            return self._dimension_values[dimension_type]
        endpoint = (
            f"{self.api_base_url}/Dimension/"
            f"{quote(dimension_type, safe='._-')}/DimensionValues"
        )
        response = _get_who_response(
            endpoint,
            params={
                "$format": "json",
                "$top": WHO_MAX_FILTER_CODES + 1,
            },
            headers={"Accept": "application/json"},
            timeout=60,
        )
        payload = _json_payload(response, f"WHO {dimension_type} values")
        rows = _odata_rows(payload)
        if _odata_next_link(payload) or len(rows) > WHO_MAX_FILTER_CODES:
            values = []
        else:
            values = []
            seen = set()
            for row in rows:
                if not isinstance(row, dict):
                    continue
                code, label = _dimension_code_label(row)
                if code is None or str(code) in seen:
                    continue
                seen.add(str(code))
                values.append({
                    "id": str(code),
                    "label": str(label),
                    "metadata": row,
                })
        self._dimension_values[dimension_type] = values
        return values

    def get_structure(self, dataset_id, metadata=None):
        metadata = metadata or self.get_metadata(dataset_id)
        raw = metadata.get("raw") or {}
        signature = _metadata_signature(raw)
        cached = get_dataset_structure(dataset_id, source=self.source)
        if (
            cached is not None
            and cached.get("metadata_signature") == signature
        ):
            return cached

        dimensions = []
        codelists = []
        indicator_name = _indicator_name(raw, dataset_id)
        indicator_description = _indicator_description(raw)
        dimensions.append({
            "id": "IndicatorCode",
            "name": "Indicator",
            "type": "MeasureDimension",
            "position": 0,
            "codelist_id": None,
            "source_field": None,
            "filterable": False,
            "codes": [{
                "id": dataset_id,
                "label": indicator_name,
            }],
            "description": indicator_description,
            "metadata": {
                "IndicatorCode": dataset_id,
                "IndicatorName": indicator_name,
            },
        })

        for type_field, value_field, category in WHO_DIMENSION_TYPES:
            dimension_type = raw.get(type_field)
            if not dimension_type:
                continue
            dimension_name = _dimension_label(
                dimension_type,
                value_field,
            )
            codes = (
                []
                if category == "time"
                else self._load_dimension_values(dimension_type)
            )
            dimension = {
                "id": value_field,
                "name": dimension_name,
                "type": (
                    "TimeDimension"
                    if category == "time"
                    else "Dimension"
                ),
                "position": len(dimensions) + 1,
                "codelist_id": f"WHO.{dimension_type}",
                "source_field": value_field,
                "filterable": True,
                "codes": [
                    {"id": item["id"], "label": item["label"]}
                    for item in codes
                ],
                "codes_truncated": category != "time" and not codes,
                "description": (
                    f"WHO {dimension_name} dimension "
                    f"({dimension_type})."
                ),
                "metadata": {
                    "type_field": type_field,
                    "dimension_type": dimension_type,
                    "category": category,
                    "code_metadata": {
                        item["id"]: item["metadata"]
                        for item in codes
                    },
                },
            }
            dimensions.append(dimension)
            codelists.append({
                "id": f"WHO.{dimension_type}",
                "name": dimension_name,
                "codes": dimension["codes"],
            })

        unit = next(
            (
                str(raw[key]).strip()
                for key in WHO_UNIT_FIELDS
                if raw.get(key) is not None and str(raw[key]).strip()
            ),
            None,
        )
        if unit:
            dimensions.append({
                "id": "UNIT_MEASURE",
                "name": "Unit of measure",
                "type": "Attribute",
                "position": len(dimensions) + 1,
                "codelist_id": None,
                "source_field": None,
                "filterable": False,
                "codes": [{"id": unit, "label": unit}],
                "description": unit,
                "metadata": {"source_metadata_field": next(
                    key
                    for key in WHO_UNIT_FIELDS
                    if raw.get(key) is not None
                    and str(raw.get(key)).strip()
                )},
            })

        structure = {
            "source": self.source,
            "dataset_id": dataset_id,
            "metadata_signature": signature,
            "dataflow": {
                "id": dataset_id,
                "agency_id": "WHO",
                "version": str(
                    raw.get("LastUpdated")
                    or raw.get("Date")
                    or ""
                ),
                "name": indicator_name,
                "description": indicator_description,
                "source_url": metadata.get("source_url"),
                "api_url": metadata.get("api_url"),
                "identifier": dataset_id,
                "metadata": raw,
            },
            "data_structure": {
                "id": f"WHO.{dataset_id}",
                "agency_id": "WHO",
                "version": str(
                    raw.get("LastUpdated")
                    or raw.get("Date")
                    or ""
                ),
                "dimensions": dimensions,
            },
            "codelists": codelists,
            "unit": unit,
            "who_metadata": raw,
        }
        return structure

    def build_concepts(self, dataset_id, metadata, structure):
        raw = metadata.get("raw") or {}
        indicator_name = metadata.get("title") or dataset_id
        description = metadata.get("description", "")
        dimension_lines = [
            f"Available dimension: {dimension['name']}"
            for dimension in structure["data_structure"]["dimensions"]
            if dimension.get("source_field")
        ]
        unit = structure.get("unit")
        semantic_parts = [
            f"WHO health indicator: {indicator_name}",
        ]
        if description:
            semantic_parts.append(f"Definition: {description}")
        if unit:
            semantic_parts.append(f"Unit: {unit}")
        for field in ("DataSource", "Source", "DataSourceDimType"):
            if raw.get(field):
                semantic_parts.append(
                    f"{field}: {raw[field]}"
                )
        semantic_parts.extend(dimension_lines)

        concepts = [{
            "source": self.source,
            "dataset_id": dataset_id,
            "dimension_id": "INDICATOR",
            "dimension_name": "Indicator",
            "code": dataset_id,
            "label": indicator_name,
            "concept_role": "semantic",
            "semantic_text": "\n".join(semantic_parts),
            "metadata": {
                "source_url": metadata.get("source_url"),
                "api_url": metadata.get("api_url"),
                "indicator_metadata": raw,
                "unit": unit,
                "definition": description,
            },
        }]

        for dimension in structure["data_structure"]["dimensions"]:
            source_field = dimension.get("source_field")
            if not source_field:
                continue
            for code in dimension.get("codes", []):
                concepts.append({
                    "source": self.source,
                    "dataset_id": dataset_id,
                    "dimension_id": source_field,
                    "dimension_name": dimension["name"],
                    "code": code["id"],
                    "label": code["label"],
                    "concept_role": "filter",
                    "semantic_text": None,
                    "metadata": {
                        "dimension_type": dimension[
                            "metadata"
                        ]["dimension_type"],
                        **dimension["metadata"].get(
                            "code_metadata", {}
                        ).get(code["id"], {}),
                    },
                })
        return concepts

    def get_indexed_structure(self, dataset_id):
        structure = self.get_cached_structure(dataset_id)
        if structure is None:
            raise LookupError(
                f"WHO indicator {dataset_id!r} is not indexed locally; "
                "index the indicator before analysis"
            )
        return structure

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
        if not 1 <= limit <= WHO_MAX_PAGE_SIZE:
            raise ValueError(
                f"limit must be between 1 and {WHO_MAX_PAGE_SIZE}"
            )
        if not isinstance(filters or {}, dict):
            raise TypeError("filters must map WHO field IDs to exact values")
        if start_period is not None and not re.fullmatch(
            r"\d{4}", str(start_period)
        ):
            raise ValueError("start_period must be a four-digit year")
        if end_period is not None and not re.fullmatch(
            r"\d{4}", str(end_period)
        ):
            raise ValueError("end_period must be a four-digit year")
        if (
            start_period is not None
            and end_period is not None
            and int(start_period) > int(end_period)
        ):
            raise ValueError("start_period must not be after end_period")

        structure = self.get_structure(dataset_id)
        filter_dimensions = {
            dimension["id"]: dimension
            for dimension in structure["data_structure"]["dimensions"]
            if dimension.get("filterable")
            and dimension.get("source_field")
        }
        requested_filters = filters or {}
        unknown = set(requested_filters) - set(filter_dimensions)
        if unknown:
            raise ValueError(
                "Unknown or non-filterable WHO field(s): "
                + ", ".join(sorted(unknown))
            )

        clauses = []
        for field, value in requested_filters.items():
            if isinstance(value, (dict, list, tuple)) or value is None:
                raise TypeError(
                    f"WHO filter {field!r} must be a scalar code or value"
                )
            dimension = filter_dimensions[field]
            known_codes = {
                code["id"]
                for code in dimension.get("codes", [])
            }
            if known_codes and str(value) not in known_codes:
                raise ValueError(
                    f"WHO filter value {value!r} is not an indexed code "
                    f"for {field!r}"
                )
            source_field = dimension["source_field"]
            literal = str(value).replace("'", "''")
            clauses.append(f"{source_field} eq '{literal}'")

        if start_period is not None:
            clauses.append(
                "date(TimeDimensionBegin) ge "
                f"{start_period}-01-01"
            )
        if end_period is not None:
            next_year = int(end_period) + 1
            clauses.append(
                "date(TimeDimensionBegin) lt "
                f"{next_year}-01-01"
            )

        params = {
            "$format": "json",
            "$top": limit,
        }
        if clauses:
            params["$filter"] = " and ".join(clauses)
        metadata = self.get_metadata(dataset_id)
        api_url = metadata.get("api_url") or (
            f"{self.api_base_url}/{quote(dataset_id, safe='._-')}"
        )
        response = _get_who_response(
            api_url,
            params=params,
            headers={"Accept": "application/json"},
            timeout=60,
        )
        payload = _json_payload(
            response,
            f"WHO observations for {dataset_id!r}",
        )
        rows = _odata_rows(payload)
        if any(not isinstance(row, dict) for row in rows):
            raise ValueError(
                f"WHO observations for {dataset_id!r} contain invalid records"
            )
        if len(rows) > limit:
            raise ValueError(
                f"WHO API returned {len(rows)} rows for limit {limit}"
            )
        return {
            "source": self.source,
            "dataset_id": dataset_id,
            "request": {
                "url": response.url,
                "params": params,
                "filters": dict(requested_filters),
                "start_period": start_period,
                "end_period": end_period,
                "limit": limit,
                "truncated": bool(_odata_next_link(payload)),
            },
            "rows": rows,
        }

    def normalize_dataframe(self, result, structure):
        if not isinstance(result, dict):
            raise TypeError("WHO result must be a mapping")
        if result.get("source") != self.source:
            raise ValueError("WHO result has an unexpected source")
        rows = result.get("rows")
        if not isinstance(rows, list) or any(
            not isinstance(row, dict) for row in rows
        ):
            raise ValueError("WHO result rows must be a list of objects")

        dataframe = pd.DataFrame.from_records(rows)
        dataframe["dataset_id"] = result["dataset_id"]
        dataframe["source"] = self.source
        if "IndicatorCode" not in dataframe:
            dataframe["IndicatorCode"] = result["dataset_id"]
        if "IndicatorName" not in dataframe:
            dataframe["IndicatorName"] = structure["dataflow"]["name"]
        if "INDICATOR" not in dataframe:
            dataframe["INDICATOR"] = dataframe["IndicatorCode"]
        if "INDICATOR_LABEL" not in dataframe:
            dataframe["INDICATOR_LABEL"] = dataframe["IndicatorName"]
        if "OBS_VALUE" not in dataframe:
            if "NumericValue" in dataframe:
                dataframe["OBS_VALUE"] = dataframe["NumericValue"]
                if "Value" in dataframe:
                    dataframe["OBS_VALUE"] = dataframe[
                        "OBS_VALUE"
                    ].where(
                        dataframe["OBS_VALUE"].notna(),
                        dataframe["Value"],
                    )
            elif "Value" in dataframe:
                dataframe["OBS_VALUE"] = dataframe["Value"]
            else:
                dataframe["OBS_VALUE"] = pd.NA
        if "TIME_PERIOD" not in dataframe:
            if "TimeDim" in dataframe:
                dataframe["TIME_PERIOD"] = dataframe["TimeDim"]
                if "TimeDimensionBegin" in dataframe:
                    dataframe["TIME_PERIOD"] = dataframe[
                        "TIME_PERIOD"
                    ].where(
                        dataframe["TIME_PERIOD"].notna(),
                        dataframe["TimeDimensionBegin"].map(
                            lambda value: (
                                str(value)[:4]
                                if pd.notna(value)
                                else value
                            )
                        ),
                    )
            elif "TimeDimensionBegin" in dataframe:
                dataframe["TIME_PERIOD"] = dataframe[
                    "TimeDimensionBegin"
                ].map(
                    lambda value: (
                        str(value)[:4] if pd.notna(value) else value
                    )
                )
            else:
                dataframe["TIME_PERIOD"] = pd.NA

        for dimension in structure["data_structure"]["dimensions"]:
            source_field = dimension.get("source_field")
            if not source_field or source_field not in dataframe:
                continue
            label_column = f"{source_field}_LABEL"
            if label_column in dataframe:
                continue
            labels = {
                code["id"]: code["label"]
                for code in dimension.get("codes", [])
            }
            value_label_column = f"{source_field}Value"
            if value_label_column in dataframe:
                dataframe[label_column] = dataframe[value_label_column]
            elif labels:
                dataframe[label_column] = dataframe[source_field].map(labels)
        dataframe.attrs["analysis_dimension_columns"] = [
            dimension["source_field"]
            for dimension in structure["data_structure"]["dimensions"]
            if dimension.get("filterable")
            and dimension.get("source_field")
            and dimension.get("type") != "TimeDimension"
        ]
        return dataframe

    def get_provenance(self, dataset_id, structure, result):
        return {
            "source": self.source,
            "dataset_id": dataset_id,
            "dataset_title": structure["dataflow"]["name"],
            "source_url": structure["dataflow"].get("source_url"),
            "indicator_code": dataset_id,
            "indicator_metadata": dict(
                structure.get("who_metadata", {})
            ),
            "data_structure_id": structure["data_structure"]["id"],
            "dimensions": [
                {
                    "id": item["id"],
                    "name": item["name"],
                    "source_field": item.get("source_field"),
                    "type": item["type"],
                    "codes_available": len(item.get("codes", [])),
                }
                for item in structure["data_structure"]["dimensions"]
            ],
            "request": dict(result["request"]),
        }

    def index_datasets(
        self,
        limit=WHO_MAX_INDEX_DATASETS,
        embedding_model="nomic-local",
    ):
        if limit is None:
            limit = WHO_MAX_INDEX_DATASETS
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("WHO indexing requires a positive dataset limit")
        return super().index_datasets(
            limit=min(limit, WHO_MAX_INDEX_DATASETS),
            embedding_model=embedding_model,
        )
