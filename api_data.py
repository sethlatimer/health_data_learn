import csv
import io
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote

import requests
import pandas as pd

import embedding
from database import (
    get_dataset_concept_embeddings,
    get_dataset_structure as load_dataset_structure,
    save_dataset_concepts,
    save_dataset_structure,
    save_dataset_semantics,
)


OECD_BASE_URL = "https://sdmx.oecd.org/public/rest"
OECD_CA_BUNDLE = Path(__file__).resolve().parent / "Certs" / "ZscalerRootCA.pem"
SDMX_ACCEPT = "application/vnd.sdmx.structure+xml;version=2.1"
OECD_INDEX_REQUEST_DELAY = 1.1
OECD_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
OECD_MAX_REQUEST_ATTEMPTS = 4
HEALTHCARE_TERMS = (
    "health",
    "healthcare",
    "hospital",
    "medical",
    "physician",
    "doctor",
    "nursing",
    "mortality",
    "sdoh"
)


def _local_name(tag):
    return tag.rsplit("}", 1)[-1]


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


def _get_oecd_response(url, params=None, headers=None, timeout=60):
    for attempt in range(OECD_MAX_REQUEST_ATTEMPTS):
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=timeout,
            verify=str(OECD_CA_BUNDLE),
        )
        if response.status_code not in OECD_RETRYABLE_STATUSES:
            response.raise_for_status()
            return response
        if attempt == OECD_MAX_REQUEST_ATTEMPTS - 1:
            response.raise_for_status()
        time.sleep(_retry_delay(response, attempt))

    raise RuntimeError("OECD request retries exhausted")


def _get_xml(url, params=None):
    response = _get_oecd_response(
        url,
        params=params,
        headers={"Accept": SDMX_ACCEPT},
    )
    return ET.fromstring(response.content)


def _text_for(element, tag_name):
    candidates = [
        child
        for child in element.iter()
        if _local_name(child.tag) == tag_name
        and (child.text or "").strip()
    ]

    if not candidates:
        return ""

    english = [
        candidate
        for candidate in candidates
        if candidate.attrib.get(
            "{http://www.w3.org/XML/1998/namespace}lang", ""
        ).lower().startswith("en")
    ]
    return (english or candidates)[0].text.strip()


def _dataflow_id(dataflow):
    return ",".join((
        dataflow.get("agencyID", "all"),
        dataflow.get("id", ""),
        dataflow.get("version", "latest"),
    ))


def _parse_dataflows(root):
    dataflows = []

    for element in root.iter():
        if _local_name(element.tag) != "Dataflow":
            continue

        dataflow_id = element.get("id")
        if not dataflow_id:
            continue

        dataflows.append({
            "dataset_id": _dataflow_id(element),
            "id": dataflow_id,
            "agency_id": element.get("agencyID", "all"),
            "version": element.get("version", "latest"),
            "title": _text_for(element, "Name"),
            "description": _text_for(element, "Description"),
        })

    return dataflows


def discover_dataflows(base_url=OECD_BASE_URL):
    root = _get_xml(
        f"{base_url.rstrip('/')}/v1/dataflow/all/all/latest",
        params={"references": "none"},
    )
    return _parse_dataflows(root)


def print_dataflows(dataflows):
    for dataflow in dataflows:
        print(f"{dataflow['dataset_id']}: {dataflow['title']}")


def find_healthcare_datasets(
    limit=10,
    base_url=OECD_BASE_URL,
):
    if limit is not None and limit < 1:
        raise ValueError("limit must be greater than zero")

    healthcare_dataflows = []
    for dataflow in discover_dataflows(base_url=base_url):
        searchable = (
            dataflow["id"] + " " + dataflow["title"] + " "
            + dataflow["description"]
        ).lower()
        if any(term in searchable for term in HEALTHCARE_TERMS):
            healthcare_dataflows.append(dataflow)

    if limit is not None:
        healthcare_dataflows = healthcare_dataflows[:limit]
    print_dataflows(healthcare_dataflows)
    return healthcare_dataflows


def _parse_dataset_id(dataset_id):
    if isinstance(dataset_id, dict):
        return (
            dataset_id["agency_id"],
            dataset_id["id"],
            dataset_id["version"],
        )

    parts = dataset_id.split(",", 2)
    if len(parts) == 3:
        return tuple(parts)
    if len(parts) == 2:
        return parts[0], parts[1], "latest"
    if not parts[0]:
        raise ValueError("dataset_id must not be empty")
    return "all", parts[0], "latest"


def _element_attributes(element):
    return {
        key.rsplit("}", 1)[-1]: value
        for key, value in element.attrib.items()
    }


def _dataset_structure_root(dataset_id, base_url):
    agency_id, flow_id, version = _parse_dataset_id(dataset_id)
    path = "/".join((
        "v1",
        "dataflow",
        quote(agency_id, safe=""),
        quote(flow_id, safe=""),
        quote(version, safe=""),
    ))
    return _get_xml(
        f"{base_url.rstrip('/')}/{path}",
        params={"references": "all"},
    )


def _structure_for_dataflow(root, agency_id, flow_id, version):
    dataflow = next(
        (
            element
            for element in root.iter()
            if _local_name(element.tag) == "Dataflow"
            and element.get("id") == flow_id
            and (
                agency_id == "all"
                or element.get("agencyID", "all") == agency_id
            )
            and (
                version == "latest"
                or element.get("version", "latest") == version
            )
        ),
        None,
    )
    if dataflow is None:
        raise ValueError(f"Dataflow not found in OECD structure response: {flow_id}")

    structure_ref = next(
        (
            element
            for element in dataflow.iter()
            if _local_name(element.tag) == "Ref"
            and element.get("package") == "datastructure"
        ),
        None,
    )
    structures = [
        element
        for element in root.iter()
        if _local_name(element.tag) == "DataStructure"
    ]

    if structure_ref is not None:
        structure_id = structure_ref.get("id")
        structure_agency = structure_ref.get("agencyID")
        structure_version = structure_ref.get("version")
        structure = next(
            (
                element
                for element in structures
                if element.get("id") == structure_id
                and (
                    not structure_agency
                    or element.get("agencyID") == structure_agency
                )
                and (
                    not structure_version
                    or element.get("version") == structure_version
                )
            ),
            None,
        )
    else:
        structure = structures[0] if len(structures) == 1 else None

    if structure is None:
        raise ValueError(
            f"OECD structure definition not found for dataflow: {flow_id}"
        )

    return dataflow, structure


def _build_codelists(root):
    parent_map = {
        child: parent
        for parent in root.iter()
        for child in list(parent)
    }
    codelists = {}

    for element in root.iter():
        if _local_name(element.tag) != "Codelist":
            continue

        codes = []
        for code in element.iter():
            if _local_name(code.tag) != "Code":
                continue
            parent = parent_map.get(code)
            parent_id = (
                parent.get("id")
                if parent is not None
                and _local_name(parent.tag) == "Code"
                else None
            )
            codes.append({
                "id": code.get("id", ""),
                "label": _text_for(code, "Name"),
                "parent_id": parent_id,
            })

        codelist = {
            "id": element.get("id", ""),
            "agency_id": element.get("agencyID", ""),
            "version": element.get("version", ""),
            "name": _text_for(element, "Name"),
            "codes": codes,
        }
        key = (
            codelist["agency_id"],
            codelist["id"],
            codelist["version"],
        )
        codelists[key] = codelist

    return codelists


def _build_concepts(root):
    parent_map = {
        child: parent
        for parent in root.iter()
        for child in list(parent)
    }
    concepts = {}

    for concept in root.iter():
        if _local_name(concept.tag) != "Concept":
            continue

        scheme = parent_map.get(concept)
        if scheme is None or _local_name(scheme.tag) != "ConceptScheme":
            continue

        key = (
            scheme.get("agencyID", ""),
            concept.get("id", ""),
            scheme.get("id", ""),
            scheme.get("version", ""),
        )
        concepts[key] = _text_for(concept, "Name")

    return concepts


def _build_dimensions(structure, codelists, concepts):
    dimension_list = next(
        (
            element
            for element in structure.iter()
            if _local_name(element.tag) == "DimensionList"
        ),
        None,
    )
    if dimension_list is None:
        return []

    dimensions = []
    for element in list(dimension_list):
        element_type = _local_name(element.tag)
        if element_type not in ("Dimension", "TimeDimension", "MeasureDimension"):
            continue

        concept_ref = next(
            (
                child
                for child in element.iter()
                if _local_name(child.tag) == "Ref"
                and child.get("package") == "conceptscheme"
            ),
            None,
        )
        concept_name = ""
        if concept_ref is not None:
            concept_name = concepts.get(
                (
                    concept_ref.get("agencyID", ""),
                    concept_ref.get("id", ""),
                    concept_ref.get("maintainableParentID", ""),
                    concept_ref.get("maintainableParentVersion", ""),
                ),
                "",
            )

        codelist_ref = next(
            (
                child
                for child in element.iter()
                if _local_name(child.tag) == "Ref"
                and child.get("package") == "codelist"
            ),
            None,
        )
        codelist = None
        if codelist_ref is not None:
            codelist = codelists.get((
                codelist_ref.get("agencyID", ""),
                codelist_ref.get("id", ""),
                codelist_ref.get("version", ""),
            ))

        dimensions.append({
            "id": element.get("id", ""),
            "name": concept_name or element.get("id", ""),
            "type": element_type,
            "position": int(element.get("position", "0")),
            "codelist_id": codelist["id"] if codelist else None,
            "codes": codelist["codes"] if codelist else [],
        })

    return sorted(dimensions, key=lambda dimension: dimension["position"])


def get_dataset_structure(dataset_id, base_url=OECD_BASE_URL):
    agency_id, flow_id, version = _parse_dataset_id(dataset_id)
    root = _dataset_structure_root(dataset_id, base_url)
    dataflow, structure = _structure_for_dataflow(
        root,
        agency_id,
        flow_id,
        version,
    )
    codelists = _build_codelists(root)
    concepts = _build_concepts(root)
    dimensions = _build_dimensions(structure, codelists, concepts)

    return {
        "dataset_id": _dataflow_id(dataflow),
        "dataflow": {
            "id": dataflow.get("id", ""),
            "agency_id": dataflow.get("agencyID", ""),
            "version": dataflow.get("version", ""),
            "name": _text_for(dataflow, "Name"),
            "description": _text_for(dataflow, "Description"),
        },
        "data_structure": {
            "id": structure.get("id", ""),
            "agency_id": structure.get("agencyID", ""),
            "version": structure.get("version", ""),
            "dimensions": dimensions,
        },
        "codelists": list(codelists.values()),
    }


def get_indexed_dataset_structure(dataset_id):
    structure = load_dataset_structure(dataset_id)
    if structure is None:
        raise LookupError(
            f"Dataset structure for {dataset_id!r} is not indexed locally; "
            "index the dataset before analyzing it"
        )
    return structure


def get_dataset_dimensions(dataset_id, base_url=OECD_BASE_URL):
    return get_dataset_structure(dataset_id, base_url)["data_structure"]["dimensions"]


def get_dataset_codelists(dataset_id, base_url=OECD_BASE_URL):
    return get_dataset_structure(dataset_id, base_url)["codelists"]


def get_observations(
    dataset_id,
    filters,
    start_period=None,
    end_period=None,
    base_url=OECD_BASE_URL,
):
    if not isinstance(filters, dict):
        raise TypeError("filters must map dimension IDs to code IDs")

    structure = get_indexed_dataset_structure(dataset_id)
    dimensions = structure["data_structure"]["dimensions"]
    dimensions_by_id = {dimension["id"]: dimension for dimension in dimensions}
    unknown_dimensions = set(filters) - set(dimensions_by_id)
    if unknown_dimensions:
        raise ValueError(
            "Unknown dataset dimension(s): "
            + ", ".join(sorted(unknown_dimensions))
        )

    for dimension_id, code_id in filters.items():
        dimension = dimensions_by_id[dimension_id]
        valid_codes = {code["id"] for code in dimension["codes"]}
        if code_id not in valid_codes:
            raise ValueError(
                f"Invalid code {code_id!r} for dimension {dimension_id!r}"
            )

    data_dimensions = [
        dimension
        for dimension in dimensions
        if dimension["type"] != "TimeDimension"
    ]
    key = ".".join(
        quote(filters.get(dimension["id"], ""), safe="")
        for dimension in data_dimensions
    )
    flow_reference = ",".join((
        structure["dataflow"]["agency_id"],
        structure["dataflow"]["id"],
        structure["dataflow"]["version"],
    ))
    url = (
        f"{base_url.rstrip('/')}/v1/data/"
        f"{flow_reference}/{key}"
    )
    params = {}
    if start_period is not None:
        params["startPeriod"] = start_period
    if end_period is not None:
        params["endPeriod"] = end_period

    response = _get_oecd_response(
        url,
        params=params,
        headers={"Accept": "text/csv;version=2"},
    )

    reader = csv.DictReader(io.StringIO(response.content.decode("utf-8-sig")))
    if not reader.fieldnames or "OBS_VALUE" not in reader.fieldnames:
        raise ValueError("OECD response CSV did not include an OBS_VALUE column")

    dimension_ids = [dimension["id"] for dimension in dimensions]
    attribute_ids = set(dimension_ids) | {"DATAFLOW", "OBS_VALUE"}
    observations = []
    for row in reader:
        observations.append({
            "dimensions": {
                dimension_id: row.get(dimension_id, "")
                for dimension_id in dimension_ids
                if dimension_id in row
            },
            "observation_value": row["OBS_VALUE"],
            "attributes": {
                name: value
                for name, value in row.items()
                if name not in attribute_ids
            },
        })

    return {
        "dataset_id": structure["dataset_id"],
        "request": {
            "url": response.url,
            "key": key,
            "parameters": params,
        },
        "observations": observations,
    }


def observations_to_dataframe(result, structure):
    if not isinstance(result, dict):
        raise TypeError("result must be the mapping returned by get_observations()")
    if not isinstance(structure, dict):
        raise TypeError("structure must be the mapping returned by get_dataset_structure()")

    try:
        dataset_id = result["dataset_id"]
        dimensions = structure["data_structure"]["dimensions"]
        observations = result["observations"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "result and structure must contain dataset_id, observations, "
            "and data_structure.dimensions"
        ) from error

    if not isinstance(dimensions, list):
        raise ValueError("structure data_structure.dimensions must be a list")
    if not isinstance(observations, list):
        raise ValueError("result observations must be a list")

    dimension_ids = [dimension.get("id") for dimension in dimensions]
    if any(not isinstance(dimension_id, str) or not dimension_id for dimension_id in dimension_ids):
        raise ValueError("Each OECD dimension must have a non-empty string ID")
    if len(set(dimension_ids)) != len(dimension_ids):
        raise ValueError("OECD structure contains duplicate dimension IDs")

    labels_by_dimension = {
        dimension["id"]: {
            item["id"]: item["label"]
            for item in dimension.get("codes", [])
        }
        for dimension in dimensions
    }

    columns = {"dataset_id"}
    for dimension in dimensions:
        dimension_id = dimension["id"]
        label_column = (
            f"{dimension_id}_LABEL"
            if dimension.get("codes")
            else None
        )
        if dimension_id in columns or (
            label_column is not None and label_column in columns
        ):
            raise ValueError(
                f"Dimension columns collide with an existing column: {dimension_id}"
            )
        columns.add(dimension_id)
        if label_column is not None:
            columns.add(label_column)

    rows = []
    numeric_values = []
    for index, observation in enumerate(observations):
        if not isinstance(observation, dict):
            raise ValueError(f"Observation {index} must be a mapping")

        observation_dimensions = observation.get("dimensions")
        if not isinstance(observation_dimensions, dict):
            raise ValueError(
                f"Observation {index} dimensions must be a mapping"
            )
        observed_dimension_ids = set(observation_dimensions)
        expected_dimension_ids = set(dimension_ids)
        if observed_dimension_ids != expected_dimension_ids:
            missing = sorted(expected_dimension_ids - observed_dimension_ids)
            extra = sorted(observed_dimension_ids - expected_dimension_ids)
            raise ValueError(
                f"Observation {index} has a dimension-count/ID mismatch "
                f"(expected {len(dimension_ids)}, got "
                f"{len(observation_dimensions)}; missing={missing}, extra={extra})"
            )

        if "observation_value" not in observation:
            raise ValueError(
                f"Observation {index} is missing observation_value"
            )
        attributes = observation.get("attributes", {})
        if not isinstance(attributes, dict):
            raise ValueError(
                f"Observation {index} attributes must be a mapping"
            )

        row = {"dataset_id": dataset_id}
        for dimension in dimensions:
            dimension_id = dimension["id"]
            code = observation_dimensions[dimension_id]
            row[dimension_id] = code
            if dimension.get("codes"):
                row[f"{dimension_id}_LABEL"] = labels_by_dimension[
                    dimension_id
                ].get(code, pd.NA)

        for name, value in attributes.items():
            if name in row:
                raise ValueError(
                    f"Observation {index} attribute collides with column {name!r}"
                )
            row[name] = value

        raw_value = observation["observation_value"]
        if raw_value is None or raw_value == "":
            numeric_values.append(pd.NA)
        else:
            try:
                numeric_values.append(pd.to_numeric(raw_value, errors="raise"))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Observation {index} has non-numeric observation_value "
                    f"{raw_value!r}"
                ) from error
        rows.append(row)

    dataframe = pd.DataFrame(rows)
    if not rows:
        dataframe = pd.DataFrame(columns=["dataset_id"])
        for dimension in dimensions:
            dataframe[dimension["id"]] = pd.Series(dtype="object")
            if dimension.get("codes"):
                dataframe[f"{dimension['id']}_LABEL"] = pd.Series(dtype="object")

    dataframe["OBS_VALUE"] = pd.array(numeric_values, dtype="Float64")
    return dataframe


def _concept_role(dimension):
    dimension_id = dimension.get("id", "").upper()
    dimension_type = dimension.get("type", "").lower()
    dimension_name = " ".join(
        dimension.get("name", "").lower().split()
    )
    codelist_id = (dimension.get("codelist_id") or "").upper()

    if dimension_type == "measuredimension":
        return "semantic"

    if dimension_id in {"MEASURE", "INDICATOR"}:
        return "semantic"

    if (
        dimension_name in {"measure", "indicator"}
        or codelist_id.startswith(("CL_MEASURE", "CL_INDICATOR"))
    ):
        return "semantic"

    return "filter"


def build_dataset_concepts(structure):
    dataset_id = structure["dataset_id"]
    dimensions = structure["data_structure"]["dimensions"]
    concepts = []

    for dimension in dimensions:
        concept_role = _concept_role(dimension)
        for code in dimension.get("codes", []):
            concepts.append({
                "dataset_id": dataset_id,
                "dimension_id": dimension["id"],
                "dimension_name": dimension["name"],
                "code": code["id"],
                "label": code["label"],
                "concept_role": concept_role,
            })

    return concepts


def _index_dataset_structure(structure, embedding_model):
    save_dataset_structure(structure)
    concepts = build_dataset_concepts(structure)
    dataset_title = structure["dataflow"]["name"]
    existing_embeddings = get_dataset_concept_embeddings(
        structure["dataset_id"],
        embedding_model,
    )
    pending = []

    for concept in concepts:
        if concept["concept_role"] != "semantic":
            continue

        embedding_text = (
            f"OECD dataset: {dataset_title}\n"
            f"Dimension: {concept['dimension_name']}\n"
            f"Code: {concept['code']}\n"
            f"Label: {concept['label']}"
        )
        concept["semantic_text"] = embedding_text
        existing = existing_embeddings.get((
            concept["dimension_id"],
            concept["code"],
        ))
        if existing is not None and existing[0] == embedding_text:
            concept["embedding"] = existing[1]
        else:
            pending.append((concept, embedding_text))
        concept["embedding_model"] = embedding_model

    pending_embeddings = embedding.embed_texts(
        (text for _, text in pending),
        embedding_model,
    )
    for (concept, _), vector in zip(pending, pending_embeddings):
        concept["embedding"] = vector

    save_dataset_concepts(concepts)

    return {
        "dataset_id": structure["dataset_id"],
        "title": dataset_title,
        "total": len(concepts),
        "semantic": sum(
            concept["concept_role"] == "semantic"
            for concept in concepts
        ),
        "filter": sum(
            concept["concept_role"] == "filter"
            for concept in concepts
        ),
        "embedded": sum(concept.get("embedding") is not None for concept in concepts),
        "embeddings_created": len(pending_embeddings),
        "embedding_model": embedding_model,
    }


def index_dataset_concepts(dataset_id, embedding_model="nomic-local"):
    structure = get_dataset_structure(dataset_id)
    return _index_dataset_structure(structure, embedding_model)


def index_healthcare_datasets(
    limit=None,
    embedding_model="nomic-local",
    base_url=OECD_BASE_URL,
):
    from data_sources.oecd import OECDDataSource

    return OECDDataSource(base_url=base_url).index_healthcare_datasets(
        limit=limit,
        embedding_model=embedding_model,
    )


def _metadata_items(root):
    items = []
    seen = set()

    for element in root.iter():
        tag = _local_name(element.tag)
        if tag not in ("Name", "Description"):
            continue

        value = (element.text or "").strip()
        if not value:
            continue

        language = element.attrib.get(
            "{http://www.w3.org/XML/1998/namespace}lang", ""
        )
        if language and not language.lower().startswith("en"):
            continue

        parent_id = ""
        parent_tag = ""
        for parent in root.iter():
            if element in list(parent):
                parent_tag = _local_name(parent.tag)
                parent_id = parent.get("id", "")
                break

        item = {
            "kind": tag.lower(),
            "parent": parent_tag,
            "id": parent_id,
            "text": value,
        }
        key = (item["kind"], item["parent"], item["id"], value)
        if key not in seen:
            seen.add(key)
            items.append(item)

    return items


def get_dataset_metadata(
    dataset_id,
    embedding_model=None,
    base_url=OECD_BASE_URL,
):
    agency_id, flow_id, version = _parse_dataset_id(dataset_id)
    path = "/".join((
        "v1",
        "dataflow",
        quote(agency_id, safe=""),
        quote(flow_id, safe=""),
        quote(version, safe=""),
    ))
    root = _get_xml(
        f"{base_url.rstrip('/')}/{path}",
        params={"references": "none"},
    )

    flow = next(
        (
            item
            for item in _parse_dataflows(root)
            if item["id"] == flow_id
        ),
        {
            "dataset_id": ",".join((agency_id, flow_id, version)),
            "id": flow_id,
            "agency_id": agency_id,
            "version": version,
            "title": _text_for(root, "Name"),
            "description": _text_for(root, "Description"),
        },
    )
    metadata = {
        **flow,
        "metadata_items": _metadata_items(root),
    }
    metadata["metadata_text"] = build_metadata_text(metadata)

    if embedding_model is not None:
        metadata["embedding"] = embedding.generate_embedding(
            "search_document: " + metadata["metadata_text"],
            embedding_model,
        )
        metadata["embedding_model"] = embedding_model

    return metadata


def index_dataset_semantics(dataset_id, embedding_model="nomic-local"):
    metadata = get_dataset_metadata(dataset_id)
    vector = embedding.generate_embedding(
        "search_document: " + metadata["metadata_text"],
        embedding_model,
    )
    save_dataset_semantics(
        metadata["dataset_id"],
        metadata["title"] or metadata["id"],
        metadata["metadata_text"],
        vector,
        embedding_model,
    )
    return {
        "dataset_id": metadata["dataset_id"],
        "title": metadata["title"],
        "embedding_model": embedding_model,
        "dimensions": len(vector),
    }


def build_metadata_text(metadata):
    lines = [
        f"OECD dataset: {metadata.get('title') or metadata.get('id', '')}",
        f"Dataset ID: {metadata.get('dataset_id', '')}",
    ]

    description = metadata.get("description")
    if description:
        lines.append(f"Description: {description}")

    for item in metadata.get("metadata_items", []):
        label = " ".join(
            part for part in (item.get("parent"), item.get("id")) if part
        )
        prefix = f"{label} " if label else ""
        lines.append(f"{item.get('kind', 'metadata').title()}: {prefix}{item['text']}")

    return "\n".join(lines)


if __name__ == "__main__":
    test_dataset_id = "OECD.ELS.HD,DSD_HCQO@DF_PC,2.1"
    test_filters = {
        "REF_AREA": "AUS",
        "FREQ": "A",
        "MEASURE": "ADMRASTH",
        "UNIT_MEASURE": "10P5HB",
        "AGE": "Y_GE15",
        "SEX": "_T",
        "STATISTICAL_OPERATION": "OBS",
        "INCOME_GROUP": "_Z",
        "HEALTH_PROF": "_Z",
    }
    result = get_observations(
        test_dataset_id,
        test_filters,
        start_period="2016",
        end_period="2023",
    )
    structure = get_dataset_structure(test_dataset_id)
    dataframe = observations_to_dataframe(result, structure)
    print("Request parameters:")
    print(f"  URL: {result['request']['url']}")
    print(f"  Key: {result['request']['key']}")
    print(f"  Periods: {result['request']['parameters']}")
    print("\nObservations:")
    print(dataframe.to_string(index=False))
    print("\nColumn names:")
    print(dataframe.columns.tolist())
    print("\nDtypes:")
    print(dataframe.dtypes.to_string())
    print("\nMissing values:")
    print(dataframe.isna().sum().to_string())
