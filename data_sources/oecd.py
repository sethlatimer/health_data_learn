import api_data
import embedding
from data_sources.base import DataSource
from database import get_dataset_record
import os


class OECDDataSource(DataSource):
    source = "OECD"

    def __init__(self, base_url=api_data.OECD_BASE_URL):
        self.base_url = base_url
        self._discovered = {}

    def discover_datasets(self, limit=None):
        datasets = api_data.find_healthcare_datasets(
            limit=limit,
            base_url=self.base_url,
        )
        self._discovered = {
            dataset["dataset_id"]: dataset
            for dataset in datasets
            if dataset.get("dataset_id")
        }
        return [
            {
                **dataset,
                "source": self.source,
            }
            for dataset in datasets
        ]

    def get_metadata(self, dataset_id):
        discovered = self._discovered.get(dataset_id)
        if discovered is not None:
            return {
                "source": self.source,
                "dataset_id": dataset_id,
                "title": discovered.get("title") or dataset_id,
                "description": discovered.get("description", ""),
                "source_url": discovered.get("source_url"),
                "raw": dict(discovered),
            }
        cached = get_dataset_record(self.source, dataset_id)
        if cached is not None:
            return cached
        metadata = api_data.get_dataset_metadata(
            dataset_id,
            base_url=self.base_url,
        )
        return {
            **metadata,
            "source": self.source,
            "source_url": None,
            "raw": metadata,
        }

    def get_structure(self, dataset_id, metadata=None):
        return api_data.get_dataset_structure(
            dataset_id,
            base_url=self.base_url,
        )

    def get_indexed_structure(self, dataset_id):
        structure = self.get_cached_structure(dataset_id)
        if structure is None:
            raise LookupError(
                f"Dataset structure for {dataset_id!r} is not indexed locally; "
                "index the dataset before analyzing it"
            )
        return structure

    def build_concepts(self, dataset_id, metadata, structure):
        return api_data.build_dataset_concepts(structure)

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
        if limit is not None and limit < 1:
            raise ValueError("limit must be greater than zero")
        return api_data.get_observations(
            dataset_id,
            filters or {},
            start_period=start_period,
            end_period=end_period,
            base_url=self.base_url,
        )

    def normalize_dataframe(self, result, structure):
        return api_data.observations_to_dataframe(result, structure)

    def get_provenance(self, dataset_id, structure, result):
        return {
            "dataset_id": structure["dataset_id"],
            "dataset_title": structure["dataflow"]["name"],
            "dataflow": dict(structure["dataflow"]),
            "data_structure": {
                "id": structure["data_structure"]["id"],
                "agency_id": structure["data_structure"]["agency_id"],
                "version": structure["data_structure"]["version"],
                "dimensions": [
                    {
                        "id": dimension["id"],
                        "name": dimension["name"],
                        "type": dimension["type"],
                        "position": dimension["position"],
                        "codelist_id": dimension["codelist_id"],
                    }
                    for dimension in structure["data_structure"]["dimensions"]
                ],
            },
            "request": dict(result["request"]),
        }

    def index_healthcare_datasets(
        self,
        limit=None,
        embedding_model="nomic-local",
    ):
        return self.index_datasets(
            limit=limit,
            embedding_model=embedding_model,
        )
