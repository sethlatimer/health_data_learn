from abc import ABC, abstractmethod

import pandas as pd

import embedding
from data_sources.compatibility import check_dataset_compatibility
from database import (
    get_dataset_structure,
    get_dataset_concept_embeddings,
    save_dataset_concepts,
    save_dataset_record,
    save_dataset_structure,
)


class DataSource(ABC):
    source = None

    @abstractmethod
    def discover_datasets(self, limit=None):
        raise NotImplementedError

    @abstractmethod
    def get_metadata(self, dataset_id):
        raise NotImplementedError

    @abstractmethod
    def get_structure(self, dataset_id, metadata=None):
        raise NotImplementedError

    @abstractmethod
    def build_concepts(self, dataset_id, metadata, structure):
        raise NotImplementedError

    @abstractmethod
    def find_filters(self, dataset_id, dimension_id=None, search=None):
        raise NotImplementedError

    @abstractmethod
    def retrieve_data(
        self,
        dataset_id,
        filters=None,
        start_period=None,
        end_period=None,
        limit=1000,
    ):
        raise NotImplementedError

    @abstractmethod
    def normalize_dataframe(self, result, structure):
        raise NotImplementedError

    def get_provenance(self, dataset_id, structure, result):
        dataflow = structure.get("dataflow", {})
        data_structure = structure.get("data_structure", {})
        return {
            "source": self.source,
            "dataset_id": dataset_id,
            "dataset_title": dataflow.get("name"),
            "source_url": dataflow.get("source_url"),
            "data_structure": {
                "id": data_structure.get("id"),
                "agency_id": data_structure.get("agency_id"),
                "version": data_structure.get("version"),
                "dimensions": [
                    {
                        "id": dimension.get("id"),
                        "name": dimension.get("name"),
                        "type": dimension.get("type"),
                        "position": dimension.get("position"),
                        "codelist_id": dimension.get("codelist_id"),
                    }
                    for dimension in data_structure.get("dimensions", [])
                ],
            },
            "request": result.get("request"),
        }

    def check_compatibility(self, profile, requirements):
        return check_dataset_compatibility(profile, requirements)

    def get_dimensions(self, dataset_id):
        structure = self.get_cached_structure(dataset_id)
        if structure is None:
            structure = self.get_structure(dataset_id)
        data_structure = structure.get("data_structure")
        if isinstance(data_structure, dict):
            dimensions = data_structure.get("dimensions")
        else:
            dimensions = structure.get("dimensions")
        if not isinstance(dimensions, list):
            raise ValueError(
                f"{self.source} structure has no ordered dimensions"
            )
        return dimensions

    def get_cached_structure(self, dataset_id):
        return get_dataset_structure(dataset_id, source=self.source)

    def search_concepts(
        self,
        query,
        dataset_id=None,
        top_k=10,
        embedding_model="nomic-local",
    ):
        return embedding.search_dataset_concepts(
            query,
            dataset_id=dataset_id,
            top_k=top_k,
            embedding_model=embedding_model,
            source=self.source,
        )

    def index_datasets(self, limit=None, embedding_model="nomic-local"):
        report = {
            "source": self.source,
            "discovered": 0,
            "indexed": [],
            "skipped": [],
            "failures": [],
        }
        try:
            datasets = self.discover_datasets(limit=limit)
        except Exception as error:
            report["failures"].append({
                "source": self.source,
                "dataset_id": None,
                "stage": "discovery",
                "error": f"{type(error).__name__}: {error}",
            })
            report["totals"] = self._index_totals(report)
            return report

        report["discovered"] = len(datasets)
        seen_dataset_ids = set()
        for discovered in datasets:
            dataset_id = discovered.get("dataset_id", "")
            title = discovered.get("title", "")
            if not dataset_id:
                report["failures"].append({
                    "source": self.source,
                    "dataset_id": None,
                    "title": title,
                    "stage": "discovery",
                    "error": "Discovered dataset is missing dataset_id",
                })
                continue
            if dataset_id in seen_dataset_ids:
                report["skipped"].append({
                    "source": self.source,
                    "dataset_id": dataset_id,
                    "title": title,
                    "reason": "Duplicate dataset in discovery results",
                })
                continue
            seen_dataset_ids.add(dataset_id)

            stage = "metadata"
            try:
                metadata = self.get_metadata(dataset_id)
                title = metadata.get("title") or title

                stage = "structure"
                structure = self.get_structure(dataset_id, metadata)

                stage = "concepts"
                concepts = self.build_concepts(
                    dataset_id,
                    metadata,
                    structure,
                )
                if not concepts:
                    report["skipped"].append({
                        "source": self.source,
                        "dataset_id": dataset_id,
                        "title": title,
                        "reason": "Dataset metadata contains no concepts",
                    })
                    continue

                stage = "embedding"
                cached = get_dataset_concept_embeddings(
                    dataset_id,
                    embedding_model,
                    source=self.source,
                )
                pending = []
                for concept in concepts:
                    concept["source"] = self.source
                    if concept["concept_role"] != "semantic":
                        concept["embedding"] = None
                        concept["embedding_model"] = None
                        continue

                    text = concept.get("semantic_text")
                    if not text:
                        text = (
                            f"{self.source} dataset: {title}\n"
                            f"Dimension: {concept['dimension_name']}\n"
                            f"Code: {concept['code']}\n"
                            f"Label: {concept['label']}"
                        )
                    concept["semantic_text"] = text
                    concept["embedding_model"] = embedding_model
                    existing = cached.get((
                        concept["dimension_id"],
                        concept["code"],
                    ))
                    if existing is not None and existing[0] == text:
                        concept["embedding"] = existing[1]
                    else:
                        concept["embedding"] = None
                        pending.append((concept, text))

                vectors = (
                    embedding.embed_texts(
                        (text for _, text in pending),
                        embedding_model,
                    )
                    if pending
                    else []
                )
                if len(vectors) != len(pending):
                    raise ValueError(
                        "Embedding response count does not match pending concepts"
                    )
                for (concept, _), vector in zip(pending, vectors):
                    concept["embedding"] = vector

                stage = "persistence"
                dimensions = (
                    structure.get("data_structure", {}).get("dimensions")
                    if isinstance(structure.get("data_structure"), dict)
                    else structure.get("dimensions")
                )
                if not isinstance(dimensions, list):
                    raise ValueError(
                        "Dataset structure must include ordered dimensions"
                    )
                save_dataset_record(
                    source=self.source,
                    dataset_id=dataset_id,
                    title=title or dataset_id,
                    description=metadata.get("description"),
                    source_url=metadata.get("source_url"),
                    metadata=metadata.get("raw", metadata),
                    is_latest_release=metadata.get("is_latest_release"),
                )
                save_dataset_structure(structure, source=self.source)
                save_dataset_concepts(concepts)

                report["indexed"].append({
                    "source": self.source,
                    "dataset_id": dataset_id,
                    "title": title or dataset_id,
                    "total": len(concepts),
                    "semantic": sum(
                        concept["concept_role"] == "semantic"
                        for concept in concepts
                    ),
                    "filter": sum(
                        concept["concept_role"] == "filter"
                        for concept in concepts
                    ),
                    "embedded": sum(
                        concept.get("embedding") is not None
                        for concept in concepts
                    ),
                    "embeddings_created": len(vectors),
                    "embedding_model": embedding_model,
                })
            except Exception as error:
                report["failures"].append({
                    "source": self.source,
                    "dataset_id": dataset_id,
                    "title": title,
                    "stage": stage,
                    "error": f"{type(error).__name__}: {error}",
                })

        report["totals"] = self._index_totals(report)
        return report

    @staticmethod
    def _index_totals(report):
        indexed = report["indexed"]
        return {
            "datasets_indexed": len(indexed),
            "datasets_skipped": len(report["skipped"]),
            "datasets_failed": len(report["failures"]),
            "total_concepts": sum(item["total"] for item in indexed),
            "semantic_concepts": sum(item["semantic"] for item in indexed),
            "filter_concepts": sum(item["filter"] for item in indexed),
            "embedded_concepts": sum(item["embedded"] for item in indexed),
            "embeddings_created": sum(
                item["embeddings_created"] for item in indexed
            ),
        }

    def normalize_dataframe_checked(self, result, structure):
        dataframe = self.normalize_dataframe(result, structure)
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Data source normalization must return a pandas DataFrame")
        return dataframe
