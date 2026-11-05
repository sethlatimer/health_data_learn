import json
import tempfile
import unittest
from unittest.mock import Mock, patch

import database
import embedding
from data_sources.cms import CMSDataSource


CMS_DATASET_IDS = (
    "11111111-1111-4111-8111-111111111111",
    "22222222-2222-4222-8222-222222222222",
    "33333333-3333-4333-8333-333333333333",
)


def _catalog_fixture():
    descriptions = (
        "Inpatient utilization trends for Medicare hospital services.",
        "Hospital spending and payment data for covered services.",
        "Provider payments and service utilization in Medicare.",
    )
    titles = (
        "Medicare Inpatient Utilization",
        "Medicare Hospital Spending",
        "Medicare Provider Payments",
    )
    datasets = []
    for index, (dataset_id, title, description) in enumerate(
        zip(CMS_DATASET_IDS, titles, descriptions),
        start=1,
    ):
        endpoint = (
            "https://data.cms.gov/data-api/v1/dataset/"
            f"{dataset_id}/data"
        )
        datasets.append({
            "@id": endpoint + "-viewer",
            "@type": "Dataset",
            "identifier": endpoint + "-viewer",
            "title": title,
            "description": description,
            "modified": "2026-01-01",
            "keyword": ["Medicare", "Hospitals & Facilities"],
            "theme": ["Health Care Use & Payments"],
            "inSeries": [{
                "@id": endpoint,
                "title": title,
            }],
            "distribution": [{
                "@type": "Distribution",
                "format": "API",
                "accessURL": endpoint,
                "description": "latest",
                "describedBy": {
                    "downloadURL": endpoint.rsplit("/", 1)[0]
                    + "/dictionary",
                },
            }],
            "release_number": index,
        })
    return {
        "@type": "Catalog",
        "dataset": datasets,
        "datasetSeries": [],
    }


def _dictionary_fixture(dataset_id):
    index = CMS_DATASET_IDS.index(dataset_id)
    measure_names = (
        ("INPATIENT_UTILIZATION", "Inpatient Utilization"),
        ("HOSPITAL_SPENDING", "Hospital Spending"),
        ("PROVIDER_PAYMENT_AMOUNT", "Provider Payment Amount"),
    )
    measure_name, measure_label = measure_names[index]
    return {
        "identifier": f"dictionary-{index}",
        "data": {
            "title": "Official CMS data dictionary",
            "fields": [
                {
                    "name": "STATE_CD",
                    "label": "State Code",
                    "description": "State abbreviation for the provider.",
                },
                {
                    "name": "YEAR",
                    "label": "Reporting Year",
                    "description": "Calendar year represented by the record.",
                },
                {
                    "name": measure_name,
                    "label": measure_label,
                    "description": (
                        f"CMS field for {measure_label.casefold()}."
                    ),
                },
            ],
        },
    }


def _response(payload, url):
    result = Mock()
    result.status_code = 200
    result.headers = {
        "ETag": "test-catalog-etag",
        "Last-Modified": "Fri, 30 Oct 2026 12:00:00 GMT",
    }
    result.url = url
    result.json.return_value = payload
    result.raise_for_status.return_value = None
    return result


def _vector_for_text(text):
    value = text.casefold()
    if "inpatient utilization" in value:
        return [1.0, 0.0, 0.0]
    if "hospital spending" in value:
        return [0.0, 1.0, 0.0]
    if "provider payments" in value:
        return [0.0, 0.0, 1.0]
    return [0.05, 0.05, 0.05]


class CMSDataSourceTests(unittest.TestCase):
    def test_existing_oecd_catalog_schema_migrates_source_neutrally(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = f"{temp_dir}\\legacy.db"
            with patch.object(database, "DB", database_path):
                conn = database.connect()
                try:
                    conn.executescript(
                        """
                        CREATE TABLE dataset_concepts (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            dataset_id TEXT NOT NULL,
                            dimension_id TEXT NOT NULL,
                            dimension_name TEXT NOT NULL,
                            code TEXT NOT NULL,
                            label TEXT NOT NULL,
                            concept_role TEXT NOT NULL,
                            semantic_text TEXT,
                            embedding_json TEXT,
                            embedding_model TEXT,
                            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE (dataset_id, dimension_id, code)
                        );
                        CREATE TABLE dataset_structures (
                            dataset_id TEXT PRIMARY KEY,
                            structure_json TEXT NOT NULL,
                            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                        );
                        CREATE TABLE dataset_semantics (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            dataset_id TEXT NOT NULL UNIQUE,
                            title TEXT NOT NULL,
                            semantic_text TEXT NOT NULL,
                            embedding_json TEXT NOT NULL,
                            embedding_model TEXT NOT NULL,
                            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                        );
                        INSERT INTO dataset_concepts (
                            dataset_id, dimension_id, dimension_name, code,
                            label, concept_role, semantic_text, embedding_json,
                            embedding_model
                        ) VALUES (
                            'shared-id', 'MEASURE', 'Measure', 'M1',
                            'Measure one', 'semantic', 'OECD measure',
                            '[1,0]', 'nomic-local'
                        );
                        INSERT INTO dataset_structures (
                            dataset_id, structure_json
                        ) VALUES ('shared-id', '{\"dataset_id\":\"shared-id\",\"data_structure\":{\"dimensions\":[]}}');
                        INSERT INTO dataset_semantics (
                            dataset_id, title, semantic_text,
                            embedding_json, embedding_model
                        ) VALUES ('shared-id', 'OECD title', 'OECD text', '[1,0]', 'nomic-local');
                        """
                    )
                    database.ensure_dataset_concepts_table(conn)
                    database.ensure_dataset_structures_table(conn)
                    database.ensure_dataset_semantics_table(conn)
                    conn.execute(
                        """
                        INSERT INTO dataset_concepts (
                            source, dataset_id, dimension_id,
                            dimension_name, code, label, concept_role
                        ) VALUES (
                            'CMS', 'shared-id', 'MEASURE', 'Measure',
                            'M1', 'CMS measure', 'filter'
                        )
                        """
                    )
                    conn.commit()
                    concepts = conn.execute(
                        """
                        SELECT source, label FROM dataset_concepts
                        WHERE dataset_id = 'shared-id'
                        ORDER BY source
                        """
                    ).fetchall()
                    structures = conn.execute(
                        "PRAGMA table_info(dataset_structures)"
                    ).fetchall()
                    semantic_unique = [
                        tuple(
                            row[2]
                            for row in conn.execute(
                                f"PRAGMA index_info('{index[1]}')"
                            )
                        )
                        for index in conn.execute(
                            "PRAGMA index_list(dataset_semantics)"
                        )
                        if index[2]
                    ]
                finally:
                    conn.close()
            self.assertEqual(
                concepts,
                [("CMS", "CMS measure"), ("OECD", "Measure one")],
            )
            self.assertIn("source", {row[1] for row in structures})
            self.assertIn(("source", "dataset_id"), semantic_unique)

    def test_bounded_index_search_retrieval_and_idempotent_upsert(self):
        catalog = _catalog_fixture()

        def cms_response(url, params=None, headers=None, timeout=60):
            if url == "https://data.cms.gov/data.json":
                return _response(catalog, url)
            for dataset_id in CMS_DATASET_IDS:
                dictionary_url = (
                    "https://data.cms.gov/data-api/v1/dataset/"
                    f"{dataset_id}/dictionary"
                )
                if url == dictionary_url:
                    return _response(
                        _dictionary_fixture(dataset_id),
                        url,
                    )
            if url.endswith("/data"):
                return _response(
                    [{"STATE_CD": "CA", "YEAR": "2025", "OBS_VALUE": "10"}],
                    url,
                )
            raise AssertionError(f"Unexpected CMS endpoint: {url}")

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(database, "DB", f"{temp_dir}\\agent.db"):
                source = CMSDataSource()
                with (
                    patch(
                        "data_sources.cms._get_cms_response",
                        side_effect=cms_response,
                    ),
                    patch.object(
                        embedding,
                        "embed_texts",
                        side_effect=lambda texts, model: [
                            _vector_for_text(text)
                            for text in texts
                        ],
                    ) as embed_texts,
                ):
                    first_report = source.index_datasets(limit=5)

                    self.assertEqual(first_report["source"], "CMS")
                    self.assertEqual(first_report["discovered"], 3)
                    self.assertEqual(first_report["totals"]["datasets_indexed"], 3)
                    self.assertGreater(
                        first_report["totals"]["semantic_concepts"],
                        3,
                    )
                    self.assertGreater(
                        first_report["totals"]["filter_concepts"],
                        0,
                    )
                    self.assertEqual(
                        first_report["totals"]["embedded_concepts"],
                        first_report["totals"]["semantic_concepts"],
                    )
                    self.assertEqual(first_report["failures"], [])

                    for query, expected_id in zip(
                        (
                            "inpatient utilization",
                            "hospital spending",
                            "provider payments",
                        ),
                        CMS_DATASET_IDS,
                    ):
                        with patch.object(
                            embedding,
                            "embed_text",
                            return_value=_vector_for_text(query),
                        ):
                            matches = embedding.search_dataset_concepts(
                                query,
                                top_k=5,
                                source="CMS",
                            )
                        self.assertEqual(matches[0]["source"], "CMS")
                        self.assertEqual(matches[0]["dataset_id"], expected_id)
                        self.assertTrue(all(
                            match["dimension_id"] != "__dataset__"
                            for match in matches
                        ))

                    filters = source.find_filters(
                        CMS_DATASET_IDS[0],
                    )
                    self.assertTrue(filters)
                    self.assertTrue(all(
                        item["source"] == "CMS"
                        for item in filters
                    ))

                    structure = source.get_structure(CMS_DATASET_IDS[0])
                    with patch(
                        "data_sources.cms._get_cms_response",
                        side_effect=cms_response,
                    ) as cms_get:
                        result = source.retrieve_data(
                            CMS_DATASET_IDS[0],
                            filters={"STATE_CD": "CA"},
                            limit=1,
                        )
                    self.assertEqual(result["request"]["limit"], 1)
                    self.assertEqual(result["request"]["params"]["size"], 1)
                    dataframe = source.normalize_dataframe(result, structure)
                    self.assertEqual(dataframe.loc[0, "source"], "CMS")
                    self.assertEqual(
                        dataframe.loc[0, "dataset_id"],
                        CMS_DATASET_IDS[0],
                    )
                    self.assertEqual(dataframe.loc[0, "STATE_CD"], "CA")
                    self.assertEqual(cms_get.call_count, 1)

                    with patch.object(
                        embedding,
                        "embed_texts",
                        side_effect=lambda texts, model: [
                            _vector_for_text(text)
                            for text in texts
                        ],
                    ) as rerun_embeddings:
                        second_report = source.index_datasets(limit=5)

                    self.assertEqual(
                        second_report["totals"]["datasets_indexed"],
                        3,
                    )
                    self.assertEqual(
                        second_report["totals"]["embeddings_created"],
                        0,
                    )
                    self.assertEqual(
                        rerun_embeddings.call_count,
                        0,
                    )

                conn = database.connect()
                try:
                    concept_count = conn.execute(
                        "SELECT COUNT(*) FROM dataset_concepts WHERE source = 'CMS'"
                    ).fetchone()[0]
                    dataset_count = conn.execute(
                        "SELECT COUNT(*) FROM dataset_records WHERE source = 'CMS'"
                    ).fetchone()[0]
                    duplicate_count = conn.execute(
                        """
                        SELECT COUNT(*) FROM (
                            SELECT source, dataset_id, dimension_id, code
                            FROM dataset_concepts
                            WHERE source = 'CMS'
                            GROUP BY source, dataset_id, dimension_id, code
                            HAVING COUNT(*) > 1
                        )
                        """
                    ).fetchone()[0]
                finally:
                    conn.close()
                self.assertEqual(dataset_count, 3)
                self.assertGreater(concept_count, 0)
                self.assertEqual(duplicate_count, 0)

                with patch.object(
                    embedding,
                    "embed_text",
                    return_value=[0.0, 0.0, 1.0],
                ):
                    database.save_dataset_concepts([{
                        "source": "OECD",
                        "dataset_id": "OECD.TEST,DSD@HEALTH,1.0",
                        "dimension_id": "MEASURE",
                        "dimension_name": "Measure",
                        "code": "PROVIDER_PAYMENTS",
                        "label": "Provider payments",
                        "concept_role": "semantic",
                        "semantic_text": "OECD provider payments",
                        "embedding": [0.0, 0.0, 1.0],
                        "embedding_model": "nomic-local",
                    }])
                    cross_source = embedding.search_dataset_concepts(
                        "provider payments",
                        top_k=10,
                    )
                self.assertEqual(
                    {item["source"] for item in cross_source},
                    {"CMS", "OECD"},
                )

                print(
                    "CMS bounded index: "
                    f"{first_report['totals']['datasets_indexed']} datasets, "
                    f"{first_report['totals']['semantic_concepts']} semantic, "
                    f"{first_report['totals']['filter_concepts']} filters, "
                    f"{first_report['totals']['embedded_concepts']} embedded"
                )
                print("Cross-source sample matches:")
                for item in cross_source[:3]:
                    print(
                        f"  {item['source']} {item['dataset_id']} "
                        f"{item['label']} ({item['score']:.3f})"
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
