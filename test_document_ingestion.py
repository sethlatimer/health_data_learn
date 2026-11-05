import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import database
import embedding
from data_sources.documents import (
    CMSDocumentSource,
    DocumentIngestionService,
    OECDDocumentSource,
    WHODocumentSource,
)
from data_sources.documents.oecd import OECDDocumentSource


OECD_DATASET = "OECD_TEST_HEALTH"
CMS_DATASET = "cms-document-fixture"
WHO_DATASET = "WHO_TEST_INDICATOR"


def _profiles():
    return [
        {
            "source": "OECD",
            "dataset_id": OECD_DATASET,
            "title": "OECD primary care outcomes",
            "metadata": {
                "documentation": {
                    "url": "https://www.oecd.org/health/test-methodology.html",
                },
            },
            "structure": {},
        },
        {
            "source": "CMS",
            "dataset_id": CMS_DATASET,
            "title": "CMS hospital payments",
            "metadata": {
                "identifier": CMS_DATASET,
                "distribution": [{
                    "format": "API",
                    "describedBy": {
                        "downloadURL": (
                            "https://data.cms.gov/api/1/datastore/"
                            "dictionary.json"
                        ),
                    },
                }],
            },
            "structure": {
                "data_structure": {
                    "dimensions": [{
                        "id": "PAYMENT_AMOUNT",
                        "type": "MeasureDimension",
                        "metadata": {
                            "name": "PAYMENT_AMOUNT",
                            "label": "Total Payment Amount",
                        },
                    }, {
                        "id": "Rndrng_Prvdr_CCN",
                        "name": "Rendering Provider CCN",
                        "type": "MeasureDimension",
                        "metadata": {
                            "name": "Rndrng_Prvdr_CCN",
                            "label": "Rendering Provider CCN",
                        },
                    }],
                },
            },
        },
        {
            "source": "WHO",
            "dataset_id": WHO_DATASET,
            "title": "Hospital admission rate",
            "source_url": (
                "https://www.who.int/data/gho/data/indicators/"
                "indicator-details/GHO/hospital-admission-rate"
            ),
            "metadata": {
                "IndicatorCode": WHO_DATASET,
                "Definition": "Hospital admission rate per population.",
                "MethodOfMeasurement": "Admissions divided by population.",
                "Unit": "Admissions per 100,000 population",
                "LastUpdated": "2025-01-01",
            },
            "structure": {},
        },
    ]


class _DownloadResponse:
    def __init__(self, content, content_type):
        self.status_code = 200
        self.headers = {"Content-Type": content_type}
        self._content = content

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        yield self._content


class DocumentIngestionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = str(Path(self.temp_dir.name) / "documents.db")
        self.cache_path = Path(self.temp_dir.name) / "cache"
        self.db_patch = patch.object(database, "DB", self.database_path)
        self.db_patch.start()
        conn = database.connect()
        database.ensure_document_tables(conn)
        conn.commit()
        conn.close()
        self.profiles_patch = patch(
            "data_sources.documents.base.database.get_indexed_dataset_profiles",
            return_value=_profiles(),
        )
        self.profiles_patch.start()
        self.sources = {
            "OECD": OECDDocumentSource(),
            "CMS": CMSDocumentSource(),
            "WHO": WHODocumentSource(),
        }

    def tearDown(self):
        self.profiles_patch.stop()
        self.db_patch.stop()
        self.temp_dir.cleanup()

    def _service(self):
        return DocumentIngestionService(
            self.sources,
            cache_root=str(self.cache_path),
        )

    def test_source_discovery_preserves_explicit_metadata_links(self):
        service = self._service()
        references = service.discover()
        self.assertEqual(
            {reference.source for reference in references},
            {"OECD", "CMS", "WHO"},
        )
        by_source = {reference.source: reference for reference in references}
        self.assertEqual(by_source["OECD"].dataset_id, OECD_DATASET)
        self.assertEqual(by_source["CMS"].document_type, "data_dictionary")
        self.assertEqual(by_source["CMS"].measure_ids, ["PAYMENT_AMOUNT"])
        self.assertEqual(by_source["WHO"].measure_ids, [WHO_DATASET])
        self.assertIn("Admissions divided by population", by_source["WHO"].inline_text)

    def test_oecd_discovery_reads_explicit_official_links_from_description(self):
        profile = {
            "source": "OECD",
            "dataset_id": OECD_DATASET,
            "title": "OECD hospital utilization",
            "description": (
                '<p>Methodology is documented in the '
                '<a href="https://www.oecd.org/health/test-methodology.pdf">'
                "technical methodology</a>.</p>"
                '<a href="https://www.oecd.org/health/indicator-definitions.pdf">'
                "HCQO Definitions document</a>"
                '<a href="https://www.oecd.org/health/sources-methods.xlsx">'
                "Primary Care data document</a>"
                '<a href="https://example.org/methodology.pdf">external methodology</a>'
                '<a href="https://www.oecd.org/data-explorer/">data explorer</a>'
            ),
            "metadata": {},
            "structure": {},
        }
        references = OECDDocumentSource().discover_documents([profile])
        self.assertEqual(len(references), 2)
        self.assertEqual(
            references[1].source_url,
            "https://www.oecd.org/health/indicator-definitions.pdf",
        )
        self.assertFalse(any(ref.source_url.endswith(".xlsx") for ref in references))
        self.assertEqual(
            references[0].source_url,
            "https://www.oecd.org/health/test-methodology.pdf",
        )
        self.assertEqual(
            references[0].metadata["link_label"],
            "technical methodology",
        )

    def test_discovery_ingestion_search_provenance_and_idempotence(self):
        downloads = {
            "https://www.oecd.org/health/test-methodology.html": (
                b"<html><nav>Navigation</nav><h1>Methodology</h1>"
                b"<p>OECD primary care measure definitions and exclusions.</p>"
                b"<footer>Copyright 2025</footer></html>",
                "text/html",
            ),
            "https://data.cms.gov/api/1/datastore/dictionary.json": (
                json.dumps({
                    "data": {
                        "fields": [{
                            "name": "PAYMENT_AMOUNT",
                            "label": "Provider payment amount",
                            "description": "Total payment made to the provider.",
                        }],
                    },
                }).encode("utf-8"),
                "application/json",
            ),
        }

        def fake_get(url, **kwargs):
            payload, content_type = downloads[url]
            return _DownloadResponse(payload, content_type)

        vector_calls = []

        def fake_embedding(text, embedding_model):
            vector_calls.append(text)
            return [1.0, 0.0, 0.0]

        with patch(
            "data_sources.documents.base.requests.get",
            side_effect=fake_get,
        ), patch.object(
            embedding,
            "generate_embedding",
            side_effect=fake_embedding,
        ), patch.object(
            embedding,
            "embed_text",
            return_value=[1.0, 0.0, 0.0],
        ):
            conn = database.connect()
            try:
                database.ensure_dataset_semantics_table(conn)
                database.ensure_dataset_concepts_table(conn)
                conn.execute(
                    """
                    INSERT INTO dataset_semantics (
                        source, dataset_id, title, semantic_text,
                        embedding_json, embedding_model
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "OECD",
                        OECD_DATASET,
                        "OECD hospital utilization",
                        "Structured dataset summary",
                        json.dumps([1.0, 0.0, 0.0]),
                        "fixture",
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO dataset_concepts (
                        source, dataset_id, dimension_id, dimension_name,
                        code, label, concept_role, semantic_text,
                        embedding_json, embedding_model
                    )
                    VALUES (?, ?, ?, ?, ?, ?, 'semantic', ?, ?, ?)
                    """,
                    (
                        "OECD",
                        OECD_DATASET,
                        "MEASURE",
                        "Measure",
                        "ADMISSION_RATE",
                        "Hospital admission rate",
                        "Hospital admission rate measure",
                        json.dumps([1.0, 0.0, 0.0]),
                        "fixture",
                    ),
                )
                conn.commit()
            finally:
                conn.close()
            service = self._service()
            first = service.ingest(embed=True)
            pdf_path = self.cache_path / "cms" / "methodology.pdf"
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            pdf_path.write_bytes(b"fixture PDF data")
            pdf_document_id, _ = database.save_source_document({
                "source": "CMS",
                "title": "CMS payment methodology",
                "document_type": "methodology_pdf",
                "source_url": "https://www.cms.gov/files/methodology.pdf",
                "dataset_id": CMS_DATASET,
                "file_name": pdf_path.name,
                "file_path": str(pdf_path),
                "checksum": "pdf-content-checksum",
                "raw_text": "Provider payment methodology from the CMS PDF.",
                "content_type": "application/pdf",
                "extraction_status": "extracted",
            })
            database.save_document_chunk(
                pdf_document_id,
                3,
                1,
                "Provider payment methodology from the CMS PDF.",
                section="Payment methods",
                stable_chunk_id="cms-methodology-pdf-chunk",
            )
            conn = database.connect()
            try:
                conn.execute(
                    """
                    UPDATE document_chunks
                    SET embedding_json = ?
                    WHERE document_id = ?
                    """,
                    (json.dumps([1.0, 0.0, 0.0]), pdf_document_id),
                )
                conn.commit()
            finally:
                conn.close()
            counts_after_first = database.count_source_documents()
            second = service.ingest(embed=True)
            search_results = embedding.search_source_documents(
                "hospital admission methods",
                source="WHO",
                dataset_id=WHO_DATASET,
                measure_id=WHO_DATASET,
            )
            oecd_results = embedding.search_source_documents(
                "measure definitions",
                source="OECD",
                dataset_id=OECD_DATASET,
            )
            pdf_results = embedding.search_source_documents(
                "provider payment methodology",
                source="CMS",
                dataset_id=CMS_DATASET,
            )
            document_scope = embedding.search_corpus(
                "hospital payments",
                scopes={"documents"},
            )
            summary_scope = embedding.search_corpus(
                "hospital payments",
                scopes={"dataset_summaries"},
            )
            concept_results = embedding.search_dataset_concepts(
                "hospital admission methods",
            )

        self.assertEqual(first["discovered"], 3)
        self.assertEqual(len(first["ingested"]), 3)
        self.assertFalse(first["failures"])
        self.assertEqual(counts_after_first["documents"], 4)
        self.assertGreaterEqual(counts_after_first["chunks"], 4)
        self.assertEqual(counts_after_first["chunks"], counts_after_first["embedded_chunks"])
        self.assertEqual(len(second["skipped"]), 3)
        self.assertFalse(second["failures"])
        self.assertEqual(
            database.count_source_documents(),
            counts_after_first,
        )
        self.assertEqual(
            sum(
                call.startswith("search_document: ")
                for call in vector_calls
            ),
            counts_after_first["embedded_chunks"] - 1,
        )
        self.assertTrue(search_results)
        result = search_results[0]
        self.assertEqual(result["source"], "WHO")
        self.assertEqual(result["dataset_id"], WHO_DATASET)
        self.assertEqual(result["measure_ids"], [WHO_DATASET])
        self.assertIn("indicator metadata", result["title"])
        self.assertTrue(result["source_url"].startswith("https://www.who.int/"))
        self.assertTrue(oecd_results)
        self.assertEqual(oecd_results[0]["section"], "Methodology")
        self.assertNotIn("Navigation", oecd_results[0]["content"])
        self.assertNotIn("Copyright", oecd_results[0]["content"])
        self.assertTrue(pdf_results)
        pdf_result = next(
            item for item in pdf_results
            if item["document_type"] == "methodology_pdf"
        )
        self.assertEqual(pdf_result["page_number"], 3)
        self.assertTrue(document_scope)
        self.assertTrue(
            all(item["source_type"] == "document" for item in document_scope)
        )
        self.assertEqual(len(summary_scope), 1)
        self.assertEqual(summary_scope[0]["source_type"], "oecd_dataset")
        self.assertEqual(len(concept_results), 1)
        self.assertEqual(concept_results[0]["code"], "ADMISSION_RATE")
        self.assertEqual(concept_results[0]["source"], "OECD")
        self.assertEqual(concept_results[0]["dataset_id"], OECD_DATASET)
        self.assertEqual(oecd_results[0]["source"], concept_results[0]["source"])
        self.assertEqual(
            oecd_results[0]["dataset_id"],
            concept_results[0]["dataset_id"],
        )
        self.assertNotIn("content", concept_results[0])

    def test_document_search_can_filter_by_source_and_dataset(self):
        source = self.sources["WHO"]
        reference = source.discover_documents([_profiles()[2]])[0]
        with patch.object(
            database,
            "get_indexed_dataset_profiles",
            return_value=[_profiles()[2]],
        ), patch.object(
            embedding,
            "generate_embedding",
            return_value=[1.0, 0.0],
        ):
            service = self._service()
            service.ingest(source="WHO", dataset_id=WHO_DATASET, embed=False)
            document_id = database.get_source_document(
                "WHO",
                reference.source_url,
            )["document_id"]
            conn = database.connect()
            try:
                conn.execute(
                    """
                    UPDATE document_chunks
                    SET embedding_json = ?
                    WHERE document_id = ?
                    """,
                    (json.dumps([1.0, 0.0]), document_id),
                )
                conn.commit()
            finally:
                conn.close()
            by_document = embedding.search_source_documents(
                "definition", document_id=document_id
            )
            wrong_source = embedding.search_source_documents(
                "definition", source="CMS"
            )
        self.assertTrue(by_document)
        self.assertEqual(by_document[0]["document_id"], document_id)
        self.assertEqual(wrong_source, [])

    def test_pdf_extraction_reuses_existing_page_parser_and_keeps_pages(self):
        pdf_path = Path(self.temp_dir.name) / "guide.pdf"
        pdf_path.write_bytes(b"fixture")
        with patch(
            "data_sources.documents.base.read_pdf",
            return_value=[
                {"page_number": 1, "text": "METHODS\n\nProvider payment definitions."},
                {"page_number": 2, "text": "LIMITATIONS\n\nValues are reported as published."},
            ],
        ):
            raw_text, chunks = DocumentIngestionService._extract_document(
                pdf_path,
                "application/pdf",
            )
        self.assertIn("[Page 1]", raw_text)
        self.assertEqual([chunk["page_number"] for chunk in chunks], [1, 2])
        self.assertEqual([chunk["section"] for chunk in chunks], [
            "METHODS",
            "LIMITATIONS",
        ])

    def test_download_errors_are_reported_with_source_dataset_and_stage(self):
        with patch(
            "data_sources.documents.base.requests.get",
            side_effect=__import__("requests").HTTPError("429 Too Many Requests"),
        ):
            report = self._service().ingest(
                source="OECD",
                dataset_id=OECD_DATASET,
            )
        self.assertEqual(len(report["failures"]), 1)
        failure = report["failures"][0]
        self.assertEqual(failure["source"], "OECD")
        self.assertEqual(failure["dataset_id"], OECD_DATASET)
        self.assertEqual(failure["stage"], "download")
        conn = database.connect()
        try:
            row = conn.execute(
                """
                SELECT source, dataset_id, stage
                FROM document_ingestion_failures
                """
            ).fetchone()
        finally:
            conn.close()
        self.assertEqual(row, ("OECD", OECD_DATASET, "download"))

    def test_existing_local_document_tables_migrate_without_data_loss(self):
        legacy_path = Path(self.temp_dir.name) / "legacy.db"
        with patch.object(database, "DB", str(legacy_path)):
            conn = database.connect()
            try:
                conn.executescript(
                    r"""
                    CREATE TABLE documents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_name TEXT NOT NULL,
                        file_path TEXT NOT NULL UNIQUE,
                        summary TEXT,
                        indexed_at TEXT DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE TABLE document_chunks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        document_id INTEGER NOT NULL,
                        page_number INTEGER,
                        chunk_number INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        embedding_json TEXT,
                        FOREIGN KEY (document_id) REFERENCES documents(id)
                    );
                    INSERT INTO documents (file_name, file_path)
                    VALUES ('legacy.pdf', 'C:\\Temp\\legacy.pdf');
                    INSERT INTO document_chunks (
                        document_id, page_number, chunk_number, content
                    )
                    VALUES (1, 4, 1, 'Preserved local PDF text.');
                    """
                )
                database.ensure_document_tables(conn)
                migrated_document = conn.execute(
                    "SELECT source, file_name FROM documents WHERE id = 1"
                ).fetchone()
                migrated_chunk = conn.execute(
                    """
                    SELECT page_number, content
                    FROM document_chunks
                    WHERE document_id = 1
                    """
                ).fetchone()
            finally:
                conn.close()
        self.assertEqual(migrated_document, ("LOCAL", "legacy.pdf"))
        self.assertEqual(migrated_chunk, (4, "Preserved local PDF text."))


if __name__ == "__main__":
    unittest.main()
