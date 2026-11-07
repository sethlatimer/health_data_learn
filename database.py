import json
import sqlite3
import time

from config import PROJECT_ROOT, get_settings

DB = str(get_settings().database_path)


def connect():
    connection = sqlite3.connect(DB)
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


def _unique_index_columns(conn, table):
    return {
        tuple(
            row[2]
            for row in conn.execute(
                f"PRAGMA index_info('{index[1]}')"
            )
        )
        for index in conn.execute(f"PRAGMA index_list({table})")
        if index[2]
    }


def _table_exists(conn, table):
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone() is not None


def ensure_document_tables(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL UNIQUE,
            summary TEXT,
            indexed_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    document_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(documents)")
    }
    document_additions = {
        "source": "TEXT NOT NULL DEFAULT 'LOCAL'",
        "title": "TEXT",
        "document_type": "TEXT",
        "source_url": "TEXT",
        "published_at": "TEXT",
        "updated_at": "TEXT",
        "dataset_id": "TEXT",
        "measure_ids_json": "TEXT",
        "checksum": "TEXT",
        "extraction_status": "TEXT",
        "raw_text": "TEXT",
        "content_type": "TEXT",
        "metadata_json": "TEXT",
    }
    for column, sql_type in document_additions.items():
        if column not in document_columns:
            conn.execute(
                f"ALTER TABLE documents ADD COLUMN {column} {sql_type}"
            )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS document_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            page_number INTEGER,
            chunk_number INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding_json TEXT,
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
        """
    )
    chunk_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(document_chunks)")
    }
    chunk_additions = {
        "section": "TEXT",
        "stable_chunk_id": "TEXT",
        "metadata_json": "TEXT",
        "content_hash": "TEXT",
    }
    for column, sql_type in chunk_additions.items():
        if column not in chunk_columns:
            conn.execute(
                f"ALTER TABLE document_chunks ADD COLUMN {column} {sql_type}"
            )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id
        ON document_chunks(document_id)
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_document_chunks_stable_id
        ON document_chunks(document_id, stable_chunk_id)
        WHERE stable_chunk_id IS NOT NULL
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_source_url
        ON documents(source, source_url)
        WHERE source_url IS NOT NULL
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS document_dataset_links (
            document_id INTEGER NOT NULL,
            source TEXT NOT NULL,
            dataset_id TEXT NOT NULL,
            measure_ids_json TEXT NOT NULL DEFAULT '[]',
            PRIMARY KEY (document_id, source, dataset_id),
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS document_ingestion_failures (
            source TEXT NOT NULL,
            source_url TEXT NOT NULL,
            dataset_id TEXT,
            stage TEXT NOT NULL,
            error TEXT NOT NULL,
            failed_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (source, source_url)
        )
        """
    )


def _begin_schema_migration(conn):
    if conn.in_transaction:
        return False
    conn.execute("BEGIN IMMEDIATE")
    return True


def _finish_schema_migration(conn, owns_transaction, success):
    if not owns_transaction:
        return
    if success:
        conn.commit()
    else:
        conn.rollback()


def ensure_dataset_semantics_table(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dataset_semantics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL DEFAULT 'OECD',
            dataset_id TEXT NOT NULL,
            title TEXT NOT NULL,
            semantic_text TEXT NOT NULL,
            embedding_json TEXT NOT NULL,
            embedding_model TEXT NOT NULL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (source, dataset_id)
        )
        """
    )
    columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(dataset_semantics)")
    }
    if "source" not in columns:
        conn.execute(
            "ALTER TABLE dataset_semantics "
            "ADD COLUMN source TEXT NOT NULL DEFAULT 'OECD'"
        )
    owns_transaction = _begin_schema_migration(conn)
    try:
        if _table_exists(conn, "dataset_semantics_new"):
            conn.execute("DROP TABLE dataset_semantics_new")
        index_columns = _unique_index_columns(
            conn,
            "dataset_semantics",
        )
        if ("dataset_id",) in index_columns:
            conn.execute(
                """
                CREATE TABLE dataset_semantics_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL DEFAULT 'OECD',
                    dataset_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    semantic_text TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (source, dataset_id)
                )
                """
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO dataset_semantics_new (
                    id, source, dataset_id, title, semantic_text,
                    embedding_json, embedding_model, updated_at
                )
                SELECT id, source, dataset_id, title, semantic_text,
                       embedding_json, embedding_model, updated_at
                FROM dataset_semantics
                """
            )
            conn.execute("DROP TABLE dataset_semantics")
            conn.execute(
                "ALTER TABLE dataset_semantics_new RENAME TO dataset_semantics"
            )
    except Exception:
        _finish_schema_migration(conn, owns_transaction, False)
        raise
    else:
        _finish_schema_migration(conn, owns_transaction, True)


def ensure_dataset_concepts_table(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dataset_concepts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL DEFAULT 'OECD',
            dataset_id TEXT NOT NULL,
            dimension_id TEXT NOT NULL,
            dimension_name TEXT NOT NULL,
            code TEXT NOT NULL,
            label TEXT NOT NULL,
            concept_role TEXT NOT NULL
                CHECK (concept_role IN ('semantic', 'filter')),
            semantic_text TEXT,
            embedding_json TEXT,
            embedding_model TEXT,
            metadata_json TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (source, dataset_id, dimension_id, code)
        )
        """
    )
    columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(dataset_concepts)")
    }
    if "semantic_text" not in columns:
        conn.execute("ALTER TABLE dataset_concepts ADD COLUMN semantic_text TEXT")
    if "source" not in columns:
        conn.execute(
            "ALTER TABLE dataset_concepts "
            "ADD COLUMN source TEXT NOT NULL DEFAULT 'OECD'"
        )
    if "metadata_json" not in columns:
        conn.execute("ALTER TABLE dataset_concepts ADD COLUMN metadata_json TEXT")
    owns_transaction = _begin_schema_migration(conn)
    try:
        if _table_exists(conn, "dataset_concepts_new"):
            conn.execute("DROP TABLE dataset_concepts_new")
        unique_indexes = _unique_index_columns(conn, "dataset_concepts")
        if ("dataset_id", "dimension_id", "code") in unique_indexes:
            conn.execute(
                """
                CREATE TABLE dataset_concepts_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL DEFAULT 'OECD',
                    dataset_id TEXT NOT NULL,
                    dimension_id TEXT NOT NULL,
                    dimension_name TEXT NOT NULL,
                    code TEXT NOT NULL,
                    label TEXT NOT NULL,
                    concept_role TEXT NOT NULL
                        CHECK (concept_role IN ('semantic', 'filter')),
                    semantic_text TEXT,
                    embedding_json TEXT,
                    embedding_model TEXT,
                    metadata_json TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (source, dataset_id, dimension_id, code)
                )
                """
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO dataset_concepts_new (
                    id, source, dataset_id, dimension_id, dimension_name, code,
                    label, concept_role, semantic_text, embedding_json,
                    embedding_model, metadata_json, updated_at
                )
                SELECT id, source, dataset_id, dimension_id, dimension_name,
                       code, label, concept_role, semantic_text,
                       embedding_json, embedding_model, metadata_json,
                       updated_at
                FROM dataset_concepts
                """
            )
            conn.execute("DROP TABLE dataset_concepts")
            conn.execute(
                "ALTER TABLE dataset_concepts_new RENAME TO dataset_concepts"
            )
    except Exception:
        _finish_schema_migration(conn, owns_transaction, False)
        raise
    else:
        _finish_schema_migration(conn, owns_transaction, True)
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS
            idx_dataset_concepts_source_identity
        ON dataset_concepts (source, dataset_id, dimension_id, code)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_dataset_concepts_source_role
        ON dataset_concepts (source, concept_role)
        """
    )


def ensure_dataset_records_table(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dataset_records (
            source TEXT NOT NULL,
            dataset_id TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            source_url TEXT,
            is_latest_release INTEGER,
            metadata_json TEXT NOT NULL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (source, dataset_id)
        )
        """
    )
    columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(dataset_records)")
    }
    if "is_latest_release" not in columns:
        conn.execute(
            "ALTER TABLE dataset_records "
            "ADD COLUMN is_latest_release INTEGER"
        )


def ensure_source_catalog_cache_table(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS source_catalog_cache (
            source TEXT PRIMARY KEY,
            payload_json TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            etag TEXT,
            last_modified TEXT
        )
        """
    )


def ensure_literature_tables(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS literature_records (
            pmid TEXT PRIMARY KEY,
            pmcid TEXT,
            doi TEXT,
            title TEXT NOT NULL,
            abstract TEXT,
            publication_year INTEGER,
            journal TEXT,
            authors_json TEXT NOT NULL DEFAULT '[]',
            publication_types_json TEXT NOT NULL DEFAULT '[]',
            keywords_json TEXT NOT NULL DEFAULT '[]',
            mesh_terms_json TEXT NOT NULL DEFAULT '[]',
            study_metadata_json TEXT NOT NULL DEFAULT '{}',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            source_url TEXT NOT NULL,
            full_text_url TEXT,
            full_text_available INTEGER NOT NULL DEFAULT 0,
            full_text_access_status TEXT NOT NULL DEFAULT 'metadata_only',
            full_text_reuse_status TEXT NOT NULL DEFAULT 'not_assessed',
            evidence_access_level TEXT NOT NULL DEFAULT 'metadata',
            semantic_text TEXT NOT NULL,
            embedding_json TEXT,
            embedding_model TEXT,
            retrieved_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    existing_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(literature_records)")
    }
    literature_column_additions = {
        "study_metadata_json": "TEXT NOT NULL DEFAULT '{}'",
        "full_text_url": "TEXT",
    }
    for column, sql_type in literature_column_additions.items():
        if column not in existing_columns:
            conn.execute(
                f"ALTER TABLE literature_records ADD COLUMN {column} {sql_type}"
            )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_literature_records_pmcid
        ON literature_records(pmcid)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_literature_records_doi
        ON literature_records(doi)
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS literature_search_cache (
            query_key TEXT PRIMARY KEY,
            query_text TEXT NOT NULL,
            pmids_json TEXT NOT NULL,
            max_results INTEGER NOT NULL,
            fetched_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def get_literature_records(pmids):
    pmids = list(dict.fromkeys(str(pmid) for pmid in pmids if pmid))
    if not pmids:
        return {}
    conn = connect()
    try:
        ensure_literature_tables(conn)
        placeholders = ",".join("?" for _ in pmids)
        rows = conn.execute(
            f"""
            SELECT pmid, pmcid, doi, title, abstract, publication_year,
                   journal, authors_json, publication_types_json,
                   keywords_json, mesh_terms_json, study_metadata_json,
                   metadata_json, source_url, full_text_url,
                   full_text_available, full_text_access_status,
                   full_text_reuse_status, evidence_access_level,
                   semantic_text, embedding_json, embedding_model,
                   retrieved_at
            FROM literature_records
            WHERE pmid IN ({placeholders})
            """,
            pmids,
        ).fetchall()
    finally:
        conn.close()
    return {
        row[0]: {
            "pmid": row[0],
            "pmcid": row[1],
            "doi": row[2],
            "title": row[3],
            "abstract": row[4],
            "publication_year": row[5],
            "journal": row[6],
            "authors": json.loads(row[7]),
            "publication_types": json.loads(row[8]),
            "keywords": json.loads(row[9]),
            "mesh_terms": json.loads(row[10]),
            "study_metadata": json.loads(row[11]),
            "metadata": json.loads(row[12]),
            "source_url": row[13],
            "full_text_url": row[14],
            "full_text_available": bool(row[15]),
            "full_text_access_status": row[16],
            "full_text_reuse_status": row[17],
            "evidence_access_level": row[18],
            "semantic_text": row[19],
            "embedding": json.loads(row[20]) if row[20] else None,
            "embedding_model": row[21],
            "retrieved_at": row[22],
        }
        for row in rows
    }


def upsert_literature_records(records):
    conn = connect()
    counts = {"inserted": 0, "updated": 0, "embedding_reused": 0}
    try:
        ensure_literature_tables(conn)
        for record in records:
            row = conn.execute(
                """
                SELECT semantic_text, embedding_json, embedding_model
                FROM literature_records WHERE pmid = ?
                """,
                (record["pmid"],),
            ).fetchone()
            embedding_json = (
                json.dumps(record["embedding"])
                if record.get("embedding") is not None
                else None
            )
            embedding_model = record.get("embedding_model")
            if (
                row
                and row[0] == record["semantic_text"]
                and row[1]
                and row[2] == embedding_model
            ):
                embedding_json = row[1]
                counts["embedding_reused"] += 1
            if row:
                counts["updated"] += 1
            else:
                counts["inserted"] += 1
            conn.execute(
                """
                INSERT INTO literature_records (
                    pmid, pmcid, doi, title, abstract, publication_year,
                    journal, authors_json, publication_types_json,
                    keywords_json, mesh_terms_json, study_metadata_json,
                    metadata_json, source_url, full_text_url,
                    full_text_available, full_text_access_status,
                    full_text_reuse_status, evidence_access_level,
                    semantic_text, embedding_json, embedding_model,
                    retrieved_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(pmid) DO UPDATE SET
                    pmcid = excluded.pmcid,
                    doi = excluded.doi,
                    title = excluded.title,
                    abstract = excluded.abstract,
                    publication_year = excluded.publication_year,
                    journal = excluded.journal,
                    authors_json = excluded.authors_json,
                    publication_types_json = excluded.publication_types_json,
                    keywords_json = excluded.keywords_json,
                    mesh_terms_json = excluded.mesh_terms_json,
                    study_metadata_json = excluded.study_metadata_json,
                    metadata_json = excluded.metadata_json,
                    source_url = excluded.source_url,
                    full_text_url = excluded.full_text_url,
                    full_text_available = excluded.full_text_available,
                    full_text_access_status = excluded.full_text_access_status,
                    full_text_reuse_status = excluded.full_text_reuse_status,
                    evidence_access_level = excluded.evidence_access_level,
                    semantic_text = excluded.semantic_text,
                    embedding_json = excluded.embedding_json,
                    embedding_model = excluded.embedding_model,
                    retrieved_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    record["pmid"],
                    record.get("pmcid"),
                    record.get("doi"),
                    record["title"],
                    record.get("abstract"),
                    record.get("publication_year"),
                    record.get("journal"),
                    json.dumps(record.get("authors") or [], ensure_ascii=False),
                    json.dumps(
                        record.get("publication_types") or [],
                        ensure_ascii=False,
                    ),
                    json.dumps(record.get("keywords") or [], ensure_ascii=False),
                    json.dumps(record.get("mesh_terms") or [], ensure_ascii=False),
                    json.dumps(
                        record.get("study_metadata") or {},
                        ensure_ascii=False,
                    ),
                    json.dumps(record.get("metadata") or {}, ensure_ascii=False),
                    record["source_url"],
                    record.get("full_text_url"),
                    int(bool(record.get("full_text_available"))),
                    record.get("full_text_access_status", "metadata_only"),
                    record.get("full_text_reuse_status", "not_assessed"),
                    record.get("evidence_access_level", "metadata"),
                    record["semantic_text"],
                    embedding_json,
                    embedding_model,
                ),
            )
        conn.commit()
        return counts
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_literature_search_cache(query_key, max_results):
    conn = connect()
    try:
        ensure_literature_tables(conn)
        row = conn.execute(
            """
            SELECT pmids_json, max_results
            FROM literature_search_cache
            WHERE query_key = ?
            """,
            (query_key,),
        ).fetchone()
    finally:
        conn.close()
    if row is None or row[1] < max_results:
        return None
    return json.loads(row[0])[:max_results]


def save_literature_search_cache(query_key, query_text, pmids, max_results):
    conn = connect()
    try:
        ensure_literature_tables(conn)
        conn.execute(
            """
            INSERT INTO literature_search_cache (
                query_key, query_text, pmids_json, max_results, fetched_at
            )
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(query_key) DO UPDATE SET
                query_text = excluded.query_text,
                pmids_json = excluded.pmids_json,
                max_results = excluded.max_results,
                fetched_at = CURRENT_TIMESTAMP
            """,
            (
                query_key,
                query_text,
                json.dumps(pmids),
                max_results,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def save_source_catalog_cache(
    source,
    payload,
    fetched_at,
    etag=None,
    last_modified=None,
):
    conn = connect()
    try:
        ensure_source_catalog_cache_table(conn)
        conn.execute(
            """
            INSERT INTO source_catalog_cache (
                source, payload_json, fetched_at, etag, last_modified
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(source) DO UPDATE SET
                payload_json = excluded.payload_json,
                fetched_at = excluded.fetched_at,
                etag = excluded.etag,
                last_modified = excluded.last_modified
            """,
            (
                source,
                json.dumps(payload, ensure_ascii=False),
                fetched_at,
                etag,
                last_modified,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_source_catalog_cache(source):
    conn = connect()
    try:
        ensure_source_catalog_cache_table(conn)
        row = conn.execute(
            """
            SELECT payload_json, fetched_at, etag, last_modified
            FROM source_catalog_cache
            WHERE source = ?
            """,
            (source,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {
        "payload": json.loads(row[0]),
        "fetched_at": row[1],
        "etag": row[2],
        "last_modified": row[3],
    }


def ensure_dataset_structures_table(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dataset_structures (
            dataset_id TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'OECD',
            structure_json TEXT NOT NULL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (source, dataset_id)
        )
        """
    )
    columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(dataset_structures)")
    }
    if "source" not in columns:
        conn.execute(
            "ALTER TABLE dataset_structures "
            "ADD COLUMN source TEXT NOT NULL DEFAULT 'OECD'"
        )
    owns_transaction = _begin_schema_migration(conn)
    try:
        if _table_exists(conn, "dataset_structures_new"):
            conn.execute("DROP TABLE dataset_structures_new")
        primary_key = tuple(
            row[1]
            for row in sorted(
                (row for row in conn.execute(
                    "PRAGMA table_info(dataset_structures)"
                ) if row[5]),
                key=lambda row: row[5],
            )
        )
        if primary_key == ("dataset_id",):
            conn.execute(
                """
                CREATE TABLE dataset_structures_new (
                    dataset_id TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'OECD',
                    structure_json TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (source, dataset_id)
                )
                """
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO dataset_structures_new (
                    dataset_id, source, structure_json, updated_at
                )
                SELECT dataset_id, source, structure_json, updated_at
                FROM dataset_structures
                """
            )
            conn.execute("DROP TABLE dataset_structures")
            conn.execute(
                "ALTER TABLE dataset_structures_new RENAME TO dataset_structures"
            )
    except Exception:
        _finish_schema_migration(conn, owns_transaction, False)
        raise
    else:
        _finish_schema_migration(conn, owns_transaction, True)
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_dataset_structures_source
        ON dataset_structures (source, dataset_id)
        """
    )


def save_dataset_structure(structure, source="OECD"):
    dataset_id = structure.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("Dataset structure must include a dataset_id")
    dimensions = structure.get("data_structure", {}).get(
        "dimensions",
        structure.get("dimensions"),
    )
    if not isinstance(dimensions, list):
        raise ValueError("Dataset structure must include ordered dimensions")

    conn = connect()
    try:
        ensure_dataset_structures_table(conn)
        conn.execute(
            """
            INSERT INTO dataset_structures (
                dataset_id, source, structure_json, updated_at
            )
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(source, dataset_id) DO UPDATE SET
                source = excluded.source,
                structure_json = excluded.structure_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                dataset_id,
                source,
                json.dumps(structure, ensure_ascii=False),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_dataset_structure(dataset_id, source="OECD"):
    conn = connect()
    try:
        ensure_dataset_structures_table(conn)
        row = conn.execute(
            """
            SELECT structure_json
            FROM dataset_structures
            WHERE dataset_id = ?
              AND source = ?
            """,
            (dataset_id, source),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None
    return json.loads(row[0])


def save_dataset_concepts(concepts):
    conn = connect()
    try:
        ensure_dataset_concepts_table(conn)
        conn.executemany(
            """
            INSERT INTO dataset_concepts (
                source,
                dataset_id,
                dimension_id,
                dimension_name,
                code,
                label,
                concept_role,
                semantic_text,
                embedding_json,
                embedding_model,
                metadata_json,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(source, dataset_id, dimension_id, code) DO UPDATE SET
                dimension_name = excluded.dimension_name,
                label = excluded.label,
                concept_role = excluded.concept_role,
                semantic_text = excluded.semantic_text,
                embedding_json = excluded.embedding_json,
                embedding_model = excluded.embedding_model,
                metadata_json = excluded.metadata_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            [
                (
                    concept.get("source", "OECD"),
                    concept["dataset_id"],
                    concept["dimension_id"],
                    concept["dimension_name"],
                    concept["code"],
                    concept["label"],
                    concept["concept_role"],
                    concept.get("semantic_text"),
                    json.dumps(concept["embedding"])
                    if concept.get("embedding") is not None
                    else None,
                    concept.get("embedding_model"),
                    json.dumps(
                        concept.get("metadata", {}),
                        ensure_ascii=False,
                    ),
                )
                for concept in concepts
            ],
        )
        conn.commit()
    finally:
        conn.close()


def get_dataset_concept_embeddings(
    dataset_id,
    embedding_model,
    source="OECD",
):
    conn = connect()
    try:
        ensure_dataset_concepts_table(conn)
        rows = conn.execute(
            """
            SELECT dimension_id, code, semantic_text, embedding_json
            FROM dataset_concepts
            WHERE dataset_id = ?
              AND source = ?
              AND concept_role = 'semantic'
              AND embedding_json IS NOT NULL
              AND embedding_model = ?
            """,
            (dataset_id, source, embedding_model),
        ).fetchall()
    finally:
        conn.close()

    return {
        (dimension_id, code): (semantic_text, json.loads(embedding_json))
        for dimension_id, code, semantic_text, embedding_json in rows
    }


def save_dataset_semantics(
    dataset_id,
    title,
    semantic_text,
    embedding,
    embedding_model,
    source="OECD",
):
    conn = connect()
    try:
        ensure_dataset_semantics_table(conn)
        conn.execute(
            """
            INSERT INTO dataset_semantics (
                dataset_id,
                title,
                semantic_text,
                embedding_json,
                embedding_model,
                source,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(source, dataset_id) DO UPDATE SET
                title = excluded.title,
                semantic_text = excluded.semantic_text,
                embedding_json = excluded.embedding_json,
                embedding_model = excluded.embedding_model,
                source = excluded.source,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                dataset_id,
                title,
                semantic_text,
                json.dumps(embedding),
                embedding_model,
                source,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def save_dataset_record(
    source,
    dataset_id,
    title,
    description,
    source_url,
    metadata,
    is_latest_release=None,
):
    if not source or not dataset_id:
        raise ValueError("Dataset source and dataset_id are required")
    conn = connect()
    try:
        ensure_dataset_records_table(conn)
        conn.execute(
            """
            INSERT INTO dataset_records (
                source, dataset_id, title, description, source_url,
                is_latest_release, metadata_json, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(source, dataset_id) DO UPDATE SET
                title = excluded.title,
                description = excluded.description,
                source_url = excluded.source_url,
                is_latest_release = excluded.is_latest_release,
                metadata_json = excluded.metadata_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                source,
                dataset_id,
                title,
                description,
                source_url,
                (
                    int(is_latest_release)
                    if is_latest_release is not None
                    else None
                ),
                json.dumps(metadata, ensure_ascii=False),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_dataset_record(source, dataset_id):
    conn = connect()
    try:
        ensure_dataset_records_table(conn)
        row = conn.execute(
            """
            SELECT title, description, source_url, metadata_json,
                   is_latest_release
            FROM dataset_records
            WHERE source = ? AND dataset_id = ?
            """,
            (source, dataset_id),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {
        "source": source,
        "dataset_id": dataset_id,
        "title": row[0],
        "description": row[1],
        "source_url": row[2],
        "metadata": json.loads(row[3]),
        "is_latest_release": (
            bool(row[4]) if row[4] is not None else None
        ),
    }


def get_indexed_dataset_profiles(
    source=None,
    include_concepts=True,
    instrumentation=None,
):
    profile_started = time.perf_counter()
    sqlite_started = time.perf_counter()
    conn = connect()
    try:
        ensure_dataset_structures_table(conn)
        ensure_dataset_records_table(conn)
        ensure_dataset_concepts_table(conn)
        sql = """
            SELECT s.source, s.dataset_id, s.structure_json,
                   r.title, r.description, r.source_url, r.metadata_json,
                   r.is_latest_release
            FROM dataset_structures s
            LEFT JOIN dataset_records r
              ON r.source = s.source
             AND r.dataset_id = s.dataset_id
        """
        params = ()
        if source is not None:
            sql += " WHERE s.source = ?"
            params = (source.upper(),)
        rows = conn.execute(sql, params).fetchall()
        if include_concepts:
            concept_sql = """
                SELECT source, dataset_id, dimension_id, dimension_name, code,
                       label, semantic_text, metadata_json
                FROM dataset_concepts
                WHERE concept_role = 'semantic'
            """
            concept_params = ()
            if source is not None:
                concept_sql += " AND source = ?"
                concept_params = (source.upper(),)
            concepts = conn.execute(
                concept_sql,
                concept_params,
            ).fetchall()
        else:
            concepts = []
    finally:
        conn.close()
    sqlite_query_seconds = time.perf_counter() - sqlite_started
    materialization_started = time.perf_counter()

    concepts_by_dataset = {}
    for (
        row_source,
        dataset_id,
        dimension_id,
        dimension_name,
        code,
        label,
        semantic_text,
        metadata_json,
    ) in concepts:
        if source is not None and row_source != source.upper():
            continue
        concepts_by_dataset.setdefault(
            (row_source, dataset_id),
            [],
        ).append({
            "source": row_source,
            "dataset_id": dataset_id,
            "dimension_id": dimension_id,
            "dimension_name": dimension_name,
            "code": code,
            "label": label,
            "semantic_text": semantic_text,
            "metadata": json.loads(metadata_json or "{}"),
        })

    profiles = []
    for (
        row_source,
        dataset_id,
        structure_json,
        title,
        description,
        source_url,
        metadata_json,
        is_latest_release,
    ) in rows:
        structure = json.loads(structure_json)
        dataflow = structure.get("dataflow", {})
        raw_metadata = (
            json.loads(metadata_json)
            if metadata_json
            else dataflow.get("metadata", {})
        )
        profiles.append({
            "source": row_source,
            "dataset_id": dataset_id,
            "title": title or dataflow.get("name") or dataset_id,
            "description": (
                description
                or dataflow.get("description")
                or raw_metadata.get("description")
                or ""
            ),
            "source_url": source_url or dataflow.get("source_url"),
            "metadata": raw_metadata,
            "is_latest_release": (
                bool(is_latest_release)
                if is_latest_release is not None
                else None
            ),
            "structure": structure,
            "concepts": concepts_by_dataset.get(
                (row_source, dataset_id),
                [],
            ),
        })
    if instrumentation is not None:
        instrumentation["sqlite_profile_load_seconds"] = sqlite_query_seconds
        instrumentation["profile_materialization_seconds"] = (
            time.perf_counter() - materialization_started
        )
        instrumentation["profile_load_seconds"] = (
            time.perf_counter() - profile_started
        )
        instrumentation["dataset_structures_loaded"] = len(rows)
        instrumentation["semantic_concept_rows_loaded"] = len(concepts)
    return profiles


def initialize():

    conn = connect()

    schema_path = PROJECT_ROOT / "schema.sql"
    with schema_path.open("r", encoding="utf-8") as schema_file:
        conn.executescript(schema_file.read())

    ensure_document_tables(conn)
    ensure_literature_tables(conn)
    conn.commit()
    conn.close()


def create_conversation(title="Chat"):

    conn = connect()

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO conversations (title)
        VALUES (?)
        """,
        (title,)
    )

    conversation_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return conversation_id


def add_message(conversation_id, role, content):

    conn = connect()

    conn.execute(
        """
        INSERT INTO messages
        (conversation_id, role, content)
        VALUES (?, ?, ?)
        """,
        (conversation_id, role, content)
    )

    conn.commit()
    conn.close()





def save_message(role, content):

    conn = connect()

    conn.execute(
        """
        INSERT INTO messages
        (role, content)
        VALUES (?, ?)
        """,
        (role, content)
    )

    conn.commit()
    conn.close()

def get_recent_messages(limit=10):

    conn = connect()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT role, content
        FROM messages
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,)
    )

    rows = cursor.fetchall()

    conn.close()

    return rows[::-1]





def test():

    conn = connect()

    cursor = conn.cursor()

    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )

    print(cursor.fetchall())

    conn.close()


if __name__ == "__main__":

    initialize()

    test()




def save_document(file_name, file_path):

    conn = connect()
    ensure_document_tables(conn)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT OR IGNORE INTO documents
        (file_name, file_path)
        VALUES (?, ?)
        """,
        (file_name, file_path)
    )

    conn.commit()

    cursor.execute(
        """
        SELECT id
        FROM documents
        WHERE file_path = ?
        """,
        (file_path,)
    )

    document_id = cursor.fetchone()[0]

    conn.close()

    return document_id


def save_document_chunk(
    document_id,
    page_number,
    chunk_number,
    content,
    section=None,
    stable_chunk_id=None,
    metadata=None,
    content_hash=None,
):

    conn = connect()
    try:
        ensure_document_tables(conn)
        conn.execute(
            """
            INSERT INTO document_chunks
            (document_id, page_number, chunk_number, content, section,
             stable_chunk_id, metadata_json, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(document_id, stable_chunk_id)
            WHERE stable_chunk_id IS NOT NULL
            DO UPDATE SET
         page_number = excluded.page_number,
         chunk_number = excluded.chunk_number,
         content = excluded.content,
         section = excluded.section,
         metadata_json = excluded.metadata_json,
         content_hash = excluded.content_hash
            """,
            (
         document_id,
         page_number,
         chunk_number,
         content,
         section,
         stable_chunk_id,
         json.dumps(metadata or {}, ensure_ascii=False),
         content_hash,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def save_source_document(document):
    source = str(document.get("source", "")).upper()
    source_url = document.get("source_url")
    file_path = document.get("file_path")
    if not source or not source_url or not file_path:
        raise ValueError(
            "Source documents require source, source_url, and file_path"
        )

    conn = connect()
    try:
        ensure_document_tables(conn)
        row = conn.execute(
            """
            SELECT id, checksum
            FROM documents
            WHERE source = ? AND source_url = ?
            """,
            (source, source_url),
        ).fetchone()
        measure_ids_json = json.dumps(
            document.get("measure_ids") or [],
            ensure_ascii=False,
        )
        metadata_json = json.dumps(
            document.get("metadata") or {},
            ensure_ascii=False,
        )
        values = (
            document.get("file_name") or file_path.rsplit("\\", 1)[-1],
            file_path,
            source,
            document.get("title"),
            document.get("document_type"),
            source_url,
            document.get("published_at"),
            document.get("updated_at"),
            document.get("dataset_id"),
            measure_ids_json,
            document.get("checksum"),
            document.get("extraction_status", "indexed"),
            document.get("raw_text"),
            document.get("content_type"),
            metadata_json,
        )
        if row is None:
            cursor = conn.execute(
                """
                INSERT INTO documents (
                    file_name, file_path, source, title, document_type,
                    source_url, published_at, updated_at, dataset_id,
                    measure_ids_json, checksum, extraction_status, raw_text,
                    content_type, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            document_id = cursor.lastrowid
            changed = True
        else:
            document_id, previous_checksum = row
            changed = previous_checksum != document.get("checksum")
            if changed:
                conn.execute(
                    "DELETE FROM document_chunks WHERE document_id = ?",
                    (document_id,),
                )
            conn.execute(
                """
                UPDATE documents
                SET file_name = ?, file_path = ?, source = ?, title = ?,
                    document_type = ?, source_url = ?, published_at = ?,
                    updated_at = ?, dataset_id = ?, measure_ids_json = ?,
                    checksum = ?, extraction_status = ?, raw_text = ?,
                    content_type = ?, metadata_json = ?,
                    indexed_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                values + (document_id,),
            )

        dataset_id = document.get("dataset_id")
        if dataset_id:
            conn.execute(
                """
                INSERT INTO document_dataset_links (
                    document_id, source, dataset_id, measure_ids_json
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(document_id, source, dataset_id)
                DO UPDATE SET measure_ids_json = excluded.measure_ids_json
                """,
                (
                    document_id,
                    source,
                    dataset_id,
                    measure_ids_json,
                ),
            )
        conn.execute(
            """
            DELETE FROM document_ingestion_failures
            WHERE source = ? AND source_url = ?
            """,
            (source, source_url),
        )
        conn.commit()
        return document_id, changed
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_source_document(source, source_url):
    conn = connect()
    try:
        ensure_document_tables(conn)
        row = conn.execute(
            """
            SELECT id, file_name, file_path, checksum, raw_text, updated_at
            FROM documents
            WHERE source = ? AND source_url = ?
            """,
            (str(source).upper(), source_url),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {
        "document_id": row[0],
        "file_name": row[1],
        "file_path": row[2],
        "checksum": row[3],
        "raw_text": row[4],
        "updated_at": row[5],
    }


def save_document_ingestion_failure(
    source,
    source_url,
    dataset_id,
    stage,
    error,
):
    conn = connect()
    try:
        ensure_document_tables(conn)
        conn.execute(
            """
            INSERT INTO document_ingestion_failures (
                source, source_url, dataset_id, stage, error, failed_at
            )
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(source, source_url) DO UPDATE SET
                dataset_id = excluded.dataset_id,
                stage = excluded.stage,
                error = excluded.error,
                failed_at = CURRENT_TIMESTAMP
            """,
            (
                str(source).upper(),
                source_url,
                dataset_id,
                stage,
                str(error),
            ),
        )
        conn.execute(
            """
            UPDATE documents
            SET extraction_status = ?
            WHERE source = ? AND source_url = ?
            """,
            (
                f"failed:{stage}",
                str(source).upper(),
                source_url,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def count_source_documents(source=None):
    conn = connect()
    try:
        ensure_document_tables(conn)
        if source is None:
            documents = conn.execute(
         "SELECT COUNT(*) FROM documents WHERE source != 'LOCAL'"
            ).fetchone()[0]
            where_sql = "d.source != 'LOCAL'"
            params = ()
        else:
            source = str(source).upper()
            documents = conn.execute(
         "SELECT COUNT(*) FROM documents WHERE source = ?",
         (source,),
            ).fetchone()[0]
            where_sql = "d.source = ?"
            params = (source,)
        chunks = conn.execute(
            f"""
            SELECT COUNT(*),
            SUM(CASE WHEN c.embedding_json IS NOT NULL THEN 1 ELSE 0 END)
            FROM document_chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE {where_sql}
            """,
            params,
        ).fetchone()
    finally:
        conn.close()
    return {
        "documents": documents,
        "chunks": chunks[0] or 0,
        "embedded_chunks": chunks[1] or 0,
    }
