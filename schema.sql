CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    started_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER,
    role TEXT,
    content TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER,
    tool_name TEXT,
    arguments_json TEXT,
    result_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT,
    summary TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name TEXT NOT NULL,
    file_path TEXT NOT NULL UNIQUE,
    summary TEXT,
    indexed_at TEXT DEFAULT CURRENT_TIMESTAMP,
    source TEXT NOT NULL DEFAULT 'LOCAL',
    title TEXT,
    document_type TEXT,
    source_url TEXT,
    published_at TEXT,
    updated_at TEXT,
    dataset_id TEXT,
    measure_ids_json TEXT,
    checksum TEXT,
    extraction_status TEXT,
    raw_text TEXT,
    content_type TEXT,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS document_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    page_number INTEGER,
    chunk_number INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding_json TEXT,
    section TEXT,
    stable_chunk_id TEXT,
    metadata_json TEXT,
    content_hash TEXT,
    FOREIGN KEY (document_id)
        REFERENCES documents(id)
);

CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id
ON document_chunks(document_id);

CREATE TABLE IF NOT EXISTS document_dataset_links (
    document_id INTEGER NOT NULL,
    source TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    measure_ids_json TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (document_id, source, dataset_id),
    FOREIGN KEY (document_id) REFERENCES documents(id)
);

CREATE TABLE IF NOT EXISTS document_ingestion_failures (
    source TEXT NOT NULL,
    source_url TEXT NOT NULL,
    dataset_id TEXT,
    stage TEXT NOT NULL,
    error TEXT NOT NULL,
    failed_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (source, source_url)
);

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
);

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
);

CREATE TABLE IF NOT EXISTS dataset_structures (
    dataset_id TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'OECD',
    structure_json TEXT NOT NULL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (source, dataset_id)
);

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
);

CREATE TABLE IF NOT EXISTS source_catalog_cache (
    source TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    etag TEXT,
    last_modified TEXT
);
