import json
import math
import time

import requests

from database import (
    connect,
    ensure_dataset_concepts_table,
    ensure_dataset_semantics_table,
    ensure_document_tables,
)


OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"


def embed_text(text, embedding_model="nomic-embed-text"):
    response = requests.post(
        OLLAMA_EMBED_URL,
        json={
            "model": embedding_model,
            "input": text,
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]


def embed_texts(texts, embedding_model="nomic-embed-text", batch_size=32):
    if batch_size < 1:
        raise ValueError("batch_size must be greater than zero")

    texts = list(texts)
    embeddings = []
    
    # Process texts in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # Make sequential requests for each text in the batch
        for text in batch:
            embedding = embed_text(text, embedding_model)
            embeddings.append(embedding)

    return embeddings


def generate_embedding(text, embedding_model="nomic-embed-text"):
    return embed_text(text, embedding_model)


def _cosine_similarity(vector_a, vector_b):
    if len(vector_a) != len(vector_b):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def search_dataset_concepts(
    query,
    dataset_id=None,
    top_k=10,
    embedding_model="nomic-embed-text",
    source=None,
    dataset_ids=None,
    instrumentation=None,
):
    if top_k < 1:
        raise ValueError("top_k must be greater than zero")
    if dataset_ids is not None:
        dataset_ids = list(dict.fromkeys(dataset_ids))
        if not dataset_ids:
            return []

    query_started = time.perf_counter()
    query_vector = embed_text(
        "search_query: " + query,
        embedding_model,
    )
    query_embedding_seconds = time.perf_counter() - query_started

    sql = """
        SELECT
            source,
            dataset_id,
            dimension_id,
            dimension_name,
            code,
            label,
            semantic_text,
            embedding_json,
            metadata_json
        FROM dataset_concepts
        WHERE concept_role = 'semantic'
          AND embedding_json IS NOT NULL
          AND dimension_id != '__dataset__'
    """
    params = ()
    if dataset_id is not None:
        sql += " AND dataset_id = ?"
        params = (dataset_id,)
    if source is not None:
        sql += " AND source = ?"
        params += (source.upper(),)
    if dataset_ids is not None:
        sql += " AND dataset_id IN (" + ",".join("?" for _ in dataset_ids) + ")"
        params += tuple(dataset_ids)

    load_started = time.perf_counter()
    conn = connect()
    try:
        ensure_dataset_concepts_table(conn)
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    vector_loading_seconds = time.perf_counter() - load_started

    results = []
    cosine_started = time.perf_counter()
    vectors_compared = 0
    for (
        source,
        row_dataset_id,
        dimension_id,
        dimension_name,
        code,
        label,
        semantic_text,
        embedding_json,
        metadata_json,
    ) in rows:
        try:
            concept_vector = json.loads(embedding_json)
            score = _cosine_similarity(query_vector, concept_vector)
            vectors_compared += 1
        except (TypeError, ValueError, json.JSONDecodeError):
            continue

        results.append({
            "scope": "structured_concepts",
            "dataset_id": row_dataset_id,
            "source": source,
            "dimension_id": dimension_id,
            "dimension_name": dimension_name,
            "code": code,
            "label": label,
            "semantic_text": semantic_text,
            "metadata": json.loads(metadata_json or "{}"),
            "score": score,
        })

    results.sort(key=lambda result: result["score"], reverse=True)
    cosine_ranking_seconds = time.perf_counter() - cosine_started
    returned = results[:top_k]
    if instrumentation is not None:
        instrumentation.update({
            "query_embedding_seconds": query_embedding_seconds,
            "vector_loading_seconds": vector_loading_seconds,
            "cosine_ranking_seconds": cosine_ranking_seconds,
            "vector_rows_loaded": len(rows),
            "vectors_compared": vectors_compared,
            "concepts_returned": len(returned),
        })
    return returned


def rank_dataset_candidates(
    query,
    embedding_model="nomic-embed-text",
    source=None,
    instrumentation=None,
):
    query_started = time.perf_counter()
    query_vector = embed_text(
        "search_query: " + query,
        embedding_model,
    )
    query_embedding_seconds = time.perf_counter() - query_started
    sql = """
        SELECT source, dataset_id, embedding_json
        FROM dataset_concepts
        WHERE concept_role = 'semantic'
          AND embedding_json IS NOT NULL
          AND embedding_model = ?
    """
    params = [embedding_model]
    if source is not None:
        sql += " AND source = ?"
        params.append(source.upper())

    load_started = time.perf_counter()
    conn = connect()
    try:
        ensure_dataset_concepts_table(conn)
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    vector_loading_seconds = time.perf_counter() - load_started

    scores_by_dataset = {}
    cosine_started = time.perf_counter()
    vectors_compared = 0
    for row_source, dataset_id, embedding_json in rows:
        try:
            vector = json.loads(embedding_json)
            score = _cosine_similarity(query_vector, vector)
            vectors_compared += 1
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        scores_by_dataset.setdefault(
            (row_source, dataset_id),
            [],
        ).append(score)

    candidates = []
    for (row_source, dataset_id), scores in scores_by_dataset.items():
        scores.sort(reverse=True)
        strongest_scores = scores[:3]
        candidates.append({
            "source": row_source,
            "dataset_id": dataset_id,
            "score": (
                0.7 * strongest_scores[0]
                + 0.3 * sum(strongest_scores) / len(strongest_scores)
            ),
            "best_concept_score": strongest_scores[0],
            "semantic_concepts_scored": len(scores),
        })
    candidates.sort(key=lambda item: item["score"], reverse=True)
    cosine_ranking_seconds = time.perf_counter() - cosine_started
    if instrumentation is not None:
        instrumentation.update({
            "query_embedding_seconds": query_embedding_seconds,
            "vector_loading_seconds": vector_loading_seconds,
            "cosine_ranking_seconds": cosine_ranking_seconds,
            "vector_rows_loaded": len(rows),
            "vectors_compared": vectors_compared,
            "candidates_returned": len(candidates),
        })
    return candidates


def find_dataset_filters(
    dataset_id,
    dimension_id=None,
    search=None,
    source=None,
):
    sql = """
        SELECT source, dataset_id, dimension_id, dimension_name, code, label
        FROM dataset_concepts
        WHERE dataset_id = ?
          AND concept_role = 'filter'
    """
    params = [dataset_id]

    if source is not None:
        sql += " AND source = ?"
        params.append(source.upper())
    if dimension_id is not None:
        sql += " AND dimension_id = ?"
        params.append(dimension_id)

    if search:
        sql += """
          AND (
              instr(lower(code), lower(?)) > 0
              OR instr(lower(label), lower(?)) > 0
          )
        """
        params.extend((search, search))

    sql += " ORDER BY dimension_id, code"

    conn = connect()
    try:
        ensure_dataset_concepts_table(conn)
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()

    return [
        {
            "source": row[0],
            "dataset_id": row[1],
            "dimension_id": row[2],
            "dimension_name": row[3],
            "code": row[4],
            "label": row[5],
        }
        for row in rows
    ]


def search_corpus(
    query,
    top_k=5,
    embedding_model="nomic-embed-text",
    include_oecd=True,
    scopes=None,
):
    allowed_scopes = {"documents", "dataset_summaries"}
    if scopes is not None:
        scopes = {scopes} if isinstance(scopes, str) else set(scopes)
        unknown_scopes = scopes - allowed_scopes
        if unknown_scopes:
            raise ValueError(
                f"Unknown corpus search scopes: {sorted(unknown_scopes)}"
            )
        include_documents = "documents" in scopes
        include_dataset_summaries = "dataset_summaries" in scopes
    else:
        include_documents = True
        include_dataset_summaries = include_oecd

    query_vector = generate_embedding(
        "search_query: " + query,
        embedding_model,
    )

    conn = connect()
    try:
        ensure_document_tables(conn)
        ensure_dataset_semantics_table(conn)
        rows = conn.execute(
            """
            SELECT
                c.id,
                c.document_id,
                c.page_number,
                c.chunk_number,
                c.content,
                c.embedding_json,
                d.file_name,
                d.file_path,
                d.source,
                d.title,
                d.document_type,
                d.source_url,
                d.dataset_id
            FROM document_chunks c
            JOIN documents d
                ON d.id = c.document_id
            WHERE c.embedding_json IS NOT NULL
            """
        ).fetchall() if include_documents else []
        dataset_rows = (
            conn.execute(
                """
                SELECT
                    dataset_id,
                    title,
                    semantic_text,
                    embedding_json
                FROM dataset_semantics
                """
            ).fetchall()
            if include_dataset_summaries
            else []
        )
    finally:
        conn.close()

    results = []

    for (
        chunk_id,
        document_id,
        page_number,
        chunk_number,
        content,
        embedding_json,
        file_name,
        file_path,
        source,
        title,
        document_type,
        source_url,
        dataset_id,
    ) in rows:
        try:
            chunk_vector = json.loads(embedding_json)
            score = _cosine_similarity(query_vector, chunk_vector)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue

        results.append({
            "scope": "documents",
            "chunk_id": chunk_id,
            "document_id": document_id,
            "source_type": "document",
            "file_name": file_name,
            "file_path": file_path,
            "source": source,
            "title": title or file_name,
            "document_type": document_type,
            "source_url": source_url,
            "dataset_id": dataset_id,
            "page_number": page_number,
            "chunk_number": chunk_number,
            "score": score,
            "content": content,
        })

    for dataset_id, title, semantic_text, embedding_json in dataset_rows:
        try:
            dataset_vector = json.loads(embedding_json)
            score = _cosine_similarity(query_vector, dataset_vector)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue

        results.append({
            "scope": "dataset_summaries",
            "source_type": "oecd_dataset",
            "dataset_id": dataset_id,
            "file_name": title,
            "file_path": "",
            "page_number": None,
            "chunk_number": None,
            "score": score,
            "content": semantic_text,
        })

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def search_document(
    document_id,
    query,
    top_k=5,
    embedding_model="nomic-embed-text",
):
    query_vector = generate_embedding(
        "search_query: " + query,
        embedding_model,
    )

    conn = connect()
    try:
        ensure_document_tables(conn)
        rows = conn.execute(
            """
            SELECT
                c.id,
                c.page_number,
                c.chunk_number,
                c.content,
                c.embedding_json,
                d.file_name
            FROM document_chunks c
            JOIN documents d
                ON d.id = c.document_id
            WHERE c.document_id = ?
              AND c.embedding_json IS NOT NULL
            """,
            (document_id,),
        ).fetchall()
    finally:
        conn.close()

    results = []

    for (
        chunk_id,
        page_number,
        chunk_number,
        content,
        embedding_json,
        file_name,
    ) in rows:
        try:
            chunk_vector = json.loads(embedding_json)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue

        score = _cosine_similarity(query_vector, chunk_vector)
        results.append({
            "scope": "documents",
            "chunk_id": chunk_id,
            "file_name": file_name,
            "page_number": page_number,
            "chunk_number": chunk_number,
            "score": score,
            "content": content,
        })

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def embed_document(document_id, embedding_model="nomic-embed-text"):
    conn = connect()
    ensure_document_tables(conn)

    rows = conn.execute(
        """
        SELECT id, content
        FROM document_chunks
        WHERE document_id = ?
          AND embedding_json IS NULL
        ORDER BY id
        """,
        (document_id,),
    ).fetchall()

    if not rows:
        conn.close()
        return {
            "document_id": document_id,
            "embedded": 0,
            "message": "No chunks need embeddings.",
        }

    embedded_count = 0

    try:
        for chunk_id, content in rows:
            embedding = generate_embedding(
                "search_document: " + content,
                embedding_model,
            )
            conn.execute(
                """
                UPDATE document_chunks
                SET embedding_json = ?
                WHERE id = ?
                """,
                (json.dumps(embedding), chunk_id),
            )
            conn.commit()
            embedded_count += 1
            print(f"Embedded chunk {embedded_count} of {len(rows)}")
    finally:
        conn.close()

    return {
        "document_id": document_id,
        "embedded": embedded_count,
        "embedding_model": embedding_model,
    }


def search_source_documents(
    query,
    top_k=5,
    source=None,
    dataset_id=None,
    document_id=None,
    measure_id=None,
    embedding_model="nomic-embed-text",
):
    if top_k < 1:
        raise ValueError("top_k must be greater than zero")
    query_vector = generate_embedding(
        "search_query: " + query,
        embedding_model,
    )
    sql = """
        SELECT
            c.id, c.document_id, c.page_number, c.chunk_number, c.section,
            c.content, c.embedding_json, d.source, d.file_name, d.file_path,
            d.title, d.document_type, d.source_url, d.dataset_id,
            d.measure_ids_json, d.published_at, c.metadata_json
        FROM document_chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.embedding_json IS NOT NULL
          AND d.source != 'LOCAL'
    """
    params = []
    if source is not None:
        sql += " AND d.source = ?"
        params.append(str(source).upper())
    if dataset_id is not None:
        sql += """
            AND EXISTS (
                SELECT 1 FROM document_dataset_links l
                WHERE l.document_id = d.id AND l.dataset_id = ?
            )
        """
        params.append(dataset_id)
    if document_id is not None:
        sql += " AND d.id = ?"
        params.append(document_id)
    if measure_id is not None:
        sql += """
            AND EXISTS (
                SELECT 1
                FROM document_dataset_links l, json_each(l.measure_ids_json) m
                WHERE l.document_id = d.id AND m.value = ?
            )
        """
        params.append(measure_id)

    conn = connect()
    try:
        ensure_document_tables(conn)
        rows = conn.execute(sql, params).fetchall()
        document_ids = sorted({row[1] for row in rows})
        link_rows = []
        if document_ids:
            placeholders = ",".join("?" for _ in document_ids)
            link_rows = conn.execute(
                f"""
                SELECT document_id, source, dataset_id, measure_ids_json
                FROM document_dataset_links
                WHERE document_id IN ({placeholders})
                ORDER BY source, dataset_id
                """,
                document_ids,
            ).fetchall()
    finally:
        conn.close()

    links_by_document = {}
    for row_document_id, row_source, linked_dataset, linked_measures in link_rows:
        links_by_document.setdefault(row_document_id, []).append({
            "source": row_source,
            "dataset_id": linked_dataset,
            "measure_ids": json.loads(linked_measures or "[]"),
        })

    results = []
    for row in rows:
        (
            chunk_id, row_document_id, page_number, chunk_number, section,
            content, embedding_json, row_source, file_name, file_path,
            title, document_type, source_url, row_dataset_id,
            measure_ids_json, published_at, metadata_json,
        ) = row
        try:
            score = _cosine_similarity(
                query_vector,
                json.loads(embedding_json),
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        try:
            measure_ids = json.loads(measure_ids_json or "[]")
        except (TypeError, ValueError, json.JSONDecodeError):
            measure_ids = []
        related_datasets = links_by_document.get(row_document_id, [])
        if related_datasets:
            measure_ids = list(dict.fromkeys(
                measure
                for relation in related_datasets
                for measure in relation["measure_ids"]
            ))
        dataset_ids = list(dict.fromkeys(
            relation["dataset_id"]
            for relation in related_datasets
        ))
        results.append({
            "scope": "documents",
            "chunk_id": chunk_id,
            "document_id": row_document_id,
            "source": row_source,
            "dataset_id": (
                dataset_id
                if dataset_id is not None
                else (dataset_ids[0] if dataset_ids else row_dataset_id)
            ),
            "dataset_ids": dataset_ids or (
                [row_dataset_id] if row_dataset_id else []
            ),
            "measure_ids": measure_ids,
            "title": title or file_name,
            "document_type": document_type,
            "source_url": source_url,
            "local_path": file_path,
            "file_name": file_name,
            "page_number": page_number,
            "section": section,
            "chunk_number": chunk_number,
            "score": score,
            "content": content,
            "metadata": json.loads(metadata_json or "{}"),
            "published_at": published_at,
        })
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    primary_care_id = "OECD.ELS.HD,DSD_HCQO@DF_PC,2.1"
    test_queries = (
        "avoidable hospital admissions",
        "diabetes admissions",
        "asthma hospitalization",
    )
    for query in test_queries:
        print(f"\nQuery: {query}")
        matches = search_dataset_concepts(
            query,
            dataset_id=primary_care_id,
            top_k=5,
        )
        for match in matches:
            print(
                f"  {match['dimension_id']}:{match['code']} "
                f"{match['label']} — {match['score']:.4f}"
            )
