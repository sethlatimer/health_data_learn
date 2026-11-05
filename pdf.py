import sqlite3
from pathlib import Path

from database import save_document, save_document_chunk
from embedding import embed_document
from tools import chunk_text, read_document, read_pdf


def ask_about_pdf(ask, pdf_path, question):
    pdf_text = read_document(pdf_path)
    document_prompt = (
        "Answer the question using only the document below.\n\n"
        "If the answer is not present in the document, say so.\n\n"
        "DOCUMENT:\n\n"
        + pdf_text[:25000]
        + "\n\nQUESTION:\n\n"
        + question
    )
    return ask(document_prompt)


def index_pdf(pdf_path):
    pdf_path = Path(pdf_path).resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"File is not a PDF: {pdf_path}")

    pages = read_pdf(pdf_path)
    document_id = save_document(pdf_path.name, str(pdf_path))
    chunk_count = 0

    for page in pages:
        page_number = page["page_number"]
        page_text = page["text"]

        if not page_text.strip():
            continue

        chunks = chunk_text(page_text)
        for chunk_number, chunk in enumerate(chunks, start=1):
            save_document_chunk(
                document_id,
                page_number,
                chunk_number,
                chunk,
            )
            chunk_count += 1

    return {
        "document_id": document_id,
        "file_name": pdf_path.name,
        "pages": len(pages),
        "chunks": chunk_count,
    }


def index_folder(
    folder_path,
    embedding_model="nomic-local",
    recursive=False,
):
    folder_path = Path(folder_path).resolve()

    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    if not folder_path.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder_path}")

    pdf_files = sorted(
        folder_path.rglob("*.pdf")
        if recursive
        else folder_path.glob("*.pdf")
    )

    with sqlite3.connect("agent.db") as conn:
        existing_paths = {
            row[0]
            for row in conn.execute(
                "SELECT file_path FROM documents"
            ).fetchall()
        }

    report = {
        "folder": str(folder_path),
        "pdfs_found": len(pdf_files),
        "indexed": [],
        "skipped": [],
        "failed": [],
    }

    for pdf_path in pdf_files:
        resolved_path = str(pdf_path.resolve())

        if resolved_path in existing_paths:
            report["skipped"].append({
                "file_name": pdf_path.name,
                "reason": "Already indexed",
            })
            print(f"Skipped: {pdf_path.name} (already indexed)")
            continue

        try:
            print(f"\nIndexing: {pdf_path.name}")
            index_result = index_pdf(resolved_path)
            document_id = index_result["document_id"]

            print(f"Embedding: {pdf_path.name}")
            embedding_result = embed_document(
                document_id,
                embedding_model,
            )
            report["indexed"].append({
                "file_name": pdf_path.name,
                "document_id": document_id,
                "pages": index_result["pages"],
                "chunks": index_result["chunks"],
                "embedded": embedding_result["embedded"],
            })
            existing_paths.add(resolved_path)
            print(f"Finished: {pdf_path.name}")
        except Exception as error:
            report["failed"].append({
                "file_name": pdf_path.name,
                "error": str(error),
            })
            print(f"Failed: {pdf_path.name}")
            print(f"Reason: {error}")

    return report
