from pathlib import Path
from pypdf import PdfReader


ALLOWED_ROOT = Path(r"C:\Temp").resolve()

TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".sql",
    ".py",
    ".r",
    ".R",
    ".json",
    ".csv",
    ".yaml",
    ".yml"
}


def safe_path(path):
    """
    Resolve a path and ensure it stays inside ALLOWED_ROOT.
    """

    resolved_path = Path(path).expanduser().resolve()

    if resolved_path != ALLOWED_ROOT:
        if ALLOWED_ROOT not in resolved_path.parents:
            raise PermissionError(
                f"Path must be inside {ALLOWED_ROOT}"
            )

    return resolved_path


def list_files(folder, recursive=False):
    """
    List files in an approved folder.

    Returns file paths, sizes, and extensions.
    """

    folder_path = safe_path(folder)

    if not folder_path.exists():
        raise FileNotFoundError(
            f"Folder not found: {folder_path}"
        )

    if not folder_path.is_dir():
        raise NotADirectoryError(
            f"Not a folder: {folder_path}"
        )

    iterator = (
        folder_path.rglob("*")
        if recursive
        else folder_path.iterdir()
    )

    results = []

    for item in iterator:

        if item.is_file():

            results.append({
                "name": item.name,
                "path": str(item),
                "extension": item.suffix.lower(),
                "size_bytes": item.stat().st_size
            })

    return results


def read_text_file(path, max_characters=100000):
    """
    Read a UTF-8 text file from the approved folder.
    """

    file_path = safe_path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"Not a file: {file_path}"
        )

    if file_path.suffix not in TEXT_EXTENSIONS:
        raise ValueError(
            f"Unsupported text file type: {file_path.suffix}"
        )

    text = file_path.read_text(
        encoding="utf-8",
        errors="replace"
    )

    return text[:max_characters]


def read_pdf(path, max_pages=None):
    """
    Extract text from a PDF in the approved folder.
    """

    file_path = safe_path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {file_path}"
        )

    if file_path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Not a PDF: {file_path}"
        )

    reader = PdfReader(str(file_path))

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        if max_pages and page_number > max_pages:
            break

        page_text = page.extract_text() or ""

        pages.append({
            "page_number": page_number,
            "text": page_text
        })

    return pages



def chunk_text(
    text,
    chunk_size=1800,
    overlap=250
):
    """
    Split text into overlapping character-based chunks.
    """

    if chunk_size <= overlap:
        raise ValueError(
            "chunk_size must be larger than overlap"
        )

    chunks = []
    start = 0

    while start < len(text):

        end = min(
            start + chunk_size,
            len(text)
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def read_document(path):
    """
    Route supported files to the appropriate reader.
    """

    file_path = safe_path(path)
    extension = file_path.suffix.lower()

    if extension == ".pdf":

        pages = read_pdf(file_path)

        return "\n\n".join(
            f"[Page {page['page_number']}]\n{page['text']}"
            for page in pages
        )

    if extension in TEXT_EXTENSIONS:
        return read_text_file(file_path)

    raise ValueError(
        f"Unsupported document type: {extension}"
    )




def chunk_text(text, chunk_size=1500, overlap=200):

    chunks = []

    start = 0

    while start < len(text):

        end = min(
            start + chunk_size,
            len(text)
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks






