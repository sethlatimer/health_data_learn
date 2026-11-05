import hashlib
import html
from html.parser import HTMLParser
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests

import database
import embedding
from tools import chunk_text, read_pdf


MAX_DOCUMENT_BYTES = 25 * 1024 * 1024
MAX_DOCUMENT_CHARS = 2_000_000
MAX_PDF_PAGES = 200
CHUNK_SIZE = 1800
DEFAULT_INGEST_LIMIT = 20
_SYSTEM_TRUST_INITIALIZED = False
BOILERPLATE_PATTERNS = (
    re.compile(r"^\s*all rights reserved\.?\s*$", re.I),
    re.compile(r"^\s*copyright\b.*$", re.I),
    re.compile(r"^\s*privacy policy\s*$", re.I),
    re.compile(r"^\s*terms of use\s*$", re.I),
)


@dataclass
class DocumentReference:
    source: str
    title: str
    document_type: str
    source_url: str
    dataset_id: str | None = None
    measure_ids: list[str] = field(default_factory=list)
    published_at: str | None = None
    updated_at: str | None = None
    metadata: dict = field(default_factory=dict)
    inline_text: str | None = None
    content_type: str | None = None

    def to_dict(self):
        return asdict(self)


class DocumentSource:
    source = None
    official_hosts = ()

    def discover_documents(self, profiles=None, dataset_id=None):
        raise NotImplementedError

    def validate_url(self, url):
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").casefold()
        return (
            parsed.scheme == "https"
            and any(
                hostname == host or hostname.endswith("." + host)
                for host in self.official_hosts
            )
        )


class _HTMLTextParser(HTMLParser):
    _ignored = {"script", "style", "nav", "header", "footer", "noscript"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.headings = []
        self._ignored_depth = 0
        self._heading_depth = 0
        self._heading_text = []

    def handle_starttag(self, tag, attrs):
        tag = tag.casefold()
        if tag in self._ignored:
            self._ignored_depth += 1
        if tag in {"h1", "h2", "h3", "h4"}:
            self._heading_depth += 1
            self._heading_text = []
        if tag in {"p", "li", "br", "section", "article"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        tag = tag.casefold()
        if tag in self._ignored and self._ignored_depth:
            self._ignored_depth -= 1
        if tag in {"h1", "h2", "h3", "h4"} and self._heading_depth:
            heading = " ".join("".join(self._heading_text).split())
            if heading:
                self.headings.append(heading)
                self.parts.extend(("\n[SECTION] ", heading, "\n"))
            self._heading_depth -= 1
            self._heading_text = []
        if tag in {"p", "li", "br", "section", "article"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._ignored_depth:
            return
        value = " ".join(data.split())
        if value:
            if self._heading_depth:
                self._heading_text.append(value)
            else:
                self.parts.append(value)
                self.parts.append(" ")


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _initialize_system_trust():
    global _SYSTEM_TRUST_INITIALIZED
    if _SYSTEM_TRUST_INITIALIZED:
        return
    try:
        import truststore
    except ImportError:
        return
    truststore.inject_into_ssl()
    _SYSTEM_TRUST_INITIALIZED = True


def _clean_text(text):
    cleaned = []
    seen = set()
    for line in text.splitlines():
        line = " ".join(line.split()).strip()
        if not line:
            if cleaned and cleaned[-1]:
                cleaned.append("")
            continue
        if any(pattern.match(line) for pattern in BOILERPLATE_PATTERNS):
            continue
        if line.casefold() in seen:
            continue
        seen.add(line.casefold())
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def _json_text(payload, prefix=""):
    lines = []

    def visit(value, path):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{path}.{key}" if path else str(key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")
        elif value is not None:
            rendered = str(value).strip()
            if rendered:
                lines.append(f"{path}: {rendered}")

    visit(payload, prefix)
    return "\n".join(lines)


def _paragraph_chunks(text, page_number=None, default_section=None):
    chunks = []
    active_section = default_section
    section_pattern = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?[A-Z][A-Z0-9 ,:/()&-]{3,90}\s*$")
    paragraphs = []
    current_paragraph = []
    for line in text.splitlines():
        line = " ".join(line.split()).strip()
        if not line:
            if current_paragraph:
                paragraphs.append(" ".join(current_paragraph))
                current_paragraph = []
            continue
        if line.startswith("[SECTION] ") or section_pattern.match(line):
            if current_paragraph:
                paragraphs.append(" ".join(current_paragraph))
                current_paragraph = []
            paragraphs.append(
                line if line.startswith("[SECTION] ")
                else f"[SECTION] {line}"
            )
        else:
            current_paragraph.append(line)
    if current_paragraph:
        paragraphs.append(" ".join(current_paragraph))
    current = []
    current_length = 0

    def emit():
        nonlocal current, current_length
        if not current:
            return
        content = "\n\n".join(current).strip()
        if len(content) > CHUNK_SIZE:
            parts = chunk_text(content, chunk_size=CHUNK_SIZE, overlap=200)
        else:
            parts = [content]
        for part in parts:
            chunks.append({
                "page_number": page_number,
                "section": active_section,
                "content": part,
            })
        current = []
        current_length = 0

    paragraph_iterator = iter(paragraphs)
    for paragraph in paragraph_iterator:
        if paragraph == "[SECTION]":
            emit()
            active_section = next(paragraph_iterator, None)
            continue
        if paragraph.startswith("[SECTION] "):
            emit()
            active_section = paragraph[len("[SECTION] "):]
            continue
        if section_pattern.match(paragraph) and len(paragraph) <= 90:
            emit()
            active_section = paragraph
            continue
        if current and current_length + len(paragraph) > CHUNK_SIZE:
            emit()
        current.append(paragraph)
        current_length += len(paragraph) + 2
    emit()
    return chunks


def _extract(path, content_type):
    path = Path(path)
    suffix = path.suffix.casefold()
    if suffix == ".pdf" or "pdf" in (content_type or "").casefold():
        pages = read_pdf(path, max_pages=MAX_PDF_PAGES)
        page_texts = []
        remaining_characters = MAX_DOCUMENT_CHARS
        for page in pages:
            page_text = _clean_text(page.get("text", ""))
            if not page_text or remaining_characters <= 0:
                continue
            page_text = page_text[:remaining_characters]
            remaining_characters -= len(page_text)
            page_texts.append({
                "page_number": page["page_number"],
                "text": page_text,
            })
        raw_text = "\n\n".join(
            f"[Page {page['page_number']}]\n{page['text']}"
            for page in page_texts
        )
        chunks = []
        for page in page_texts:
            chunks.extend(_paragraph_chunks(page["text"], page["page_number"]))
        return raw_text, chunks

    raw_bytes = path.read_bytes()
    if len(raw_bytes) > MAX_DOCUMENT_BYTES:
        raise ValueError(f"Document exceeds {MAX_DOCUMENT_BYTES} byte limit")
    text = raw_bytes.decode("utf-8-sig", errors="replace")
    if suffix in {".html", ".htm"} or "html" in (content_type or "").casefold():
        parser = _HTMLTextParser()
        parser.feed(text)
        text = "\n".join(parser.parts)
    elif suffix == ".json" or "json" in (content_type or "").casefold():
        try:
            text = _json_text(json.loads(text))
        except json.JSONDecodeError as error:
            raise ValueError("Document JSON could not be parsed") from error
    text = _clean_text(html.unescape(text))[:MAX_DOCUMENT_CHARS]
    return text, _paragraph_chunks(text)


def _file_extension(content_type, url):
    suffix = Path(urlparse(url).path).suffix.casefold()
    if suffix in {".pdf", ".html", ".htm", ".json", ".txt", ".xml"}:
        return suffix
    lowered = (content_type or "").casefold()
    if "pdf" in lowered:
        return ".pdf"
    if "html" in lowered:
        return ".html"
    if "json" in lowered:
        return ".json"
    if "text" in lowered or "xml" in lowered:
        return ".txt"
    raise ValueError(f"Unsupported document content type: {content_type!r}")


class DocumentIngestionService:
    def __init__(
        self,
        sources=None,
        cache_root=r"C:\Temp\public_documents",
        embedding_model="nomic-local",
    ):
        if sources is None:
            from data_sources.documents.cms import CMSDocumentSource
            from data_sources.documents.oecd import OECDDocumentSource
            from data_sources.documents.who import WHODocumentSource

            sources = {
                "OECD": OECDDocumentSource(),
                "CMS": CMSDocumentSource(),
                "WHO": WHODocumentSource(),
            }
        self.sources = {
            key.upper(): value
            for key, value in (sources or {}).items()
        }
        self.cache_root = Path(cache_root).resolve()
        self.embedding_model = embedding_model

    def discover(self, source=None, dataset_id=None, limit=None):
        selected_sources = (
            {str(source).upper(): self.sources[str(source).upper()]}
            if source is not None
            else self.sources
        )
        profiles = database.get_indexed_dataset_profiles(
            include_concepts=False,
        )
        results = []
        for source_name, adapter in selected_sources.items():
            source_profiles = [
                profile for profile in profiles
                if profile["source"].upper() == source_name
                and (
                    dataset_id is None
                    or profile["dataset_id"] == dataset_id
                )
            ]
            results.extend(
                adapter.discover_documents(
                    profiles=source_profiles,
                    dataset_id=dataset_id,
                )
            )
        if limit is not None:
            if not isinstance(limit, int) or limit < 1:
                raise ValueError("limit must be a positive integer")
            return results[:limit]
        return results

    def ingest(
        self,
        source=None,
        dataset_id=None,
        limit=DEFAULT_INGEST_LIMIT,
        embed=True,
    ):
        references = self.discover(source, dataset_id, limit)
        report = {
            "discovered": len(references),
            "ingested": [],
            "skipped": [],
            "failures": [],
        }
        for reference in references:
            source_name = reference.source.upper()
            stage = "validate"
            try:
                adapter = self.sources[source_name]
                if not adapter.validate_url(reference.source_url):
                    raise ValueError(
                        f"URL is not an allowed official {source_name} host"
                    )
                stage = "download"
                document, was_cached = self._get_document(reference, adapter)
                stage = "extract"
                raw_text, chunks = self._extract_document(
                    document["path"],
                    document["content_type"],
                )
                if not raw_text:
                    raise ValueError("Document contained no usable text")
                document_checksum = _sha256(raw_text.encode("utf-8"))
                stage = "persist"
                document_id, changed = database.save_source_document({
                    **reference.to_dict(),
                    "file_name": document["path"].name,
                    "file_path": str(document["path"]),
                    "checksum": document_checksum,
                    "raw_text": raw_text,
                    "content_type": document["content_type"],
                    "extraction_status": "extracted",
                })
                for chunk_number, chunk in enumerate(chunks, start=1):
                    content_hash = _sha256(
                        chunk["content"].encode("utf-8")
                    )
                    stable_id = _sha256(
                        (
                            f"{reference.source}\n{reference.source_url}\n"
                            f"{chunk.get('page_number')}\n"
                            f"{chunk.get('section')}\n{content_hash}"
                        ).encode("utf-8")
                    )
                    database.save_document_chunk(
                        document_id,
                        chunk.get("page_number"),
                        chunk_number,
                        chunk["content"],
                        section=chunk.get("section"),
                        stable_chunk_id=stable_id,
                        metadata={
                            "source": source_name,
                            "title": reference.title,
                            "source_url": reference.source_url,
                            "dataset_id": reference.dataset_id,
                            "measure_ids": reference.measure_ids,
                        },
                        content_hash=content_hash,
                    )
                stage = "embedding"
                embedding_result = (
                    embedding.embed_document(
                        document_id,
                        self.embedding_model,
                    )
                    if embed
                    else {"embedded": 0}
                )
                item = {
                    "source": source_name,
                    "dataset_id": reference.dataset_id,
                    "document_id": document_id,
                    "title": reference.title,
                    "source_url": reference.source_url,
                    "chunks": len(chunks),
                    "embedded": embedding_result["embedded"],
                    "cached": was_cached,
                    "content_changed": changed,
                }
                if was_cached and not changed and not embedding_result["embedded"]:
                    report["skipped"].append(item)
                else:
                    report["ingested"].append(item)
            except Exception as error:
                database.save_document_ingestion_failure(
                    source_name,
                    reference.source_url,
                    reference.dataset_id,
                    stage,
                    error,
                )
                report["failures"].append({
                    "source": source_name,
                    "dataset_id": reference.dataset_id,
                    "source_url": reference.source_url,
                    "stage": stage,
                    "error": str(error),
                })
        return report

    def search(
        self,
        query,
        top_k=5,
        source=None,
        dataset_id=None,
        document_id=None,
        measure_id=None,
    ):
        return embedding.search_source_documents(
            query,
            top_k=top_k,
            source=source,
            dataset_id=dataset_id,
            document_id=document_id,
            measure_id=measure_id,
            embedding_model=self.embedding_model,
        )

    def _get_document(self, reference, adapter):
        existing = database.get_source_document(
            reference.source,
            reference.source_url,
        )
        cache_is_current = (
            existing is not None
            and existing.get("updated_at") == reference.updated_at
        )
        if cache_is_current and Path(existing["file_path"]).is_file():
            path = Path(existing["file_path"])
            if reference.inline_text is not None:
                existing_text = existing.get("raw_text")
                if existing_text == _clean_text(reference.inline_text):
                    content_type = reference.content_type or _guess_content_type(path)
                    return {"path": path, "content_type": content_type}, True
            else:
                content_type = reference.content_type or _guess_content_type(path)
                return {"path": path, "content_type": content_type}, True

        source_directory = self.cache_root / reference.source.casefold()
        source_directory.mkdir(parents=True, exist_ok=True)
        if reference.inline_text is not None:
            payload = reference.inline_text.encode("utf-8")
            content_type = reference.content_type or "text/plain"
            extension = _file_extension(content_type, reference.source_url)
        else:
            _initialize_system_trust()
            request_url = reference.source_url
            for redirect_number in range(6):
                response = requests.get(
                    request_url,
                    timeout=(10, 60),
                    stream=True,
                    allow_redirects=False,
                    headers={
                        "Accept": (
                            "application/pdf, text/html, application/json, "
                            "text/plain"
                        ),
                    },
                )
                if 300 <= response.status_code < 400:
                    location = response.headers.get("Location")
                    if not location:
                        raise requests.HTTPError(
                            "Document endpoint redirected without a Location header",
                            response=response,
                        )
                    next_url = urljoin(request_url, location)
                    if not adapter.validate_url(next_url):
                        raise ValueError(
                            "Document redirect left the source's official domains"
                        )
                    if redirect_number == 5:
                        raise requests.HTTPError(
                            "Document endpoint exceeded the redirect limit",
                            response=response,
                        )
                    request_url = next_url
                    continue
                response.raise_for_status()
                break
            content_type = response.headers.get("Content-Type", "").split(";")[0]
            declared_size = response.headers.get("Content-Length")
            if declared_size and int(declared_size) > MAX_DOCUMENT_BYTES:
                raise ValueError(
                    f"Document exceeds {MAX_DOCUMENT_BYTES} byte limit"
                )
            extension = _file_extension(content_type, reference.source_url)
            content_parts = []
            total = 0
            for part in response.iter_content(chunk_size=64 * 1024):
                if not part:
                    continue
                total += len(part)
                if total > MAX_DOCUMENT_BYTES:
                    raise ValueError(
                        f"Document exceeds {MAX_DOCUMENT_BYTES} byte limit"
                    )
                content_parts.append(part)
            payload = b"".join(content_parts)
        if not payload:
            raise ValueError("Downloaded document is empty")
        url_key = _sha256(reference.source_url.encode("utf-8"))[:24]
        path = source_directory / f"{url_key}{extension}"
        path.write_bytes(payload)
        return {"path": path, "content_type": content_type}, False

    @staticmethod
    def _extract_document(path, content_type):
        return _extract(path, content_type)


def _guess_content_type(path):
    return {
        ".pdf": "application/pdf",
        ".html": "text/html",
        ".htm": "text/html",
        ".json": "application/json",
        ".txt": "text/plain",
        ".xml": "text/xml",
    }.get(path.suffix.casefold(), "text/plain")


def search_documents(
    query,
    top_k=5,
    source=None,
    dataset_id=None,
    document_id=None,
    measure_id=None,
    embedding_model="nomic-local",
):
    return embedding.search_source_documents(
        query,
        top_k=top_k,
        source=source,
        dataset_id=dataset_id,
        document_id=document_id,
        measure_id=measure_id,
        embedding_model=embedding_model,
    )
