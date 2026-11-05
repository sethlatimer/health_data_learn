from html.parser import HTMLParser
from urllib.parse import urlparse

from data_sources.documents.base import DocumentReference, DocumentSource


_DOCUMENT_KEYS = (
    "documentation",
    "methodology",
    "technical",
    "reference",
    "userguide",
    "user_guide",
    "definitionurl",
    "landingpage",
)
_DOCUMENT_LABELS = (
    "documentation",
    "methodology",
    "technical",
    "definition",
    "metadata",
    "data source",
    "data-source",
    "calculation",
    "comparability",
    "limitation",
    "manual",
    "guideline",
    "indicator",
    "document",
)


class _DescriptionLinkParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links = []
        self._href = None
        self._label = []

    def handle_starttag(self, tag, attrs):
        if tag.casefold() == "a":
            attributes = dict(attrs)
            self._href = attributes.get("href")
            self._label = []

    def handle_data(self, data):
        if self._href is not None:
            self._label.append(data)

    def handle_endtag(self, tag):
        if tag.casefold() == "a" and self._href is not None:
            label = " ".join(" ".join(self._label).split())
            self.links.append((label, self._href))
            self._href = None
            self._label = []


def _description_links(description):
    if not description:
        return []
    parser = _DescriptionLinkParser()
    parser.feed(description)
    return parser.links


def _is_document_link(label, url):
    label = label.casefold()
    path = urlparse(url).path.casefold()
    suffix = path.rsplit("/", 1)[-1].rsplit(".", 1)
    extension = "." + suffix[-1] if len(suffix) == 2 else ""
    supported_extensions = {".pdf", ".html", ".htm", ".txt", ".json"}
    if extension and extension not in supported_extensions:
        return False
    document_extension = extension in supported_extensions
    return (
        any(token in label for token in _DOCUMENT_LABELS)
        or (
            document_extension
            and any(
                token in path
                for token in (
                    "method",
                    "technical",
                    "definition",
                    "metadata",
                    "documentation",
                    "guide",
                    "quality",
                    "source",
                )
            )
        )
    )


def _walk(value, parent_key=""):
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{parent_key}.{key}" if parent_key else str(key)
            yield from _walk(item, path)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item, parent_key)
    elif isinstance(value, str) and value.startswith("https://"):
        yield parent_key, value


class OECDDocumentSource(DocumentSource):
    source = "OECD"
    official_hosts = ("oecd.org",)

    def discover_documents(self, profiles=None, dataset_id=None):
        references = []
        seen = set()
        for profile in profiles or []:
            if dataset_id and profile["dataset_id"] != dataset_id:
                continue
            candidates = []
            for metadata in (
                profile.get("metadata") or {},
                profile.get("structure") or {},
            ):
                for key, url in _walk(metadata):
                    key_normalized = "".join(
                        character for character in key.casefold()
                        if character.isalnum() or character == "_"
                    )
                    parsed = urlparse(url)
                    if (
                        not any(token in key_normalized for token in _DOCUMENT_KEYS)
                        or not self.validate_url(url)
                        or "api" in parsed.path.casefold()
                        or "/sdmx-json/data/" in parsed.path.casefold()
                        or url in seen
                    ):
                        continue
                    candidates.append((
                        url,
                        "methodology",
                        {"catalog_field": key},
                    ))
            for description in (
                profile.get("description"),
                (profile.get("structure") or {}).get(
                    "dataflow", {}
                ).get("description"),
            ):
                for label, url in _description_links(description):
                    if (
                        self.validate_url(url)
                        and _is_document_link(label, url)
                    ):
                        document_type = (
                            "technical_documentation"
                            if any(
                                word in label.casefold()
                                for word in ("documentation", "technical")
                            )
                            else "methodology"
                        )
                        candidates.append((
                            url,
                            document_type,
                            {
                                "catalog_field": "dataflow.description",
                                "link_label": label,
                            },
                        ))
            for url, document_type, metadata in candidates:
                if url in seen:
                    continue
                seen.add(url)
                references.append(DocumentReference(
                    source=self.source,
                    title=f"{profile['title']} documentation",
                    document_type=document_type,
                    source_url=url,
                    dataset_id=profile["dataset_id"],
                    metadata=metadata,
                ))
        return references
