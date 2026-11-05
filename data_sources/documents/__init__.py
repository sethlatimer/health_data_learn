from data_sources.documents.base import (
    DocumentIngestionService,
    DocumentReference,
    DocumentSource,
    search_documents,
)
from data_sources.documents.cms import CMSDocumentSource
from data_sources.documents.oecd import OECDDocumentSource
from data_sources.documents.who import WHODocumentSource


def default_document_sources():
    return {
        "OECD": OECDDocumentSource(),
        "CMS": CMSDocumentSource(),
        "WHO": WHODocumentSource(),
    }


__all__ = [
    "CMSDocumentSource",
    "DocumentIngestionService",
    "DocumentReference",
    "DocumentSource",
    "OECDDocumentSource",
    "WHODocumentSource",
    "default_document_sources",
    "search_documents",
]
