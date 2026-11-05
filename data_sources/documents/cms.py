from data_sources.cms import _is_filter_field
from data_sources.documents.base import DocumentReference, DocumentSource


class CMSDocumentSource(DocumentSource):
    source = "CMS"
    official_hosts = ("cms.gov",)

    def discover_documents(self, profiles=None, dataset_id=None):
        references = []
        seen = set()
        for profile in profiles or []:
            if dataset_id and profile["dataset_id"] != dataset_id:
                continue
            raw = profile.get("metadata") or {}
            dimensions = (
                profile.get("structure", {})
                .get("data_structure", {})
                .get("dimensions", [])
            )
            measure_ids = [
                dimension["id"]
                for dimension in dimensions
                if dimension.get("type") == "MeasureDimension"
                and dimension.get("id")
                and not _is_filter_field(dimension.get("metadata") or {})
            ]
            for distribution in raw.get("distribution", []) or []:
                if not isinstance(distribution, dict):
                    continue
                described_by = distribution.get("describedBy")
                if isinstance(described_by, dict):
                    url = (
                        described_by.get("downloadURL")
                        or described_by.get("@id")
                    )
                    if (
                        url
                        and self.validate_url(url)
                        and url not in seen
                    ):
                        seen.add(url)
                        references.append(DocumentReference(
                            source=self.source,
                            title=f"{profile['title']} data dictionary",
                            document_type="data_dictionary",
                            source_url=url,
                            dataset_id=profile["dataset_id"],
                            measure_ids=measure_ids,
                            metadata={
                                "catalog_identifier": raw.get("identifier"),
                                "distribution_format": distribution.get("format"),
                            },
                            content_type="application/json",
                        ))
            for key in ("documentation", "methodology", "landingPage", "references"):
                values = raw.get(key) or []
                if isinstance(values, str):
                    values = [values]
                for value in values:
                    url = (
                        value.get("@id") or value.get("url")
                        if isinstance(value, dict)
                        else value
                    )
                    if (
                        isinstance(url, str)
                        and self.validate_url(url)
                        and url not in seen
                    ):
                        seen.add(url)
                        references.append(DocumentReference(
                            source=self.source,
                            title=f"{profile['title']} {key}",
                            document_type=key.casefold(),
                            source_url=url,
                            dataset_id=profile["dataset_id"],
                            metadata={"catalog_field": key},
                        ))
        return references
