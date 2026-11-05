from data_sources.documents.base import DocumentReference, DocumentSource


_TEXT_FIELDS = (
    "Definition",
    "Rationale",
    "MethodOfMeasurement",
    "MethodOfEstimation",
    "Interpretation",
    "Limitations",
    "DataSource",
    "Source",
    "Unit",
)


class WHODocumentSource(DocumentSource):
    source = "WHO"
    official_hosts = ("who.int", "ghoapi.azureedge.net")

    def discover_documents(self, profiles=None, dataset_id=None):
        references = []
        for profile in profiles or []:
            if dataset_id and profile["dataset_id"] != dataset_id:
                continue
            raw = profile.get("metadata") or {}
            body = [
                f"WHO indicator: {profile['title']}",
                f"Indicator code: {profile['dataset_id']}",
            ]
            for key in _TEXT_FIELDS:
                value = raw.get(key)
                if value is not None and str(value).strip():
                    body.append(f"{key}: {value}")
            content = "\n\n".join(body)
            source_url = profile.get("source_url")
            if not source_url or not self.validate_url(source_url):
                source_url = profile.get("structure", {}).get(
                    "dataflow", {}
                ).get("api_url")
            if not source_url or not self.validate_url(source_url):
                source_url = "https://www.who.int/data/gho/info/gho-odata-api"
            references.append(DocumentReference(
                source=self.source,
                title=f"{profile['title']} indicator metadata",
                document_type="indicator_metadata",
                source_url=source_url,
                dataset_id=profile["dataset_id"],
                measure_ids=[profile["dataset_id"]],
                updated_at=(
                    raw.get("LastUpdated")
                    or raw.get("Date")
                ),
                metadata={
                    "indicator_code": profile["dataset_id"],
                    "metadata_fields": [
                        key for key in _TEXT_FIELDS
                        if raw.get(key) is not None
                    ],
                    "authority": "WHO GHO indicator metadata",
                },
                inline_text=content,
                content_type="text/plain",
            ))
        return references
