import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import database
import embedding
from data_sources.literature import LiteratureSource, PubMedSource
from data_sources.literature.pubmed import (
    _article_metadata,
    _factual_relevance,
    build_pubmed_search_query,
)
from evaluation.literature_benchmark import run_benchmark
from evidence_synthesis import build_evidence_package


PUBMED_XML = b"""<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>101</PMID>
      <Article>
        <Journal><Title>Journal of Home Care</Title>
          <JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue>
        </Journal>
        <ArticleTitle>Hospital at home and acute utilization</ArticleTitle>
        <Abstract><AbstractText Label="POPULATION">Adults with acute illness.</AbstractText>
          <AbstractText Label="INTERVENTION">Hospital at home care.</AbstractText>
          <AbstractText Label="OUTCOMES">Hospital utilization was associated with lower emergency visits.</AbstractText>
        </Abstract>
        <AuthorList><Author><ForeName>Alex</ForeName><LastName>Example</LastName></Author></AuthorList>
        <PublicationTypeList><PublicationType>Cohort Study</PublicationType></PublicationTypeList>
      </Article>
      <MeshHeadingList><MeshHeading><DescriptorName>Home Care Services</DescriptorName></MeshHeading></MeshHeadingList>
    </MedlineCitation>
    <PubmedData><ArticleIdList>
      <ArticleId IdType="pubmed">101</ArticleId>
      <ArticleId IdType="pmc">PMC1010</ArticleId>
      <ArticleId IdType="doi">10.5555/home.101</ArticleId>
    </ArticleIdList></PubmedData>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>202</PMID>
      <Article>
        <Journal><Title>Health Services Research</Title>
          <JournalIssue><PubDate><Year>2023</Year></PubDate></JournalIssue>
        </Journal>
        <ArticleTitle>Hospital at home randomized trial with null utilization findings</ArticleTitle>
        <Abstract><AbstractText Label="RESULTS">There was no significant difference in acute utilization between groups.</AbstractText></Abstract>
        <PublicationTypeList><PublicationType>Randomized Controlled Trial</PublicationType></PublicationTypeList>
      </Article>
    </MedlineCitation>
    <PubmedData><ArticleIdList><ArticleId IdType="pubmed">202</ArticleId></ArticleIdList></PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""


class _Response:
    def __init__(self, *, payload=None, content=None):
        self._payload = payload
        self.content = content

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FixtureRetrieval:
    def retrieve(self, question, **kwargs):
        return {
            "selected_source": "OECD",
            "selected_dataset": {"source": "OECD", "dataset_id": "d1"},
            "selected_concept": {"code": "M1", "label": "Hospital admission rate"},
            "compatible_dataset_candidates": [
                {"source": "OECD", "dataset_id": "d1"},
                {"source": "CMS", "dataset_id": "d2"},
            ],
            "instrumentation": {"counts": {"datasets_considered": 2}},
        }


class _FixtureDocuments:
    def search(self, question, **kwargs):
        return [{
            "source": "OECD",
            "dataset_id": kwargs["dataset_id"],
            "document_id": 1,
            "chunk_id": 2,
            "source_url": "https://oecd.example/methods",
            "page_number": 4,
            "section": "Definitions",
            "score": 0.8,
            "content": "The indicator definition and limitations.",
        }]


class _FixtureLiterature(LiteratureSource):
    source = "PUBMED"

    def __init__(self, articles):
        self.articles = articles
        self.last_trace = {"selected_article": articles[0]["pmid"]}

    def search(self, query, limit=20, embedding_model="nomic-local"):
        return self.articles[:limit]

    def get_article(self, identifier, embedding_model="nomic-local"):
        return next(item for item in self.articles if item["pmid"] == identifier)


class PubMedSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_patch = patch.object(
            database,
            "DB",
            str(Path(self.temp_dir.name) / "literature.db"),
        )
        self.db_patch.start()

    def tearDown(self):
        self.db_patch.stop()
        self.temp_dir.cleanup()

    def test_metadata_preserves_ids_access_and_structured_study_details(self):
        import xml.etree.ElementTree as ET

        article = ET.fromstring(PUBMED_XML).find("PubmedArticle")
        record = _article_metadata(article)
        self.assertEqual(record["pmid"], "101")
        self.assertEqual(record["pmcid"], "PMC1010")
        self.assertEqual(record["doi"], "10.5555/home.101")
        self.assertEqual(record["publication_year"], 2024)
        self.assertEqual(record["journal"], "Journal of Home Care")
        self.assertEqual(record["authors"], ["Alex Example"])
        self.assertEqual(record["study_metadata"]["population"], ["Adults with acute illness."])
        self.assertEqual(record["study_metadata"]["intervention"], ["Hospital at home care."])
        self.assertTrue(record["full_text_available"])
        self.assertEqual(
            record["full_text_url"],
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC1010/",
        )
        self.assertEqual(record["full_text_reuse_status"], "license_not_assessed")
        no_pmc_article = _article_metadata(
            ET.fromstring(PUBMED_XML).findall("PubmedArticle")[1]
        )
        self.assertFalse(no_pmc_article["full_text_available"])
        self.assertEqual(
            no_pmc_article["full_text_access_status"],
            "not_indicated_by_pubmed_metadata",
        )
        self.assertEqual(no_pmc_article["full_text_reuse_status"], "not_assessed")
        self.assertEqual(record["evidence_access_level"], "abstract")
        self.assertIn("PMID 101", record["semantic_text"])
        self.assertIn("raw_pubmed_xml", record["metadata"])

    def test_search_is_bounded_cached_and_upserts_without_duplicate_records(self):
        session = Mock()
        session.get.side_effect = [
            _Response(payload={"esearchresult": {"idlist": ["101", "202"]}}),
            _Response(content=PUBMED_XML),
        ]
        source = PubMedSource(session=session)
        with patch.object(
            embedding,
            "embed_texts",
            side_effect=lambda texts, embedding_model: [
                [1.0, 0.0] for _ in texts
            ],
        ), patch.object(
            embedding,
            "embed_text",
            return_value=[1.0, 0.0],
        ):
            first = source.search(
                "hospital at home acute utilization",
                limit=2,
            )
            second = source.search(
                "hospital at home acute utilization",
                limit=2,
            )

        self.assertEqual({item["pmid"] for item in first}, {"101", "202"})
        self.assertEqual(len(second), 2)
        self.assertTrue(any(
            item["causal_inference_status"] == "association_only"
            for item in first
        ))
        self.assertEqual(source.last_trace["candidate_stage"]["cache_hit"], True)
        self.assertEqual(source.last_trace["metadata_stage"]["metadata_cache_hits"], 2)
        self.assertEqual(source.last_trace["embedding"]["cache_hits"], 2)
        self.assertEqual(session.get.call_count, 2)
        conn = database.connect()
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM literature_records"
            ).fetchone()[0]
            self.assertEqual(count, 2)
        finally:
            conn.close()
        for call in session.get.call_args_list:
            self.assertIn("eutils.ncbi.nlm.nih.gov", call.args[0])

    def test_factual_fit_keeps_study_design_unknown_without_evidence(self):
        relevance = _factual_relevance(
            "Find evidence about hospital utilization",
            {
                "title": "Hospital use",
                "abstract": None,
                "publication_types": [],
                "mesh_terms": [],
            },
        )
        self.assertEqual(relevance["study_type"], ["unknown"])
        self.assertIsNone(relevance["population_fit"]["score"])

    def test_evidence_package_preserves_opposing_literature_and_provenance(self):
        articles = [
            {
                "pmid": "101",
                "pmcid": "PMC1010",
                "doi": "10.5555/home.101",
                "title": "Hospital at home and acute utilization",
                "abstract": "In this cohort study, intervention was associated with lower emergency visits.",
                "publication_year": 2024,
                "journal": "Journal of Home Care",
                "publication_types": ["Cohort Study"],
                "study_metadata": {"study_type": ["observational"]},
                "factual_relevance": {"study_type": ["observational"]},
                "scores": {"semantic_relevance": 0.9},
                "evidence_access_level": "abstract",
                "full_text_available": True,
                "full_text_reuse_status": "license_not_assessed",
                "causal_inference_status": "association_only",
                "source_url": "https://pubmed.ncbi.nlm.nih.gov/101/",
            },
            {
                "pmid": "202",
                "pmcid": None,
                "doi": None,
                "title": "Hospital at home randomized trial with null findings",
                "abstract": "There was no significant difference in acute utilization.",
                "publication_year": 2023,
                "journal": "Health Services Research",
                "publication_types": ["Randomized Controlled Trial"],
                "study_metadata": {"study_type": ["randomized_trial"]},
                "factual_relevance": {"study_type": ["randomized_trial"]},
                "scores": {"semantic_relevance": 0.85},
                "evidence_access_level": "abstract",
                "full_text_available": False,
                "full_text_reuse_status": "not_available_in_pmc_metadata",
                "causal_inference_status": "not_assessed",
                "source_url": "https://pubmed.ncbi.nlm.nih.gov/202/",
            },
        ]
        deterministic_result = {
            "statistics": {"first_value": 10, "last_value": 8},
            "provenance": {"dataset_id": "d1", "source": "OECD"},
        }
        package = build_evidence_package(
            "Where might hospital at home reduce acute utilization?",
            structured_retrieval=_FixtureRetrieval(),
            document_service=_FixtureDocuments(),
            literature_source=_FixtureLiterature(articles),
            top_articles=5,
            analysis_result=deterministic_result,
        )
        layers = package["evidence_layers"]
        studies = layers["academic_literature"]
        self.assertEqual([item["evidence_id"] for item in studies], [
            "pubmed:101",
            "pubmed:202",
        ])
        self.assertEqual(studies[0]["causal_inference_status"], "association_only")
        self.assertIn("no significant difference", studies[1]["reported_findings"])
        self.assertEqual(len(layers["official_documentation"]), 2)
        self.assertEqual(
            len(layers["structured_data_candidates"]["dataset_candidates"]),
            2,
        )
        self.assertEqual(len(layers["structured_data_candidates"]["compatible_candidates"]), 2)
        self.assertEqual(
            layers["computed_structured_findings"][0]["result"],
            deterministic_result,
        )
        self.assertEqual(len(layers["llm_interpretation"]), 0)
        self.assertEqual(len(layers["possible_actions"]), 0)
        self.assertIsNone(package["generated_narrative"])
        self.assertFalse(package["synthesis_constraints"]["llm_narrative_generated"])
        literature_provenance = [
            item for item in package["claim_provenance"]
            if item.get("evidence_type") != "deterministic_analysis"
        ]
        self.assertTrue(all(item.get("source_url") for item in literature_provenance))
        deterministic_provenance = package["claim_provenance"][0]
        self.assertEqual(deterministic_provenance["dataset_id"], "d1")

    def test_search_limit_is_validated(self):
        with self.assertRaises(ValueError):
            PubMedSource().search("diabetes", limit=101)

    def test_pubmed_query_drops_conversational_scaffolding(self):
        query = build_pubmed_search_query(
            "Where might a home-based intervention reduce avoidable acute "
            "utilization, and what evidence contradicts it?"
        )
        self.assertIn('"hospital at home"[Title/Abstract]', query)
        self.assertIn('"healthcare utilization"[Title/Abstract]', query)
        self.assertIn(" OR ", query)
        self.assertIn('"no significant difference"[Title/Abstract]', query)
        self.assertNotIn("contradicts", query)

    def test_fixture_benchmark_runs_all_vbc_cases(self):
        with contextlib.redirect_stdout(io.StringIO()):
            report = run_benchmark(mode="fixture", limit=5)
        self.assertEqual(report["aggregate_report"]["cases"], 15)
        self.assertEqual(report["aggregate_report"]["topic_recall_at_5"], 1.0)
        self.assertFalse(report["aggregate_report"]["failed_cases"])


if __name__ == "__main__":
    unittest.main()
