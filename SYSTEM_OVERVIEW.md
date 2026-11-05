# Local Health Data and Evidence Assistant

## What this system is

This project is a local, question-driven assistant for health, healthcare quality, utilization, access, and value-based-care (VBC) exploration. It brings together three kinds of evidence:

1. **Structured public data** from OECD and CMS, with a WHO adapter whose live catalog is currently unavailable.
2. **Official documentation** associated with datasets, such as definitions, methods, units, and caveats.
3. **Academic literature** retrieved from PubMed and indexed from its official metadata and abstracts.

The central design rule is to keep evidence types distinct. APIs are authoritative for numeric observations; official documents explain what those observations mean; research papers report study findings. Similarity is useful for discovery, but is not proof that two measures or conclusions are equivalent.

The codebase is a local Python application, not yet a production multi-user service. Its current user-facing orchestration primarily routes chat, document search, and OECD analysis. The literature evidence-package builder exists as a callable component but has not yet been wired into the chat routes.

## Goals and design considerations

- **Source adapters, shared concepts.** OECD, CMS, and WHO implement the structured `DataSource` interface while handling their own API formats. PubMed implements a separate `LiteratureSource`; research articles are not forced into the observation-data interface.
- **Bounded, auditable data access.** Dataset and concept discovery precede observation retrieval. Observation APIs use filters and limits. Retrieval traces retain candidates, compatibility outcomes, rejection reasons, and timings.
- **Do not substitute incompatible measures.** Deterministic parsing and metadata-based compatibility distinguish measure family, quantity, unit, geography, population, dates, and analysis intent when possible. Unknown stays unknown rather than being asserted as a match.
- **Keep vector scopes separate.** Dataset summaries, structured concepts, document chunks, and literature records have independent representations and retrieval paths. Raw observations are not embedded.
- **Preserve provenance and source values.** Dataset IDs, native codes, source URLs, article identifiers, document locations, and source-native values are retained where available.
- **Use deterministic analysis for calculations.** Analysis operates on normalized tabular observations. Literature can inform interpretation, but must not alter computed statistics; observational evidence should not be stated as causal evidence.
- **Prefer reproducible local infrastructure.** SQLite stores catalog structures, concepts, document chunks, and literature metadata; a local Ollama embedding endpoint supplies vectors. Upserts and cached-vector reuse limit repeated work.
- **Treat metadata associations conservatively.** Dataset/document or measure/document links are persisted when explicit in source metadata, not inferred merely because text vectors are similar.

## System map

```text
Question
   |
   +--> deterministic intent parsing (LLM resolution only for selected ambiguities)
   |
   +--> structured retrieval
   |      1. rank indexed datasets
   |      2. check dataset compatibility
   |      3. search concepts within candidate datasets
   |      4. select a compatible concept, or report why none qualifies
   |      5. adapter retrieves bounded observations
   |      6. normalize to pandas and calculate deterministic statistics
   |
   +--> official document search (separate document-vector scope)
   |
   +--> PubMed literature search
          1. bounded ESearch candidate list
          2. EFetch article metadata/abstracts
          3. semantic ranking plus separate factual-fit characteristics
          4. preserve identifiers and evidence-access level
   |
   +--> evidence package
          structured findings | official documentation | literature
          interpretation and actions remain empty until deliberately produced
```

The evidence-package boundary is designed to permit later synthesis without collapsing quantitative findings, methodology, and reported study results into one opaque score.

## Main components and entry points

| Component | Responsibility |
|---|---|
| [server.py](./server.py) | Small Flask HTTP wrapper for chat and local PDF indexing. |
| [agent.py](./agent.py) | Current conversational routing, OECD analysis workflow, document search, and legacy local-PDF indexing. |
| [data_sources/base.py](./data_sources/base.py) | Shared structured `DataSource` interface and common indexing flow. |
| [data_sources/oecd.py](./data_sources/oecd.py), [api_data.py](./api_data.py) | OECD SDMX catalog, structures, concepts, observations, and DataFrame conversion. |
| [data_sources/cms.py](./data_sources/cms.py) | CMS catalog and dataset-series handling, dictionaries/fields, deterministic filters, bounded API retrieval, and DataFrame conversion. |
| [data_sources/who.py](./data_sources/who.py) | WHO GHO indicator metadata, dimensions, concepts, filters, and bounded observations. |
| [data_sources/retrieval.py](./data_sources/retrieval.py) | Source-neutral two-stage dataset/concept retrieval, deterministic requirements parsing, compatibility checks, and trace instrumentation. |
| [data_sources/compatibility.py](./data_sources/compatibility.py) | Normalized measure-family and quantity/unit classification and compatibility decisions. |
| [data_sources/documents/](./data_sources/documents/) | Shared document discovery/ingestion/search plus OECD, CMS, and WHO documentation adapters. |
| [data_sources/literature/base.py](./data_sources/literature/base.py), [data_sources/literature/pubmed.py](./data_sources/literature/pubmed.py) | Literature interface and PubMed E-utilities search, metadata parsing, caching, embeddings, and ranking. |
| [evidence_synthesis.py](./evidence_synthesis.py) | Produces a separated, provenance-bearing evidence package; does not generate an LLM narrative. |
| [embedding.py](./embedding.py) | Local embedding calls, cosine ranking, and distinct structured/document scopes. |
| [database.py](./database.py), [schema.sql](./schema.sql) | SQLite schema, migrations, catalog persistence, document storage, and literature persistence. |
| [analysis.py](./analysis.py) | Deterministic time-series and DataFrame analysis with request and source provenance. |
| [pdf.py](./pdf.py), [tools.py](./tools.py) | Existing PDF/text extraction and chunking utilities reused by local and public-document ingestion. |
| [evaluation/retrieval_benchmark.py](./evaluation/retrieval_benchmark.py), [evaluation/literature_benchmark.py](./evaluation/literature_benchmark.py) | Structured and literature retrieval benchmark runners and their JSONL case definitions. |

## Data sources and current indexed corpus

### Structured data

- **OECD:** 138 indexed dataset structures, 9,665 semantic concepts, and 351,972 filter-code records.
- **CMS:** 5 indexed dataset structures, 134 semantic concepts, and 71 filter-code records.
- **WHO:** adapter and fixture tests exist, but no WHO structures are indexed in the current database. Live GHO indicator discovery returned HTTP 500 during prior validation.

The CMS live catalog is intentionally a small selected set, not an exhaustive CMS download. Dataset discovery does not mean every dataset is indexed or suitable for every question.

### Official documentation

The validated public document corpus contains:

- OECD: 5 documents, 712 chunks and embeddings.
- CMS: 13 documents, 179 chunks and embeddings.
- WHO: none currently ingested.
- Total public-source corpus: 18 documents and 891 embedded chunks.

The cache has 4.31 MB of downloaded source documents. Documentation supports interpretation and provenance; PDFs are not used as substitutes for structured observation APIs.

### Academic literature

The local PubMed cache currently contains 159 article records with 159 embeddings:

- 144 records include abstracts; 15 are metadata-only.
- 79 records have a PubMed metadata indication of a PMCID.
- A PMCID is not proof that a full-text copy was downloaded, that it is open access, or that its content can be reused. Full-text licenses have not been checked.

Search uses official NCBI E-utilities (`ESearch` and `EFetch`); queries are bounded to at most 100 results per call and use request pacing. Article metadata and abstracts are cached locally by PMID. PubMed indexing status does not guarantee abstract or full-text availability.

## Retrieval, analysis, and evidence handling

Structured retrieval first ranks datasets, then checks whether the dataset can support the question, then searches concepts only within candidate datasets. Compatibility rejects explicit mismatches such as spending for a utilization request or count for an explicitly requested rate. The trace exposes timing, candidates, checks, reasons, and selected dataset/concept; a missing compatible concept is a valid outcome.

Data adapters retrieve observations using native filters and limits. Results are converted to pandas while retaining source identifiers and useful labels. Deterministic analysis summarizes values and periods; it does not use a language model to calculate the result.

Document search is a distinct vector scope and can be filtered by source, dataset, document, or associated measure. Chunks preserve provenance such as URL, page, and section when available.

PubMed retrieval has two stages: ESearch finds bounded candidate articles, and local semantic/factual ranking orders returned metadata records. Characteristics remain individually visible (semantic relevance, study type, publication year, population, intervention, and outcome fit). The aggregate rank is used for ordering only; it is not a clinical evidence grade or certainty estimate.

The evidence package can carry:

1. deterministic analysis results,
2. official documentation,
3. academic literature,
4. LLM interpretation,
5. possible actions.

At present, layers 4 and 5 are intentionally empty. Provenance is attached to evidence records; the project does not yet extract and verify every sentence-level claim in an LLM narrative.

## Potential use cases

- Find which public datasets and measures could describe a population-health or VBC question.
- Compare utilization, rates, counts, spending, provider capacity, or quality without silently swapping measure types.
- Retrieve bounded country-, state-, provider-, or period-specific observations where the selected source supports them.
- Interpret a measure using its official definition, denominator, unit, methodology, limitations, and comparability notes.
- Locate research on a proposed intervention, inspect article design and abstract findings, and keep negative or null results visible.
- Assemble a traceable evidence package for a human analyst to assess a hypothesis and design a local evaluation.
- Use historical data as context for planning; do not treat cross-source associations or observational findings as proof that an intervention will work locally.

## Resource use: current observed footprint and planning notes

### Measured on this machine

| Resource | Current observed size |
|---|---:|
| `agent.db` SQLite file | 502.35 MB |
| Repository `embedding-model` directory | 101.14 MB |
| Public-document cache | 4.31 MB |
| **Combined measured footprint** | **about 607.8 MB (0.61 GB)** |

The 101.14 MB model folder is primarily a 99.59 MB Nomic GGUF file plus tokenizer/configuration files. The Ollama model store and runtime were not measured and may contain a separate model copy. The SQLite database includes application tables beyond the evidence corpus and should not be interpreted as a pure vector-store size.

### Work performed per request

- Structured retrieval reads and compares approximately 9,799 indexed semantic concept vectors across the current OECD/CMS catalog during initial dataset ranking; it then searches a narrower subset for concepts. The 351,972 deterministic filter records are not semantic vectors.
- Document retrieval compares vectors in the requested document scope; the current corpus contains 891 public-document chunk vectors.
- Literature search requests a bounded PubMed candidate set and compares only the returned/cached article vectors. The 15-query live benchmark compared an average of about 4.9 vectors per case at a five-result limit.
- Query/document/article embeddings use a local Ollama HTTP endpoint. Once content is cached and embedded, reruns can reuse vectors and avoid re-embedding.
- Live literature benchmark latency varied with cold versus warm caches and network/API calls. The final observed 15-case run reported a 0.092-second median and 1.657-second p95; these are benchmark-run figures, not a general service-level guarantee.

### What is not measured

Peak RAM, Ollama model residency, CPU/GPU utilization, per-request energy use, and concurrent-user capacity have not been benchmarked. Startup/indexing time also varies with source API availability and how many datasets or documents are selected. Plan disk for SQLite growth, backups, logs, temporary downloads, and any separate Ollama model storage in addition to the measured 0.61 GB.

There are no per-request hosted-model fees in the current local embedding path. Public APIs still depend on network availability, source rate limits, and their terms of use.

## Limitations and risks

1. **WHO live readiness is blocked.** Fixture behavior is tested; live GHO catalog access has failed with HTTP 500. No current WHO structures are available locally.
2. **Literature is abstract/metadata evidence.** Full texts are not fetched. PMCID availability is only an indicator; license, access, and reuse permission are not established.
3. **Search is not systematic review methodology.** PubMed query expansion and ranking are heuristic and bounded. A high-ranked result is not necessarily the strongest or most complete evidence, and absence from the result list is not evidence of absence.
4. **Study-type labels are imperfect.** PubMed publication types and textual signals support useful categories, but some articles remain `unknown`; design classification is not a formal risk-of-bias assessment.
5. **The literature benchmark is a smoke/recall proxy.** Its 15/15 top-five “topic recall” uses expected-term matches. It does not represent human-adjudicated relevance, evidence quality, effect direction, or completeness.
6. **Compatibility can be cautious to a fault.** Broad questions may have plausible datasets yet no concept passes compatibility. The system should expose that rather than invent a selection; an example home-based-utilization package retained candidates and documents but selected no compatible measure.
7. **No production synthesis workflow yet.** The evidence package is callable and tested, but the chat agent does not yet orchestrate structured analysis, documents, and literature into one user-facing answer. It deliberately produces no narrative or actions.
8. **Claim-level traceability is incomplete.** Evidence records have identifiers and provenance, but findings inside abstracts have not been normalized into independently verified atomic claims.
9. **Local server hardening is needed before network exposure.** The current Flask development server binds to all interfaces and enables debug mode. There is no authentication or authorization layer in the shown server wrapper. Use only in a trusted environment until this is corrected.
10. **Configuration and operations are still local-first.** Database paths, corpus paths, local services, and some workflow assumptions are machine-specific. Backups, retention, migrations, observability, and concurrency need operational policy before broader use.

## Validation snapshot

- Most recent full suite after literature implementation: **43 tests run; 40 passed and 3 opt-in WHO live tests skipped**.
- Literature fixture benchmark: **15/15 cases**, topic-hit Recall@5 **1.0**.
- Live PubMed benchmark: **15/15 cases**, topic-hit Recall@5 **1.0** in the tested run.
- Live home-based intervention evidence-package check returned structured dataset candidates, official documentation, and PubMed articles. It also exposed that zero concepts met compatibility; therefore no dataset/concept or numeric finding was selected. Observational articles were marked `association_only`, and abstracts containing negative/null language remained retrievable.
- Earlier structured-source validation confirmed live OECD and CMS documentation and selected examples. Current counts and API availability can change; rerun source smoke tests before relying on a particular live dataset.

The benchmark figures above are useful regression indicators, not a clinical validation or a guarantee of correct recommendations.

## Recommended next steps

1. **Integrate an explicit evidence workflow into orchestration.** Add a route or method that independently retrieves structured candidates, official documents, and literature, then requests observations only after a measure passes compatibility. Keep the evidence package inspectable before any LLM narrative.
2. **Improve evaluation quality.** Replace keyword-only success checks with a small human-reviewed set covering relevance, study design, effect direction, null/conflicting evidence, populations, and inappropriate causal language. Report per-case records and reasons.
3. **Strengthen evidence characterization.** Add explicit effect/outcome extraction only with source spans and reviewable provenance; distinguish association, randomized assignment, and unsupported causal wording. Do not equate study type alone with evidence certainty.
4. **Use PMC full text carefully.** If needed, integrate official PMC services for articles whose reuse license is explicitly compatible; persist the license and exact access level. Do not infer reusability from PMCID alone.
5. **Measure runtime and storage growth.** Profile peak memory and latency on representative workloads, and evaluate database backup/restore and vector/index growth before expanding the corpus.
6. **Restore and revalidate WHO.** Retry official WHO discovery, then run source and observation smoke tests before claiming live cross-source coverage.
7. **Harden deployment and configuration.** Disable Flask debug mode, avoid unrestricted network binding by default, add authentication if remote use is intended, and externalize local paths/service settings.
8. **Keep structured catalog growth selective.** Index datasets by useful metadata and a bounded use case; do not indiscriminately fetch and embed every source catalog or raw observation.

## A small artistic analogy

Think of the evidence layers like Charles Ives's *The Unanswered Question*: several musical lines can sound at once, but this system should let a reader hear which line came from public measurements, which from official methodology, and which from research literature. The point is not to force them into unison; it is to preserve enough provenance to understand their differences.
