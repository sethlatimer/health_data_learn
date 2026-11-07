# Local Health Data and Evidence Assistant

A local Python prototype for exploring population-health and value-based-care questions with public health datasets, official dataset documentation, and PubMed literature. It keeps numeric observations, source methodology, and research findings as separate evidence types; analysis is performed deterministically rather than delegated to an LLM.

## Start here

- [`SYSTEM_OVERVIEW.md`](SYSTEM_OVERVIEW.md) — architecture, current indexed corpus, validation notes, limitations, and next steps.
- [`vbc_population_health_agent_vision_and_evaluation.md`](vbc_population_health_agent_vision_and_evaluation.md) — intended workflows, evaluation framework, and acceptance criteria.

## Code map

- `server.py` — Flask wrapper and local UI endpoint.
- `agent.py` — conversational routing and analysis orchestration.
- `data_sources/` — OECD, CMS, WHO, document, and PubMed adapters.
- `analysis.py`, `evidence_synthesis.py` — deterministic analysis and provenance-bearing evidence packages.
- `database.py`, `schema.sql`, `embedding.py` — SQLite persistence and local semantic retrieval.
- `evaluation/` and `test_*.py` — benchmark runners and regression tests.

## Current status

The overview describes important remaining limitations, including incomplete chat integration for the cross-source evidence package and abstract/metadata-only literature retrieval in most cases. The Flask wrapper is intended for local use and is now configured to bind only to loopback with debug mode off by default; it still has no authentication, so do not expose it to an untrusted network.

## Setup and run

Requires Python 3.10 or newer and a local Ollama service for model-backed requests.

```sh
python -m venv .venv
. .venv/bin/activate                 # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements-lock.txt
python -m unittest discover -v
python server.py
```

The server defaults to `127.0.0.1:5000` with debug mode off. It has no authentication; keep it local. Do not set `HEALTH_DATA_HOST=0.0.0.0` unless you have separately secured the service and network.

## Configuration

Set these environment variables before starting the app; relative paths resolve from the project directory.

| Variable | Default |
|---|---|
| `HEALTH_DATA_DB_PATH` | `agent.db` in the project directory |
| `HEALTH_DATA_CORPUS_DIR` | `corpus/` |
| `HEALTH_DATA_ALLOWED_FILE_ROOT` | Same as the corpus directory |
| `HEALTH_DATA_DOCUMENT_CACHE_DIR` | Existing legacy cache if detected; otherwise `data/public_documents/` |
| `HEALTH_DATA_HOST` | `127.0.0.1` |
| `HEALTH_DATA_PORT` | `5000` |
| `HEALTH_DATA_DEBUG` | `false` |

The lock file pins the Python dependencies used for the local regression run; update it alongside `requirements.txt` when dependencies change.
