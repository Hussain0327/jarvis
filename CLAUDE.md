# Project Jarvis

Intelligent Traffic Intelligence Platform — AI-powered traffic camera analysis with vehicle detection, tracking, analytics, and a natural language interface.

## Tech Stack

- **Language:** Python 3.11+
- **ML/DL:** PyTorch, Ultralytics YOLOv8, TrOCR, BoxMOT, scikit-learn
- **Database:** PostgreSQL 16 + pgvector (SQLAlchemy 2.0 sync ORM, Alembic migrations)
- **API:** FastAPI
- **Dashboard:** Streamlit + Folium + Plotly
- **LLM:** Qwen3-30B-A3B via Ollama (local) / vLLM (production)
- **Cache:** Redis (deferred to Phase 6)
- **Config:** pydantic-settings + YAML defaults (`config/config.yaml`)
- **Logging:** loguru (console human-readable + JSON file)
- **Linting:** ruff
- **Testing:** pytest

## Commands

```bash
# Install (dev)
pip install -e ".[dev]"

# Install (everything)
pip install -e ".[all]"

# Lint
ruff check src/ tests/

# Test (all unit tests — 108 tests, no DB required)
pytest tests/ -v

# Test (with coverage)
pytest tests/ --cov=src --cov-report=term-missing

# Test (integration — requires PostgreSQL)
docker compose -f docker/docker-compose.yml up -d
pytest -m integration

# Database migrations
alembic revision --autogenerate -m "description"
alembic upgrade head

# Start PostgreSQL
docker compose -f docker/docker-compose.yml up -d
```

## Project Layout

```
src/
  __init__.py          # Package root, __version__
  config.py            # Settings (pydantic-settings + YAML)
  logging.py           # loguru setup
  base.py              # ABCs: BaseProcessor, BaseTracker, BaseAnalyzer + dataclasses
  database/
    models.py          # SQLAlchemy ORM: Camera, Vehicle, Detection
    session.py         # Engine, session factory, pgvector extension
    queries.py         # Query functions (populated per-phase)
    migrations/        # Alembic
  ingestion/           # Phase 1: camera feeds
  perception/          # Phase 1: YOLO + TrOCR
  tracking/            # Phase 2: MOT + re-ID
  analytics/           # Phase 4: clustering, anomaly, prediction
  api/                 # Phase 5: FastAPI
  llm/                 # Phase 6: RAG + function calling
  dashboard/           # Phase 5: Streamlit
tests/
config/config.yaml     # Default configuration
docker/docker-compose.yml  # PostgreSQL 16 + pgvector
```

## Conventions

- **Config priority:** env vars > .env > config.yaml > defaults
- **Env var prefix:** `JARVIS_` (nested: `JARVIS_DATABASE__HOST`)
- **ORM:** SQLAlchemy 2.0 mapped_column style (sync, not async)
- **Migrations:** Alembic autogenerate from `src/database/models.py`
- **Base classes:** All processors inherit `BaseProcessor[In, Out]`, trackers inherit `BaseTracker`, analyzers inherit `BaseAnalyzer[In, Out]`
- **Line length:** 100 chars (ruff)
- **Target Python:** 3.11+
- **Imports:** use `from __future__ import annotations` in all modules
- **Thread safety:** All shared mutable state uses `threading.Lock` (double-checked locking for singletons)
- **DB upserts:** Use `INSERT ... ON CONFLICT DO UPDATE` via `sqlalchemy.dialects.postgresql.insert`, never SELECT-then-INSERT
- **Credentials:** Never hardcode passwords; use env vars or `.env` file (see `.env.example`)
- **Processor guards:** All `BaseProcessor` subclasses must raise `RuntimeError` if `process()` is called before `load()`

## Current Status

- **Phase 1 (Ingestion + Perception):** Complete
- **Phase 2 (Tracking):** Complete
- **Audit:** 31-issue codebase audit complete (thread safety, security, config, reliability, performance, tests)
- **Tests:** 108 passing, 4 skipped (DB integration), lint clean
- **Next:** Phase 3 (Database Pipeline), Phase 4 (Analytics Engine)
