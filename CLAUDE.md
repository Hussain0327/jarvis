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

# Test (unit only)
pytest tests/test_config.py

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
