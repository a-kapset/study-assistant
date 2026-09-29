# Study Assistant — app

A domain-independent study assistant served as a FastAPI web service. Package: `study_assistant`.

## Stack

- Python 3.14, [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://uvicorn.dev/)
- [Poetry 2](https://python-poetry.org/) for dependencies and the virtual environment
- Development: pytest, ruff (format and lint), mypy (strict), pre-commit

## Structure

```text
app/
├── pyproject.toml          # project metadata, dependencies, ruff / mypy / pytest config
├── poetry.lock
├── src/study_assistant/
│   └── main.py             # application factory create_app()
└── tests/
    └── unit/               # in-process tests with FastAPI's TestClient
```

## Getting started

Requires Python 3.14 and Poetry 2. All commands run from `app/`.

```powershell
poetry install --with lint
poetry run uvicorn study_assistant.main:create_app --factory
```

The service listens on `http://127.0.0.1:8000`; interactive API docs are at `/docs`.

```powershell
curl.exe -i http://127.0.0.1:8000/health
```

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness: `{"status": "ok"}` while the process is serving HTTP |

## Development checks

```powershell
poetry run ruff format --check .
poetry run ruff check .
poetry run mypy .
poetry run pytest
```

The same checks, a secret scan and a check that app code contains no course-specific names run as
pre-commit hooks from the repository root:

```powershell
app\.venv\Scripts\pre-commit install          # once, from the repository root
app\.venv\Scripts\pre-commit run --all-files
```
