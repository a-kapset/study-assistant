# Study Assistant — app

A domain-independent study assistant served as a FastAPI web service. Package: `study_assistant`.

How the app is tested, at which level and why: [TESTING.md](../TESTING.md).

## Stack

- Python 3.14, [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://uvicorn.dev/),
  [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) for configuration,
  [structlog](https://www.structlog.org/) for JSON logs
- [Poetry 2](https://python-poetry.org/) for dependencies and the virtual environment
- Development: pytest, ruff (format and lint), mypy (strict), pre-commit

## Structure

```text
app/
├── pyproject.toml          # project metadata, dependencies, ruff / mypy / pytest config
├── poetry.lock
├── .env.example            # every APP_* variable with its default
├── src/study_assistant/
│   ├── main.py             # server entry point create_app(): settings from the environment
│   ├── composition.py      # build_app(settings): the one place where parts are built and wired
│   ├── settings.py         # typed settings; unknown APP_* variables stop startup
│   ├── logging_setup.py    # structlog, one JSON object per line on stdout
│   └── api/                # one router factory per group of endpoints
│       ├── health.py
│       └── config.py
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

## Configuration

All settings come from environment variables with the `APP_` prefix; `.env.example` lists them with
their defaults. The app reads the process environment only and does not load a `.env` file itself.
An `APP_*` variable that matches no setting stops startup with the unknown names listed, so a typo
never falls back to a default silently.

| Variable | Default | Meaning |
|---|---|---|
| `APP_LOG_LEVEL` | `INFO` | Lowest log level written: `DEBUG`, `INFO`, `WARNING` or `ERROR` |
| `APP_GIT_COMMIT` | `unknown` | Commit the running code was built from; set by the build |

In PowerShell, set a variable for the current window before starting the server:

```powershell
$env:APP_LOG_LEVEL = "DEBUG"
poetry run uvicorn study_assistant.main:create_app --factory
```

Logs are JSON lines on stdout; uvicorn's own startup lines stay plain text.

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness: `{"status": "ok"}` while the process is serving HTTP |
| `GET` | `/v1/config` | Effective configuration: app version and settings, every secret masked |

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
