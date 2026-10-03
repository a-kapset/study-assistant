# Evaluation harness

Evaluates the study assistant from the outside: it calls the running app over HTTP and records every
exchange as an observation, so a failed request is data for a test, never an exception. Package: `evalharness`.

How checks are split between the app's own tests and the harness: [TESTING.md](../TESTING.md).

## Stack

- Python 3.14, [httpx2](https://github.com/pydantic/httpx2) (async HTTP client),
  [pydantic](https://docs.pydantic.dev/) and pydantic-settings, [structlog](https://www.structlog.org/) (JSON logs)
- [Poetry 2](https://python-poetry.org/) for dependencies and the virtual environment
- Development: pytest with pytest-asyncio, ruff (format and lint), mypy (strict), pre-commit

## Structure

```text
evaluation-harness/
├── pyproject.toml          # project metadata, dependencies, ruff / mypy / pytest config
├── poetry.lock
├── src/evalharness/
│   ├── settings.py         # HarnessSettings: where the system under test is (EVAL_* variables)
│   ├── transport.py        # SutTransport: async HTTP client that never raises on HTTP or network errors
│   └── observation.py      # Observation: one request/response record, or the transport error instead
└── tests/
    ├── unit/               # offline: HTTP is simulated, no app needed
    └── api/                # need a running system under test; selected with -m api
```

## Getting started

Requires Python 3.14 and Poetry 2. All commands run from `evaluation-harness/`.

```powershell
poetry install --with lint
```

## Configuration

Settings are read from environment variables when a run starts.

| Variable | Default | Meaning |
|---|---|---|
| `EVAL_SUT_BASE_URL` | `http://127.0.0.1:8000` | Root URL of the system under test; request paths are joined to it |
| `EVAL_REQUEST_TIMEOUT_S` | `10.0` | Per-request timeout in seconds (must be greater than 0) |

```powershell
$env:EVAL_SUT_BASE_URL = "http://127.0.0.1:8001"   # only if the app is not on the default address
```

## Running the tests

```powershell
poetry run pytest           # offline tests only; api tests are deselected
poetry run pytest -m api    # api tests against the running app
```

Start the app first for `-m api` (see [app/README.md](../app/README.md)). If the app is unreachable,
the api tests fail with the transport error in the message; they are never skipped, so a run with no
app cannot look green.

## Development checks

```powershell
poetry run ruff format --check .
poetry run ruff check .
poetry run mypy .
```

The same checks run as pre-commit hooks from the repository root. Run pre-commit from a terminal with
no activated virtual environment: Poetry prefers an active one over this project's own.
