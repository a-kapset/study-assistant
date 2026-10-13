# Study Assistant — app

A domain-independent study assistant served as a FastAPI web service. Package: `study_assistant`.

How the app is tested, at which level and why: [TESTING.md](../TESTING.md).

> **Status: work in progress.** Both the app and the evaluation harness are under active development;
> the READMEs of each part describe what already works.

## Stack

- Python 3.14, [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://uvicorn.dev/),
  [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) for configuration,
  [structlog](https://www.structlog.org/) for JSON logs, [PyYAML](https://pyyaml.org/) for course manifests,
  [pypdfium2](https://pypdfium2.readthedocs.io/) for the text layer of PDF sources,
  [LangChain text splitters](https://docs.langchain.com/oss/python/integrations/splitters) for cutting sections
  into chunks
- PostgreSQL 18 with [pgvector](https://github.com/pgvector/pgvector), reached through
  [psycopg 3](https://www.psycopg.org/psycopg3/) and its connection pool
- Docker Compose for running the app next to its database
- [Poetry 2](https://python-poetry.org/) for dependencies and the virtual environment
- Development: pytest with [Hypothesis](https://hypothesis.readthedocs.io/) for property tests, ruff (format and lint),
  mypy (strict), pre-commit

## Structure

```text
app/
├── pyproject.toml          # project metadata, dependencies, ruff / mypy / pytest config
├── poetry.lock
├── Dockerfile              # two-stage image: Poetry installs runtime dependencies, a non-root user runs the app
├── .dockerignore
├── src/study_assistant/
│   ├── main.py             # server entry point create_app(): settings from the environment
│   ├── composition.py      # build_app(settings): the one place where parts are built and wired
│   ├── settings.py         # typed settings; unknown APP_* variables stop startup
│   ├── logging_setup.py    # structlog, one JSON object per line on stdout
│   ├── errors.py           # the error envelope and the handlers that use it
│   ├── db.py               # Database protocol and the PostgreSQL pool, opened at startup, closed at shutdown
│   ├── migrate.py          # one-shot migration run: applies new SQL files, prepares the app's database role
│   ├── migrations/         # numbered SQL files (001_vector.sql, 002_corpus.sql, ...), applied in order
│   ├── courses/            # courses as data: nothing course-specific lives in the code
│   │   ├── models.py       # the course manifest: metadata, structure rules, source files
│   │   └── registry.py     # loads the enabled courses' manifests at startup; a bad one stops the app
│   ├── ingest/             # course sources to clean page text, the course structure and chunks
│   │   ├── parser.py       # PDF text layer page by page, with the parser and PDF engine versions
│   │   ├── normalize.py    # header and footer lines, words split at line ends, spacing, Unicode form; page status
│   │   ├── structure.py    # sections and learning objectives found by the course's structure rules
│   │   └── chunker.py      # chunks within sections: size cap, pages, objectives, content-derived ids, rules version
│   └── api/                # HTTP layer: one router factory per group of endpoints, plus middleware
│       ├── middleware.py   # request id on every request; unhandled exceptions become a 500 envelope
│       ├── health.py
│       ├── courses.py
│       └── config.py
└── tests/
    └── unit/               # in-process tests: the API through FastAPI's TestClient, parsing on the demo course PDF
```

## Getting started

### With Docker (the app and its database)

From the repository root, with Docker running:

```powershell
Copy-Item .env.example .env                   # then set DB_PASSWORD and DB_APP_PASSWORD in .env
$env:GIT_COMMIT = git rev-parse --short HEAD  # optional: shown by /v1/config
docker compose up -d --build --wait
docker compose logs migrate --no-log-prefix   # which migration files were applied
curl.exe -i http://127.0.0.1:8000/ready
```

The stack starts in order: the database, then `migrate`, a one-shot run of the same image that applies new
migration files as the database owner and prepares the app's role, then the app, only after `migrate` has
exited without an error. `--wait` returns once the database and the app are healthy (the app counts as
healthy when `/ready` answers 200); `docker compose ps -a` shows `migrate` as exited with code 0.

`docker compose down` stops the stack and keeps the database volume; `docker compose down -v` also
deletes it, which is needed after changing `DB_OWNER` or `DB_PASSWORD`, because PostgreSQL sets the owner
only when the volume is first created. The database port is not published; use
`docker compose exec db psql -U study_assistant -d study_assistant` (the `DB_OWNER` and `DB_NAME` from `.env`)
to reach it.

Compose mounts the repository's `courses/` folder read-only and enables the courses in
`APP_COURSES__ENABLED` (`demo` in `.env.example`); check with `curl.exe http://127.0.0.1:8000/v1/courses`.


### Without Docker

Requires Python 3.14 and Poetry 2. All commands run from `app/`.

```powershell
poetry install --with lint
$env:APP_DB__PASSWORD = "change-me"
poetry run uvicorn study_assistant.main:create_app --factory
```

The service listens on `http://127.0.0.1:8000`; interactive API docs are at `/docs`. It starts even
when no database is reachable: `/health` answers 200 and `/ready` 503. On Windows, Uvicorn runs on
an event loop that async psycopg cannot use, so the database is reachable only when the app runs in
Docker.

## Configuration

Everything the local stack is configured with is in one file: the repository root's
[`.env.example`](../.env.example), copied to `.env`. It lists every value with what it means and which app
setting (`APP_*`) it becomes. `compose.yaml` takes each value from `.env` (a missing one stops Compose with
its name) and gives each service only the values it needs: the app gets the app role's password, never the
database owner's.

The app reads its settings from the process environment only, as `APP_*` variables (nested groups as
`APP_GROUP__FIELD`, e.g. `APP_DB__HOST`), and does not load a `.env` file itself. An `APP_*` variable that
matches no setting stops startup with the unknown names listed, so a typo never falls back to a default
silently. Defaults in the code matter only for unit tests and runs without Compose; `APP_DB__PASSWORD` has
none. `GET /v1/config` shows the effective values with every secret masked.

The migration run (`python -m study_assistant.migrate`) reads the same database variables with the owner's
user and password, plus `APP_SERVER_ROLE__NAME` and `APP_SERVER_ROLE__PASSWORD` for the app's role; the app
server refuses these two.

In PowerShell, set a variable for the current window before starting the server:

```powershell
$env:APP_LOG_LEVEL = "DEBUG"
poetry run uvicorn study_assistant.main:create_app --factory
```

Logs are JSON lines on stdout, each with the request id while a request is handled; uvicorn's own
startup and access lines stay plain text.

## Database

The schema is created by numbered SQL files in `src/study_assistant/migrations/`. The migration run applies
the files the database has not seen yet, in number order, each in its own transaction together with its row
in the `schema_migrations` journal, so a failing file leaves nothing behind. The journal keeps each file's
SHA-256: an applied file must never be edited (the run stops and names it); a change goes into a new file
with the next number.

After the files, the run creates the app's role (`DB_APP_USER`) or resets its password, and lets it select,
insert, update and delete rows of every table and only read the journal. The app connects as that role, so it
cannot create, alter or drop anything; only the migration run uses the owner.

Tables: `courses`, `documents` (one per source file, with its SHA-256 and the chunking rules that produced its
rows), `sections`, `learning_objectives` and `chunks`. Every row carries its course id; a document's
sections, objectives and chunks are deleted with it.

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness: `{"status": "ok"}` while the process is serving HTTP |
| `GET` | `/v1/config` | Effective configuration: app version and settings, every secret masked |
| `GET` | `/ready` | Readiness: 200 `{"status": "ok", "checks": {"db": "ok"}}` when the database answers, otherwise 503 in the error envelope |
| `GET` | `/v1/courses` | The enabled courses in the configured order: `id`, `title`, `description`, `language` |


## Errors and request ids

Every response carries an `X-Request-ID` header. A client may send its own id (letters, digits,
`.`, `_`, `-`, up to 128 characters) and gets it back; otherwise the app generates a UUID. The same
id is in every log line written while the request is handled, so a response can be matched with its
logs.

Every error has the status code that fits it and one body format:

```json
{"status": "error", "error": {"code": "not_found", "message": "Not Found"}, "request_id": "..."}
```

`code` is stable for programs to read: `not_found`, `method_not_allowed`, `internal_server_error` and
so on. A 422 also lists `details` with the location and reason of each invalid field; the values that were
sent are not echoed. A 503 from `/ready` lists the failed checks the same way, for example
`{"loc": ["checks", "db"], "msg": "unavailable"}`. A 500 never shows the exception; it is written to the log with the
traceback and the request id.

```powershell
curl.exe -i http://127.0.0.1:8000/nope
curl.exe -i -H "X-Request-ID: my-test-1" http://127.0.0.1:8000/health
```

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
