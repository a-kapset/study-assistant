# Evaluation harness

Evaluates the study assistant from the outside: it calls the running app over HTTP and records every
exchange as an observation, so a failed request is data for a test, never an exception. Package: `evalharness`.

How checks are split between the app's own tests and the harness: [TESTING.md](../TESTING.md).
How every metric the harness reports is computed and how to read the metrics together: [METRICS.md](METRICS.md).

> **Status: work in progress.** Both the app and the evaluation harness are under active development;
> the READMEs of each part describe what already works.

## Stack

- Python 3.14, [httpx2](https://github.com/pydantic/httpx2) (async HTTP client),
  [pydantic](https://docs.pydantic.dev/) and pydantic-settings, [structlog](https://www.structlog.org/) (JSON logs),
  [psycopg 3](https://www.psycopg.org/psycopg3/) for reading the app's database directly
- [Poetry 2](https://python-poetry.org/) for dependencies and the virtual environment
- Development: [pytest](https://docs.pytest.org/) with [pytest-asyncio](https://pytest-asyncio.readthedocs.io/),
  [ruff](https://docs.astral.sh/ruff/) (format and lint), [mypy](https://mypy.readthedocs.io/) (strict),
  [pre-commit](https://pre-commit.com/)

## Structure

```text
evaluation-harness/
├── pyproject.toml          # project metadata, dependencies, ruff / mypy / pytest config
├── poetry.lock
├── METRICS.md              # the formula and meaning of every metric the harness reports
├── datasets/               # gold set declarations, one folder per course (the questions stay local)
├── runs/                   # run records, one JSON file per run (not committed)
├── src/evalharness/
│   ├── settings.py         # where the system under test is (EVAL_*), and the values shared with the stack's .env
│   ├── corpus_db.py        # direct reads of the app's stored corpus, as the read-only database role
│   ├── datasets/           # builds a gold set from a course's local question bank under its declaration
│   ├── oracles/            # scores one search against the answer section, and summarises a group of questions
│   ├── evaluation/         # a retrieval run: checks, two passes over the search, the run record, the command
│   ├── testdata/           # test data written into the app's stores: created and removed in one place
│   │   ├── __init__.py     # the name prefix every piece of test data carries
│   │   ├── cleanup.py      # removes all test data from every store and lists what is left (no kind of data yet)
│   │   └── __main__.py     # the clean / check command
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
| `EVAL_STACK_ENV_FILE` | `../.env` | The `.env` file the stack under test runs with |
| `EVAL_DB_HOST` | `127.0.0.1` | Host on which the stack publishes its database port |

Values the harness shares with the stack are read from the stack's own `.env` (the repository root's, see
[`.env.example`](../.env.example)), so they are set in one place: `DB_NAME`, `DB_PUBLISHED_PORT`, the read-only
role `DB_READER_USER` / `DB_READER_PASSWORD`, the app's role `DB_APP_USER` / `DB_APP_PASSWORD`,
`ADMIN_API_KEY`, and the local course `LOCAL_COURSE_ID` / `LOCAL_COURSE_SOURCES` (where a retrieval run finds
the question bank). A variable of the same name in the environment wins over the file.

Every read of the database goes through the read-only role, so a check cannot change what it checks. The
app's role is meant only for writing test data and deleting it again; no check uses it today.

## Test data

Every state the current checks need is reached through the app itself (its configuration, its API, its input
files), so the harness writes nothing into the app's stores today. When a check needs data the app cannot be
brought to, for example recorded model answers, that data is created through `evalharness.testdata`, not by
single test files, and every name it writes starts with `evalharness_`. Removal deletes everything with that
prefix, so no course or file in `courses/` may use it (a unit test checks this).

Run the api tests against one stack at a time. Two runs at once on the same stack would ingest the same demo
course at the same time; for parallel runs, start a separate stack for each run
(`docker compose -p <name> up -d`) and point each run at its own one.

After a run that was stopped before its cleanup, remove the data by hand:

```powershell
poetry run python -m evalharness.testdata check   # list test data still in the app's stores (exit code 1 if any)
poetry run python -m evalharness.testdata clean   # delete it all, then check again
```

```powershell
$env:EVAL_SUT_BASE_URL = "http://127.0.0.1:8001"   # only if the app is not on the default address
```

## Retrieval evaluation

A run sends every question of a gold set to the app's search twice, with the app's own settings and with no
lowest score, scores each result against the question's answer section, writes a run record to `runs/` and
prints the results per exam set, chapter and knowledge level. The metrics: [METRICS.md](METRICS.md).

The gold set is built from the course's local question bank each time; only its declaration is committed
(`datasets/<course>/`). The stack must serve that course: in the root `.env`, `LOCAL_COURSE_ID` names it and
`LOCAL_COURSE_SOURCES` is the folder with its materials and question bank.

```powershell
poetry run python -m evalharness.evaluation datasets/istqb_ctfl/retrieval_gold_v0.toml
poetry run python -m evalharness.evaluation datasets/istqb_ctfl/retrieval_gold_v0.toml --sets D --include-held-out
```

Without `--sets` the run takes the sets the declaration's main result is measured on. Held-out sets run only
with `--include-held-out`, and every such run is counted. A run is refused before the first search, with exit
code 2, when the bank differs from the declared one, the stack serves another course, a set is unknown, an
answer section is not a section of the stored course, or the start of a gold question appears word for word in the
stored course text (a source that reprints the questions would let the search find the question itself). A
completed run exits with 0, whatever its numbers.


## Running the tests

```powershell
poetry run pytest           # offline tests only; api tests are deselected
poetry run pytest -m api    # api tests against the running app
```

Start the system first for `-m api`: from the repository root, `docker compose up -d --build --wait`
(see [app/README.md](../app/README.md)).
If the app is unreachable, the api tests fail with the transport error in the message; they are never skipped, so a run with no
app cannot look green.

Every request carries the harness's own `X-Request-ID`, and the app echoes it and writes it into its
log lines, so a failing observation can be found in the app's logs by its `request_id`.

## Development checks

```powershell
poetry run ruff format --check .
poetry run ruff check .
poetry run mypy .
```

CI (`.github/workflows/ci.yml`) runs the unit tests on every push to `main` and every pull request, and the api
tests against a stack started from `compose.yaml` with the demo course, followed by
`python -m evalharness.testdata check`.

The same checks run as pre-commit hooks from the repository root. Run pre-commit from a terminal with
no activated virtual environment: Poetry prefers an active one over this project's own.
