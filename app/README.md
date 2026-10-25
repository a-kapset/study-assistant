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
- Embeddings: [Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference) (TEI, CPU
  image, ONNX Runtime backend) as a separate Compose service, called through LangChain's
  [`OpenAIEmbeddings`](https://docs.langchain.com/oss/python/integrations/embeddings/index) against its
  OpenAI-compatible endpoint; the app image holds no model and no PyTorch
- PostgreSQL 18 with [pgvector](https://github.com/pgvector/pgvector), reached through
  [psycopg 3](https://www.psycopg.org/psycopg3/) and its connection pool
- Docker Compose for running the app next to its database
- [Poetry 2](https://python-poetry.org/) for dependencies and the virtual environment
- Development: [pytest](https://docs.pytest.org/) with [Hypothesis](https://hypothesis.readthedocs.io/) for property
  tests, [Testcontainers](https://testcontainers-python.readthedocs.io/) for a throwaway database in the
  integration tests, [ruff](https://docs.astral.sh/ruff/) (format and lint), [mypy](https://mypy.readthedocs.io/)
  (strict), [pre-commit](https://pre-commit.com/)

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
│   ├── migrate.py          # one-shot migration run: applies new SQL files, the embedding space's vector table, the two roles
│   ├── ingest_courses.py   # one-shot ingestion run: asks the running app to ingest every course it serves; exit 1 if any failed
│   ├── migrations/         # numbered SQL files (001_vector.sql, 002_corpus.sql, ...), applied in order
│   ├── courses/            # courses as data: nothing course-specific lives in the code
│   │   ├── models.py       # the course manifest: metadata, structure rules, source files
│   │   └── registry.py     # loads the enabled courses' manifests at startup; a bad one stops the app
│   ├── embed/              # text to vectors
│   │   ├── base.py         # embedding spaces (which model made a vector) and the Embedder protocol
│   │   └── remote.py       # the client of the embedding server: adds the prefixes, sends texts in batches, readiness ping, typed errors
│   ├── ingest/             # course sources to clean page text, the course structure and chunks, stored per course
│   │   ├── parser.py       # PDF text layer page by page, with the parser and PDF engine versions
│   │   ├── normalize.py    # header and footer lines, words split at line ends, spacing, Unicode form; page status
│   │   ├── structure.py    # sections and learning objectives found by the course's structure rules
│   │   ├── chunker.py      # chunks within sections: size cap, pages, objectives, content-derived ids, rules version
│   │   ├── service.py      # ingestion of a course: read every source, compute missing vectors, then add, keep, replace or remove documents
│   │   └── repository.py   # the corpus tables in PostgreSQL: one transaction per course, counts read back
│   ├── retrieval/          # finding the course passages closest to a question
│   │   ├── models.py       # the search result: hits with score and rank, status ok / empty, the values used
│   │   ├── repository.py   # the two search statements: a course's coverage by vectors, its closest passages
│   │   └── service.py      # vector search in one course: refuse a course not fully embedded, keep hits above the lowest score
│   ├── exams/              # a course's question bank
│   │   └── bank.py         # the common bank format (one question per JSON line) and its reader; a bad line names its number and a reason code
│   └── api/                # HTTP layer: one router factory per group of endpoints, plus middleware
│       ├── middleware.py   # request id on every request; unhandled exceptions become a 500 envelope
│       ├── health.py
│       ├── courses.py
│       ├── search.py       # POST /v1/search
│       ├── admin.py        # endpoints behind the admin key: course ingestion
│       └── config.py
├── tools/
│   └── make_embedder_reference.py  # makes the reference vectors, in a throwaway container (see "Reference vectors")
└── tests/
    ├── integration/        # the real database driver without a server: startup and readiness while the database is down;
    │                       # the corpus SQL and the migration run on a throwaway PostgreSQL (pytest -m database);
    │                       # the running embedding server against reference vectors (pytest -m embedding_server);
    │                       # data/embedder_reference.json: the reference vectors
    └── unit/               # in-process tests: the API through FastAPI's TestClient, parsing on the demo course PDF,
                            # the embedding client on a fake HTTP server
```

## Getting started

### With Docker (the app and its database)

From the repository root, with Docker running:

```powershell
Copy-Item .env.example .env                   # then set the passwords and ADMIN_API_KEY in .env
$env:GIT_COMMIT = git rev-parse --short HEAD  # optional: shown by /v1/config
docker compose up -d --build --wait
docker compose wait ingest                    # waits for the one-shot ingestion run and returns its exit code
docker compose logs migrate --no-log-prefix   # which migration files were applied
docker compose logs ingest --no-log-prefix    # what happened to each course
curl.exe -i http://127.0.0.1:8000/ready
```

The stack starts in order: the database, then `migrate`, a one-shot run of the same image that applies new
migration files as the database owner and prepares the app's role and the read-only role; next to it
`model-check` checks the model files and `embedder` loads the model; then the app, only after `migrate` has
exited without an error and `embedder` is healthy. `--wait` returns once the database, the embedding server and
the app are healthy (the app counts as healthy when `/ready` answers 200); `docker compose ps -a` shows `migrate`
and `model-check` as exited with code 0. The model files must be in place first (see "Model files" below).

Once the app is healthy, `ingest`, another one-shot run of the same image, asks the app to ingest every course
it serves (see "Ingestion"), so the stored data matches the course files after every start, also on an empty
volume; unchanged courses cost nothing. It tries every course and exits with 1 if any failed, naming the course
and the app's error code in its log. Nothing depends on it, so `--wait` neither waits for it nor sees it fail:
`docker compose wait ingest` returns its exit code (0 when every course was ingested).

`docker compose down` stops the stack and keeps the database volume; `docker compose down -v` also
deletes it, which is needed after changing `DB_OWNER` or `DB_PASSWORD`, because PostgreSQL sets the owner
only when the volume is first created. The database port is published on `127.0.0.1` only
(`DB_PUBLISHED_PORT`), for checks that read stored rows with the read-only role (`DB_READER_USER`); as the owner,
use `docker compose exec db psql -U study_assistant -d study_assistant` (the `DB_OWNER` and `DB_NAME` from `.env`).

Compose mounts the repository's `courses/` folder read-only and enables the courses in
`APP_COURSES__ENABLED` (`demo` in `.env.example`); check with `curl.exe http://127.0.0.1:8000/v1/courses`.
Course sources are mounted read-only too: `courses/` (the demo course's PDF is committed) and one more folder
for a course whose material stays on this machine (`LOCAL_COURSE_ID`, `LOCAL_COURSE_SOURCES`). The `ingest`
run stores them at startup; check with `curl.exe http://127.0.0.1:8000/v1/courses/demo`.
A changed source file is picked up by the next start. The app reads the manifests only when it starts, so after
editing a `course.yaml` restart it first; otherwise the run ingests with the old manifest and reports every
document as unchanged:

```powershell
docker compose restart app                                          # only after a manifest change
docker compose up -d --wait; if ($?) { docker compose wait ingest }
```

To ingest one course by hand, call the admin endpoint:

```powershell
$key = (Select-String '^ADMIN_API_KEY=(.*)' .env).Matches.Groups[1].Value
curl.exe -X POST -H "X-Admin-Key: $key" http://127.0.0.1:8000/v1/admin/courses/demo/ingest
```

### Saving and restoring the database

Everything stored so far is made from the migrations and the course files, so `down -v` and a start rebuild it.
A dump keeps an exact state, such as the corpus an evaluation ran on, and saves the ingestion time. Dumps go to
`backups/`, which git ignores because a dump holds the course text. The dump is written inside the database
container and copied out: in Windows PowerShell 5.1, `>` would re-encode the binary output as text.

```powershell
New-Item -ItemType Directory -Force backups | Out-Null
docker compose exec db sh -c 'pg_dump -U $POSTGRES_USER -d $POSTGRES_DB -Fc -f /tmp/db.dump'
docker compose cp db:/tmp/db.dump "backups/db-$(Get-Date -Format yyyyMMdd-HHmm).dump"
docker compose exec db rm /tmp/db.dump
```

A dump is restored into an empty volume. The migration run first creates the tables and the two roles the dump
grants rights to, so the stack must use the same role names (`DB_APP_USER`, `DB_READER_USER`); `--clean
--if-exists` then replaces those empty tables with the dump's, in one transaction:

```powershell
docker compose down -v
docker compose run --rm migrate
docker compose cp backups/<file>.dump db:/tmp/restore.dump
docker compose exec db sh -c 'pg_restore -U $POSTGRES_USER -d $POSTGRES_DB --clean --if-exists --single-transaction /tmp/restore.dump && rm /tmp/restore.dump'
docker compose up -d --wait; if ($?) { docker compose wait ingest }
```

After a restore the ingestion run reports every document as unchanged.

### Model files

The embedding server reads the files of the pinned model revision from `models/<model>/<revision>/` in the
repository root (not committed, about 490 MB). Before it starts, `model-check` compares every file with the SHA-256
committed in [`models.sha256`](../models.sha256); a missing or changed file is named in
`docker compose logs model-check` and the server does not start. The server never downloads anything itself.

Download them once from the repository root, in a throwaway container:

```powershell
docker run --rm -v "${PWD}/models:/models" python:3.14.8-slim-trixie sh -c "pip install -q huggingface-hub==2.2.0 && hf download intfloat/multilingual-e5-small 1_Pooling/config.json config.json modules.json onnx/model.onnx sentence_bert_config.json sentencepiece.bpe.model special_tokens_map.json tokenizer.json tokenizer_config.json --revision 614241f622f53c4eeff9890bdc4f31cfecc418b3 --local-dir /models/intfloat/multilingual-e5-small/614241f622f53c4eeff9890bdc4f31cfecc418b3"
```

To change the model or its revision: download the new revision's files the same way; check that the SHA-256 of
`onnx/model.onnx` equals the one Hugging Face shows for that file; write the new hashes into `models.sha256`
(`docker run --rm -v "${PWD}/models/<model>/<revision>:/m:ro" -w /m busybox:1.37.0 sha256sum <files>`); change
`APP_EMBEDDING__*` in `.env` and `.env.example`; commit both files together. The new revision is a new embedding
space, so the next ingestion computes every vector again. A new model or revision also needs new reference vectors
for the reference-vector check (see "Reference vectors" under "Development checks"). If `model-check` fails while nothing
was changed, the local files are damaged: download them again and leave `models.sha256` as it is.

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
user and password, plus `APP_SERVER_ROLE__NAME` / `APP_SERVER_ROLE__PASSWORD` for the app's role and
`APP_READER_ROLE__NAME` / `APP_READER_ROLE__PASSWORD` for the read-only role; the app server refuses these.
The app server needs `APP_ADMIN__API_KEY` (at least 16 characters) and does not start without it.

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

Then the run prepares the configured embedding space (`APP_EMBEDDING__*`): it registers the space in
`embedding_spaces` and creates its vector table `vectors_<space id>`. The table has no vector index: a search
compares the question with every vector of the course, so its result is exact and does not change with the way
PostgreSQL plans the query. A space that is already registered is left as it is; one registered with another vector
length stops the run. A new model, revision or prefix is a new space with its own table, so vectors of two models
are never mixed, and the tables of earlier spaces stay.

After that, the run creates two roles or resets their passwords, each starting from no table rights:
the app's role (`DB_APP_USER`) may select, insert, update and delete rows of every table and only read the
journal and the space registry; the read-only role (`DB_READER_USER`) may only select. The app connects as its
role, so it cannot create, alter or drop anything; only the migration run uses the owner.

Every row carries its course id; a document's sections, objectives and chunks are deleted with it, and a chunk's
vectors with the chunk.

```mermaid
erDiagram
    courses ||--o{ documents : "has source files"
    documents ||--o{ sections : "split into"
    documents ||--o{ learning_objectives : "defines"
    documents ||--o{ chunks : "cut into"
    sections |o--o{ learning_objectives : "owns"
    sections |o--o{ chunks : "holds"
    chunks ||--o| vectors_space_id : "has a vector in each space"
    embedding_spaces ||--|| vectors_space_id : "names"
    courses {
        text course_id PK
        text title
        text language
    }
    documents {
        text course_id PK
        text source_file PK
        text sha256 "edition of the file"
        text chunker_version "chunking rules and sizes"
        text rules_sha256 "manifest rules that shape the rows"
        text parser_version
        int page_count
    }
    sections {
        text course_id PK
        text source_file PK
        int position PK
        text section_id "null before the first heading"
        text parent_id
        text title
    }
    learning_objectives {
        text course_id PK
        text source_file PK
        text objective_id PK
        text section_id
        text knowledge_level
    }
    chunks {
        text chunk_id PK "derived from the content"
        text course_id
        text source_file
        int position
        text section_id
        text_array objective_ids
        text text
    }
    embedding_spaces {
        text space_id PK "model name and a hash of the four fields"
        text model
        text revision
        text passage_prefix
        text query_prefix
        int dimension
    }
    vectors_space_id {
        text chunk_id PK
        text course_id "the chunk's course"
        vector embedding "compared by cosine distance, no index"
    }
```

## Ingestion

`POST /v1/admin/courses/{course_id}/ingest` turns a course's source files into rows of the tables above.
Each link of the chain takes its rules from the course manifest; the code holds none of them:

```mermaid
flowchart LR
    M["course.yaml<br/>(manifest)"] --> P["parser.py<br/>PDF text layer"]
    P --> N["normalize.py<br/>clean page text"]
    N --> S["structure.py<br/>sections, objectives"]
    S --> C["chunker.py<br/>chunks with ids"]
    C --> E["remote.py<br/>vectors from the embedding server"]
    E --> R["service.py + repository.py<br/>one transaction per course"]
    R --> DB[("PostgreSQL")]
    M -. "sources: file, role, authority" .-> P
    M -. "noise_lines" .-> N
    M -. "section_heading, objective_id, knowledge_level" .-> S
    M -. "language" .-> C
```

Every source is read and chunked, and the missing vectors are computed, before anything is written, so a source
that cannot be read (422 `ingest_failed`, naming the file) or an embedding server that cannot be reached (503
`embedding_unavailable`) or answers with an error (502 `embedding_failed`) leaves the database as it was. Then, in
one transaction per course:

- a document whose file, parser, chunking rules and shaping manifest values are all unchanged is kept as it is;
- a changed document is deleted with all its rows and written again;
- a document the manifest no longer lists is deleted;
- every chunk gets one vector in the configured embedding space, stored in that space's table; an unchanged
  document whose chunks have no vectors there (after a change of model) gets only its vectors (outcome
  `embedded`). Vectors are deleted with their chunks.

The vectors are computed outside the transaction, so no database connection waits for the embedding server. If
another ingestion of the same course changed it in the meantime so that a chunk needs a vector that was not
computed, the ingestion stops with 409 `ingest_conflict` and nothing written; running it again is safe.

Ingesting the same files twice therefore changes nothing, and chunk ids stay the same. The answer lists what
happened to each file, the counts read back from the database, the embedding space and the number of the
course's chunks with a vector in it. Two ingestions of one course run one
after the other. Chunk length and overlap are settings (`APP_INGEST__CHUNK_SIZE`, `APP_INGEST__CHUNK_OVERLAP`);
changing them rebuilds every document on the next ingestion. Only PDF sources are ingested so far; question
banks are not chunked.

## Search

`POST /v1/search` with `course_id` and `query` returns the course's passages closest to the question by cosine
similarity, closest first, each with its section, file, pages, text, `score` and `rank`. Passages below the
lowest score are left out; when none is left the status is `empty`, otherwise `ok`. The answer repeats the
number of hits asked for (`k`), the lowest score (`min_score`) and the embedding space it used, so a client can
record what produced it. `k` and `min_score` may be sent with the request; otherwise `APP_SEARCH__DEFAULT_K`
and `APP_SEARCH__MIN_SCORE` apply, and `k` above `APP_SEARCH__MAX_K` is a 422. The default model scores almost
any two texts above 0.7, so the useful lowest scores lie in a narrow range (see `.env.example`).

A course that is not ingested, or whose chunks do not all have a vector in the configured space (after a
change of model), is refused with 409 `course_not_searchable` instead of being searched in part; ingesting it
again fixes it. An embedding server that cannot be reached is a 503, one that answers with an error a 502.

```powershell
'{"course_id": "demo", "query": "How many barges does each player receive?"}' | Set-Content search.json
curl.exe -X POST -H "Content-Type: application/json" --data-binary "@search.json" http://127.0.0.1:8000/v1/search
```

The JSON goes through a file because Windows PowerShell 5.1 removes the inner double quotes of a JSON argument
to `curl.exe`. Its `Invoke-RestMethod` also shows non-ASCII course text garbled, although the answer is UTF-8.

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness: `{"status": "ok"}` while the process is serving HTTP |
| `GET` | `/v1/config` | Effective configuration: app version and settings, every secret masked |
| `GET` | `/ready` | Readiness: 200 `{"status": "ok", "checks": {"db": "ok", "embedder": "ok"}}` when the database and the embedding server answer, otherwise 503 in the error envelope naming each failed check |
| `GET` | `/v1/courses` | The enabled courses in the configured order: `id`, `title`, `description`, `language` |
| `GET` | `/v1/courses/{course_id}` | One course, its stored `corpus` counts (documents, sections, learning objectives, chunks), the configured `embedding_space` and the number of the course's `vectors` in it (0 after a change of model until the next ingestion); 404 for a course not served |
| `POST` | `/v1/search` | The passages of a course closest to a question: `status` (`ok` / `empty`), `hits` with score and rank, the `k`, `min_score` and `embedding_space` used; 404 for a course not served, 409 for a course not fully embedded (see "Search") |
| `POST` | `/v1/admin/courses/{course_id}/ingest` | Ingest a course (needs the `X-Admin-Key` header, otherwise 401): per-file outcome, removed files, stored counts, embedding space and vector count |


## Errors and request ids

Every response carries an `X-Request-ID` header. A client may send its own id (letters, digits,
`.`, `_`, `-`, up to 128 characters) and gets it back; otherwise the app generates a UUID. The same
id is in every log line written while the request is handled, so a response can be matched with its
logs.

Every error has the status code that fits it and one body format:

```json
{"status": "error", "error": {"code": "not_found", "message": "Not Found"}, "request_id": "..."}
```

`code` is stable for programs to read: `not_found`, `method_not_allowed`, `unauthorized`, `ingest_failed`,
`ingest_conflict`, `embedding_failed`, `embedding_unavailable`,
`internal_server_error` and so on. A 422 also lists `details` with the location and reason of each invalid field; the values that were
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

A plain `pytest` needs no server and no model files. The database tests and the reference-vector check below
run only when asked for.

CI (`.github/workflows/ci.yml`) runs all of them on every push to `main` and every pull request: the checks and
unit tests, `pytest -m database`, and `pytest -m embedding_server` on a stack started from `compose.yaml` with the
demo course. It downloads the model files of `models.sha256` once and keeps them in the Actions cache.

### Database tests

`tests/integration/test_corpus_sql.py` runs the app's corpus SQL and its migration run against a real PostgreSQL
with pgvector: counts and vector counts per course, the chunks still missing a vector in a space, the lock that
keeps two ingestions of one course apart, and the app role's rights on a newly created vector table.
`tests/integration/test_embedding_space_change.py` plays a change of the embedding model: the migration run
prepares the new space, and the next ingestion adds a vector for every stored chunk there while the old space
keeps its own. `tests/integration/test_ingest_sql.py` changes a course the way an author does (a source dropped
from the manifest, a PDF saved again) and checks what ingestion deletes, also next to a course with files of the
same names. `tests/integration/test_search_sql.py` checks the search statements: the closest passages of the
asked course only, with their similarity, and a course's coverage by vectors per space. The tests start a
throwaway container of the stack's database image with Testcontainers, prepare it
with the app's own migration run and remove it at the end; the stack's database is never touched. They need
Docker running and nothing else:

```powershell
poetry run pytest -m database
```

A plain `pytest` leaves them out (marker `database`); without Docker they fail, never skip. Run them after any
change of the SQL in `ingest/repository.py` or `retrieval/repository.py`, of `migrate.py` or of a migration file.

### Reference vectors

**What they are.** `tests/integration/data/embedder_reference.json` holds the vectors that the model's reference
implementation (sentence-transformers on CPU, the pinned revision downloaded from Hugging Face) gives for five
invented texts: four passages, among them a Russian text and one longer than the model's limit of 512 tokens, and
one query. The file also names the model, the revision, the passage and query prefixes and the library versions
the vectors were made with.

**How the check uses them.** `tests/integration/test_embedding_server_reference.py` sends the same texts through
the app's own embedding client to the running embedding server, with the model, revision and prefixes named in
the file, and expects every vector to point the same way as the reference (cosine at least 0.9999). It sees what
an HTTP status cannot: a lost or wrong prefix, token ids sent instead of text, a long text cut in another place,
other model files, a changed server image or client library. It needs the stack (`docker compose up -d --wait`)
and runs only when asked for:

```powershell
poetry run pytest -m embedding_server
```

A plain `pytest` leaves it out (marker `embedding_server`); asked for without a server, it fails instead of being
skipped, so a check that did not run can never look like one that passed. Run it after any change of the
embedding client, its libraries, the server image or the model files: none of these is part of the embedding
space, so only this comparison shows that the vectors stayed the same.

**When to make new reference vectors.** Only for another model, revision or prefixes. Never because the check
fails after an upgrade of the server image, the client or its libraries: that failure is the finding, and new
reference vectors would hide it.

**How.** `tools/make_embedder_reference.py` makes the file in a throwaway container, with the library versions
pinned to those the current file was made with. From `app/`:

```powershell
docker run --rm -v "${PWD}/tools:/tools:ro" -v "${PWD}/tests/integration/data:/out" python:3.14.8-slim-trixie sh -c "pip install -q --root-user-action=ignore --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 && pip install -q --root-user-action=ignore sentence-transformers==6.1.0 transformers==5.20.1 && python /tools/make_embedder_reference.py --out /out/embedder_reference.json"
```

It downloads about 700 MB (the libraries and the model). For another model, add the values of
`APP_EMBEDDING__*` from `.env` at the end: `--model <model> --revision <revision> --passage-prefix "<prefix>"
--query-prefix "<prefix>"`. Run with the defaults, the command gives exactly the vectors in the committed file;
`git diff` on the file then shows no change. Commit the new file together with the `.env.example` change, then
run the check against the stack with the new model.

The same checks, a secret scan and a check that app code contains no course-specific names run as
pre-commit hooks from the repository root:

```powershell
app\.venv\Scripts\pre-commit install          # once, from the repository root
app\.venv\Scripts\pre-commit run --all-files
```
