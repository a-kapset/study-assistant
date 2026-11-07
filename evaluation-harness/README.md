# Evaluation harness

Evaluates the study assistant from the outside: it calls the running app over HTTP and records every
exchange as an observation, so a failed request is data for a test, never an exception. Package: `evalharness`.

How checks are split between the app's own tests and the harness: [TESTING.md](../TESTING.md).
How every metric the harness reports is computed and how to read the metrics together: [METRICS.md](METRICS.md).
Which metamorphic relations are checked, what they change and what they found: [RELATIONS.md](RELATIONS.md).

> **Status: work in progress.** Both the app and the evaluation harness are under active development;
> the READMEs of each part describe what already works.

## Stack

- Python 3.14, [httpx2](https://github.com/pydantic/httpx2) (async HTTP client),
  [pydantic](https://docs.pydantic.dev/) and pydantic-settings, [structlog](https://www.structlog.org/) (JSON logs),
  [psycopg 3](https://www.psycopg.org/psycopg3/) for reading the app's database directly
- [Starlette](https://www.starlette.io/) and [uvicorn](https://www.uvicorn.org/) for the LLM proxy
- [SciPy](https://scipy.org/) for the confidence intervals of the evaluation runs (dependency group `eval`, so the
  proxy image does not carry it); `scipy-stubs` for its types
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
├── RELATIONS.md            # the metamorphic relations: transformation, relation type, results
├── datasets/               # per course: retrieval gold set declarations (questions stay local), Q&A sets, paraphrase pairs
├── runs/                   # the proxy's ledger and, per kind of run, its records, e.g. runs/retrieval/ (not committed)
├── cassettes/              # recorded replies of invented courses (committed)
├── Dockerfile              # the image of the LLM proxy
├── compose.proxy.yaml      # puts the proxy between the app and its model provider
├── src/evalharness/
│   ├── settings.py         # where the system under test is (EVAL_*), and the values shared with the stack's .env
│   ├── corpus_db.py        # direct reads of the app's stored corpus, as the read-only database role
│   ├── datasets/           # the retrieval gold set (built from a local question bank), the Q&A set and paraphrase
│   │                       #   pair files and their checks
│   ├── oracles/            # scores one search or one answer against the gold section, and sums up a run
│   ├── evaluation/         # the retrieval, answer and paraphrase runs: checks, requests, run records, commands
│   ├── relations/          # metamorphic relations: how a pair of inputs is scored and a relation judged
│   ├── stats/              # confidence intervals, target verdicts, and the change between two runs
│   ├── proxy/              # the LLM proxy: ledger, recordings, prices; one module per provider API in surfaces/
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
| `EVAL_PROXY_LEDGER_FILE` | `runs/proxy_ledger.jsonl` | The ledger of the LLM proxy on this machine, read by the api tests |

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
lowest score, scores each result against the question's answer section, writes a run record to `runs/retrieval/` and
prints the results per exam set, chapter and knowledge level, each recall with its 95 % interval. The metrics
and the intervals: [METRICS.md](METRICS.md).

The gold set is built from the course's question bank each time; only its declaration is committed
(`datasets/<course>/`). The stack must serve that course: in the root `.env`, `LOCAL_COURSE_ID` names it and
`LOCAL_COURSE_SOURCES` is the folder with its materials and question bank. The values of `.env.example` mount the
invented `lumen_moths` course, whose material and bank are committed, so its gold set runs without anything from
outside the repository; a real course's bank stays on the local machine.

```powershell
poetry run python -m evalharness.evaluation datasets/lumen_moths/retrieval_gold_v0.toml
poetry run python -m evalharness.evaluation datasets/istqb_ctfl/retrieval_gold_v0.toml --sets D --include-held-out
poetry run python -m evalharness.evaluation datasets/lumen_moths/retrieval_gold_v0.toml --require-target
```

Without `--sets` the run takes the sets the declaration's main result is measured on. Held-out sets run only
with `--include-held-out`, and every such run is counted. A run is refused before the first search, with exit
code 2, when the bank differs from the declared one, the stack serves another course, a set is unknown, an
answer section is not a section of the stored course, or the start of a gold question appears word for word in the
stored course text (a source that reprints the questions would let the search find the question itself). A
completed run exits with 0, whatever its numbers. The target verdict reads the interval: "met" when even its lower
end reaches the target, "not confirmed" when only the measured value does, "below" otherwise; with
`--require-target` the run exits with 1 unless the verdict is "met". CI runs the `lumen_moths` gold set that way: its number is a regression signal for
search, not the app's quality, because a language model wrote that course and its questions.

Two recorded runs of the same gold set are compared question by question, for example before and after a source
was added to the corpus:

```powershell
poetry run python -m evalharness.evaluation.compare runs/retrieval/<before>.json runs/retrieval/<after>.json --factor corpus
poetry run python -m evalharness.evaluation.compare runs/retrieval/<before>.json runs/retrieval/<after>.json
```

`--factor` (`corpus`, `embedding_space`, `k`, `min_score`) names the one recorded value changed on purpose; leave it
out for a repeated run or a change of the app's code, which a record does not hold. The comparison is refused, with
exit code 2, when the runs measured other questions, when any other recorded value differs as well, or when the gold
declaration has no `[comparison] min_meaningful_change`. It prints the change of recall with its interval, the
questions gained and lost, McNemar's exact p and a verdict (better, worse, no meaningful change, underpowered,
unchanged), and the change of MRR with its interval: [METRICS.md](METRICS.md#comparison-of-two-runs).

## Answer evaluation

A run asks every question of a course's Q&A set through `/v1/ask`, first searching for it with the app's own
settings, and scores facts of each answer: whether the app answered or declined and why, and whether the
citations include the question's gold section. Questions without a gold section are ones the course does not
answer; an answer to one of them is a blocker. The answer text itself is not scored. The run writes a record to
`runs/answers/` and prints where the answerable questions went wrong (split by whether search found the gold
section), how the others were declined, the cost, and the results against the targets in the set's file, each
share with its 95 % interval and judged by it, like the retrieval target. The metrics: [METRICS.md](METRICS.md).

The set is committed (`datasets/<course>/qa_v1.toml`; `qa_v0.toml` is the first, smaller set): its questions
are written for it, not taken from course material. Each question lists its key terms; before a set is used,
check it against the stored course, which needs the stack but asks no question:

```powershell
poetry run python -m evalharness.datasets.check_qa_set datasets/istqb_ctfl/qa_v1.toml
```

It fails when a term of an answerable question is missing from its gold section, or a term of an unanswerable
question occurs anywhere in the course.

Run the set on the stack started with the LLM proxy, serving that course. The replies quote course passages,
so for a course whose material is not committed set `EVAL_PROXY_CASSETTES=./evaluation-harness/cassettes_local`
in the root `.env`; record once with `EVAL_PROXY_MODE=record` (paid), then run in `replay` for free.

```powershell
poetry run python -m evalharness.evaluation.answer_run datasets/istqb_ctfl/qa_v1.toml
poetry run python -m evalharness.evaluation.answer_run datasets/istqb_ctfl/qa_v1.toml --reps 3
```

`--reps n` asks every question n times. Each repetition carries its number in the request id (`<id>.rep2`), the
app passes the id on, and the proxy keeps a recording per repetition, so the model answers each one anew; the
first repetition uses the recordings of a run without `--reps`. Every repetition gets its own results and target
verdicts, followed by the flip rate: the share of questions whose verdict was not the same in all repetitions,
with the list of those questions.

A run is refused before the first question, with exit code 2, when a gold section is not a section of the stored
course. A completed run exits with 0, whatever its numbers. A question that got no answer at all (for example a
recording that no longer matches the request, after a change of the prompt, the passages or the model) makes the
run invalid: it is reported with the reasons and gets no target verdict; record again and compare.


## Metamorphic relations

A relation changes the input in a controlled way and checks that a stated relation between the two outputs holds,
where no single right output exists. The relations, their transformations and their latest results:
[RELATIONS.md](RELATIONS.md); the formulas: [METRICS.md](METRICS.md#metamorphic-relations-on-search).

The paraphrase relation searches every answerable question of a Q&A set and its rewording without the question's
key terms (`datasets/<course>/paraphrase_v0.toml`). A pair whose rewording still contains a key term, or whose
question is unknown or unanswerable, is refused when the file is read. Before the pairs are used, check that none
asks what a held-out question of the local question bank asks. The check compares meaning: it embeds every text
with the stack's embedding server (published on this machine at `EMBEDDER_PUBLISHED_PORT`) and flags a text closer
to a held-out question than any two questions of the set from different chapters; a flagged pair is read and, when
it asks the same thing, left out of the pair file. The overlap a pair needs to count as stable is calibrated on
searches with known answers (unchanged questions, different questions of one chapter):

```powershell
poetry run python -m evalharness.datasets.check_paraphrases datasets/istqb_ctfl/paraphrase_v0.toml datasets/istqb_ctfl/retrieval_gold_v0.toml
poetry run python -m evalharness.evaluation.calibrate_overlap datasets/istqb_ctfl/paraphrase_v0.toml
poetry run python -m evalharness.evaluation.paraphrase_run datasets/istqb_ctfl/paraphrase_v0.toml
```

The run needs the stack serving the course and calls no model. It searches both texts of every pair with the app's
own settings and again with no lowest score, writes a record to `runs/relations/`, and prints the counts per pass
and slice, the answer-kept relation (change of recall with its interval, McNemar's p, verdict) for both passes,
the top-k stable relation on the pass without a lowest score, and the pairs that lost their answer. A run is
refused, with exit code 2, when a gold section is not a section of the stored course; a completed run exits with 0
whatever its verdicts.

## Running the tests

```powershell
poetry run pytest           # offline tests only; api tests are deselected
poetry run pytest -m api    # api tests against the running app
```

Start the system first for `-m api`, from the repository root (see [app/README.md](../app/README.md)):
`docker compose -f compose.yaml -f evaluation-harness/compose.proxy.yaml up -d --build --wait`.
If the app is unreachable, the api tests fail with the transport error in the message; they are never skipped, so a run with no
app cannot look green.

Every request carries the harness's own `X-Request-ID`, and the app echoes it and writes it into its
log lines, so a failing observation can be found in the app's logs by its `request_id`.

## LLM proxy

The app calls the proxy instead of the model provider, and the proxy holds the real key. Every call adds one line
to `runs/proxy_ledger.jsonl` with the request id, tokens, latency and cost. `EVAL_PROXY_MODE` in the root `.env`
chooses the mode:

- `replay` answers only from recordings and never pays. A call without a recording gets 404 `replay_miss`.
- `record` answers from a recording when there is one; a new call goes to the provider, and its reply is kept.

To record again, delete the recording: its file name is the `key` in the ledger. Replies quote the course, so a
course whose material is not committed uses `EVAL_PROXY_CASSETTES=./evaluation-harness/cassettes_local`.

### What a recording is for

Only `POST /v1/ask` calls the model, so only its checks use recordings. `POST /v1/search` calls no model: the
embedding server and the exact search give the same result every time, and its checks run against it directly.

A recording replaces the model with a fixed reply. Everything else on the path of a question runs for real:

```text
question → search → request to the model → [model = recording] → reading the reply → citations, abstain reason
         → API response → harness oracle
```

The key of a recording is the SHA-256 of the whole request body (instructions, passages, question, model, reply
format), so a replay catches three kinds of change:

- **before the model** — the request is no longer the recorded one (a changed prompt, other passages found, another
  model): the proxy answers `replay_miss`, the run is invalid, and the recording has to be made again;
- **after the model** — our handling of a known reply broke: for example citation numbers mapped to the wrong
  passage, the decline reason lost, a field missing from the response;
- **in the harness** — the same replies must give the same scores; other numbers mean the scoring changed.

A `replay_miss` also follows when the question was never recorded, or when the stack mounts another recordings
folder than the one the run was recorded into (`EVAL_PROXY_CASSETTES`).

A recording run measures the model and the app together, at the time of recording. A replay checks only our code
around the model: it says nothing about how good or stable the model is, or how it answers a new request.

Recordings are used by the api test that answers a recorded demo question (committed in `cassettes/`, run in CI)
and by the answer evaluation of a course (for a course whose material is not committed, in `cassettes_local/`,
run locally).

## Development checks

```powershell
poetry run ruff format --check .
poetry run ruff check .
poetry run mypy .
```

CI (`.github/workflows/ci.yml`) runs the unit tests on every push to `main` and every pull request, and the api
tests against a stack started with the LLM proxy in replay mode and the two invented courses, followed by the
`lumen_moths` retrieval run with `--require-target` and `python -m evalharness.testdata check`.

The same checks run as pre-commit hooks from the repository root. Run pre-commit from a terminal with
no activated virtual environment: Poetry prefers an active one over this project's own.
