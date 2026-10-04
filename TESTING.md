# Testing strategy

Every check lives at the level where it is most useful: the cheapest place to run it, the closest to
where the property actually lives, and the most precise about what broke. This file lists the
principles behind the checks, the levels, how a level is chosen, and every check that exists today
with the reason for its level.

## Principles

These rules come from three practitioner books on evaluating AI systems (see Sources at the end) and
shape every check below. "Not yet" means the part of the harness that applies the rule does not exist
yet.

| Principle | What it means here | Today |
|---|---|---|
| A check feeds a decision | Every evaluation run ends in a decision (pass, block or invalid) against gate rules written down before the run. A metric that changes no decision is labelled diagnostic. | Not yet |
| One run proves nothing | Quality is estimated from repeated runs and reported with a confidence interval; a gate uses the lower bound, and a difference too small to tell apart from noise is reported as underpowered. Replayed runs in CI catch regressions; they are not evidence of quality. | Not yet |
| Cheapest reliable check first | Exact and schema checks run first, then reference metrics, then a model as a judge. A judge is never used where an exact check can decide. | Only exact checks exist |
| Expected behaviour, not expected text | Open-ended answers are checked against what they must and must not contain. Exact matching is kept for things that really are exact, such as an exam answer key. | Not yet |
| Every result carries its versions | A run records the app commit, configuration, models, prompts, corpus, datasets and checks it used. Two runs are compared only when these match; otherwise the difference is shown. | Not yet |
| A run is checked before it is read | Infrastructure failures (timeouts, rate limits, empty responses, budget stops) are classified before any quality check. A run that did not exercise the system as declared is marked invalid and never feeds a gate. | Transport failures are recorded as data, never raised |
| Failed requests still count | Infrastructure failures stay out of quality statistics, but every gate also reads the reliability numbers, so good answers on the requests that succeeded cannot hide the ones that failed. | Not yet |
| Averages hide failures | Cases carry slice tags (course, section, question type, language) and a risk level. Reports show each slice with its size; critical slices have their own gates. Hard blockers, such as a leaked secret, fail a gate whatever the average. | Not yet |
| Datasets are governed | Each dataset has a version, a hash, a split (development or hold-out) and an origin (written, synthetic or public). The hold-out is never used for tuning, and public material is marked because a model may have seen it during training. | Not yet |
| The suite remembers | A failure found once becomes a permanent case, recorded in the dataset's changelog. | Not yet |
| Checks are tested too | Defects are injected on purpose to measure which checks catch them. A model used as a judge is compared with human labels before its scores count. | Not yet |

## Levels


| Level | Where | What belongs here | Why here |
|---|---|---|---|
| App unit and property | `app/tests/unit` | Pure logic with direct access to code: parsing, validation, chunking, ranking, scoring, prompt assembly | Milliseconds per case, so property-based tests can try thousands of inputs; a failure names the function |
| App integration | `app/tests/integration` (not yet) | Behaviour that lives in the database: migrations, vector and full-text queries, access rights | Mocking the database would test the mock |
| Harness unit | `evaluation-harness/tests/unit` | The harness's own logic: settings, observations, transport, later oracles and statistics | The harness is code too; a wrong oracle gives wrong verdicts about the app |
| Harness API | `evaluation-harness/tests/api` | The HTTP contract as a client sees it: status codes, schemas, error envelope, degraded modes, sequences of calls | Runs against the real process, exactly as a client uses it |
| Harness evaluation | `evaluation-harness` (not yet) | Retrieval and answer quality on fixed datasets: reference answers, relations between related inputs, model-graded rubrics, repeated runs with confidence intervals, cost and latency | Quality belongs to the whole system on a fixed corpus and pinned models; it needs datasets and statistics, not assertions on one call |
| Harness experiments | `evaluation-harness` (not yet) | Defects injected on purpose (bad configuration, failing dependencies, patched code) to see which checks catch them | Shows that the checks have power, not only that they pass |
| Gates | pre-commit; CI (not yet) | Format, lint, types, secret scan, course-name check; later the test levels above | Stops a known-bad change before it is shared |

## Choosing a level

Ask in this order and stop at the first yes:

1. Can the property be checked on a function without I/O? → app unit.
2. Does it live in the database or another real dependency? → app integration.
3. Is it a promise the HTTP API makes to its clients? → harness API.
4. Is it about quality, or does it need a dataset, repeated runs or a model as a judge? → harness evaluation.
5. Is the question "would we notice if this broke"? → harness experiment.

The same property may be checked at two levels when each catches something the other cannot; the
table below then says what each one adds.

## How the harness reaches the app

- Over HTTP, through configuration and environment variables it owns, and through an LLM proxy set as
  the app's provider base URL.
- Directly in the app's data stores when that is cheaper or more informative. Reads are free. Writes
  only set up a test or inject a fault: through the API when the API can create the state, inside the
  test's own scope, cleaned up afterwards, and never in the frozen corpus an evaluation measures.
- Never by importing app code: a check that needs internals belongs in the app's tests.

## Checks today

| Check | Level | Where | Why this level |
|---|---|---|---|
| `/health` returns 200 `{"status": "ok"}` | App unit | `app/tests/unit/test_health.py` | In-process, no server: catches a broken route or response model on every commit |
| `/health` returns 200 `{"status": "ok"}` from the running app | Harness API | `evaluation-harness/tests/api/test_health.py` | Adds what the unit test cannot see: the app starts and serves over the network; every later harness check relies on it |
| `EVAL_*` variables configure the harness | Harness unit | `evaluation-harness/tests/unit/test_settings.py` | A renamed setting would otherwise fall back to its default without any error |
| An observation is either a response or a transport error, never both or neither | Harness unit | `evaluation-harness/tests/unit/test_observation.py` | A contradictory record would mislead every verdict built on it; checking the model is instant |
| The transport returns failures as data, classifies them, sends its own request id and keeps a base-URL path | Harness unit | `evaluation-harness/tests/unit/test_transport.py` | Simulated HTTP makes every failure kind reproducible without a network |
| No course-specific names in app code and app tests | Gate (pre-commit) | `.pre-commit-config.yaml` | A text search is enough; before every commit it costs nothing |
| No secrets in commits | Gate (pre-commit) | `.pre-commit-config.yaml` (gitleaks) | The repository is public: a leak must stop before it is pushed |
| Format, lint and strict types in both projects | Gate (pre-commit) | `.pre-commit-config.yaml` (ruff, mypy) | Seconds per run, with the exact file and line |

How to run each project's tests: [app/README.md](app/README.md), [evaluation-harness/README.md](evaluation-harness/README.md).


## Sources

- A. Mohanna, I. Kar, Z. Ralte. *Practical LLM Evaluation for Production Systems*. Packt, 2026.
- L. Nassery. *AI Model Evaluation*. Manning, 2026.
- J. Arbon. *Testing AI: Engineering Confidence in Non-Deterministic Systems*. Online draft, 2026.
