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
| `/ready` answers 200 with each check when the database answers, and 503 in the error envelope naming the failed check when it does not; a bug in the check is a 500, not a 503; `/health` stays 200 while the database is down; the database opens at startup and closes at shutdown | App unit | `app/tests/unit/test_ready.py` | A fake database is switched off in one line, so every rule is checked without a server or PostgreSQL, and a failure names the rule |
| `/ready` returns 200 with the database ok from the stack in Docker | Harness API | `evaluation-harness/tests/api/test_ready.py` | Adds what the unit tests cannot see: the image, the Compose network, the password both services share and the real PostgreSQL; a wrong host or password shows up here as a 503 naming the failed check |
| `APP_*` variables configure the app, including grouped settings such as `APP_DB__HOST` | App unit | `app/tests/unit/test_settings.py` | A renamed setting, prefix or group separator would otherwise fall back to its default without any error |
| A missing database password stops startup and names the setting | App unit | `app/tests/unit/test_settings.py` | A password with a default would only show up later as a failed login in the pool's log |
| An unknown `APP_*` variable stops startup and is named; names inside setting groups are recognised | App unit | `app/tests/unit/test_settings.py` | Pure name logic: the test passes any environment as a dictionary, no process start needed |
| Settings named like keys, tokens, passwords or DSNs are typed as secrets | App unit | `app/tests/unit/test_settings.py` | A rule over the settings model: a plain-text secret is caught when it is declared, before any endpoint can show it |
| `/v1/config` shows the settings the app was built with and its version, and never a secret | App unit | `app/tests/unit/test_config.py` | Only in-process can the app be built with a stand-in secret; a unique canary value is searched for in the whole response |
| `/v1/config` keeps its contract in the running app | Harness API | `evaluation-harness/tests/api/test_config.py` | Adds the server's own startup path (environment, settings, app) and pins the shape that evaluation runs will record |
| `/v1/courses` lists the enabled courses in the order the settings give them, with only their id, title, description and language | App unit | `app/tests/unit/test_courses.py` | The app is built in-process with manifests in a temporary folder, so the order and the response contract are checked without a server or real course files |
| A course that cannot be loaded stops startup, and the message names the course, the file and the reason: missing file, broken YAML, unknown key, broken regular expression, no sources, or an id that differs from its folder | App unit | `app/tests/unit/test_courses.py` | Each broken manifest is one small file in a temporary folder; the message is all a person sees at startup, so its content is checked case by case |
| `APP_COURSES__ENABLED` is a comma-separated list of safe, unique course ids | App unit | `app/tests/unit/test_courses.py` | Pure validation; an id from the environment becomes part of a file path, so unsafe and repeated ids are refused before any file is opened |
| Every course manifest committed in `courses/` loads | App unit | `app/tests/unit/test_courses.py` | Runs the app's own loader over the repository's files, so a broken edit fails the test run instead of someone's startup |
| Every error answers with its exact status in one envelope: 404, 405 with `Allow`, 422 naming the invalid fields but not their values, 500 without the exception text | App unit | `app/tests/unit/test_errors.py` | Only in-process can routes that fail on purpose be added to the real app; a unique canary value shows that input and exception text never reach the body |
| A well-formed client request id is kept, a malformed one is replaced by a new UUID4, and every request gets its own | App unit | `app/tests/unit/test_request_id.py` | Pure format logic, milliseconds per case; each malformed value targets one way the check could become looser |
| An unhandled exception is logged once, as JSON, with the request id and the traceback | App unit | `app/tests/unit/test_request_id.py` | Needs the process's own output and a route that raises; this log line is how a 500 is traced back to its cause |
| Unknown path and wrong method answer 404 and 405 (with `Allow`) in the envelope, and the harness's request id comes back | Harness API | `evaluation-harness/tests/api/test_error_envelope.py` | Adds the real server path: headers and body as a client receives them, and the id by which every later run result is matched with the app's logs |
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
