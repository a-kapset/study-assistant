# Planted defects: which checks notice them

Experiment of 2026-11-19, with the request traces read on 2026-11-20.

**Question.** Thirteen defects were planted in the running app one at a time, across five of its parts. Which of
the harness's five checks notice which defect, which defects no check notices, and do the request traces point to
the part that was made faulty?

**Answer in short.** Every planted defect except two was noticed by at least one check, and no check raised a false
alarm on the run without a defect. The two defects nobody noticed are an answer model swapped for a smaller one and
a lowest score of 0: the checks of the answers cannot see either of them today. For ten defects whose part has a span
in the request traces, the traces point to the faulty part for eight.

**Decision (for the harness, not for a release).**

1. Next blind spot to close: the answer model swap. The answer run compares only which section an answer cites and
   whether it declines; a weaker model that cites the same sections looks the same. It needs a check of the answer
   itself (a judge of the answer against its passages).
2. The lowest score of 0 stays a known blind spot of the answer run until there are off-topic questions close to the
   course: the nine off-topic questions that the search normally turns away reached the model with passages, and
   the model declined all of them by itself, so nothing a learner sees changed.
3. The underpowered cells of the chunk-size mutants are not repeated now. They would be worth a larger set or more
   repetitions only when a decision about the chunk size depends on them.
4. Two planted defects are invisible in the traces: the answer step's mode and an empty reply of the provider. A span
   attribute for the answer step's mode and for an empty reply would make them visible; that is a change of the app,
   noted as a candidate.
5. Next: defects planted in the app's source code, with the same matrix.

This report is rebuilt from the run records. Every table below is filled by
[`evalharness/mutation/report.py`](../src/evalharness/mutation/report.py) from the ids of the two experiments; the
text around them is written. The records hold course material and stay on the machine that made them. There, from
`evaluation-harness/`:

```text
poetry run cog -r -U reports/mutant-experiment-2026-11.md       # fill the tables again from the records
poetry run cog --check -U reports/mutant-experiment-2026-11.md  # fails when the records give other tables
```

<!-- [[[cog
import cog
from evalharness.mutation import report
experiment = report.load("8f173c22", "d2a03189")
cog.out(report.configuration_block(experiment))
]]] -->

| Item | Value |
|---|---|
| Judged experiment | `8f173c2229aa48eb941d3b2e9e05f2bf` |
| Course | `istqb_ctfl` |
| Catalog | `mutants/istqb_ctfl.toml` (SHA-256 `b5764454f950`) |
| App commit | `48066f358da1` |
| Harness commit | `bbc8c966e397` with changes not committed |
| Embedding model | `intfloat/multilingual-e5-small` |
| Answer model | `openai/gpt-5.4-mini` |
| Runs | 15, valid 15; first 2026-11-19 16:59 UTC, last 2026-11-19 18:44 UTC |
| Traces read from | `d2a03189de4d4b78ba18f3d52e3dcbac`: 15 runs, valid 15 |

<!-- [[[end]]] -->

## How the experiment ran

A mutant is one entry of the catalog [`mutants/istqb_ctfl.toml`](../mutants/istqb_ctfl.toml): the settings it
changes, the part of the app it makes faulty, the symptom expected and the checks expected to notice it, all written
before the first run. The command `python -m evalharness.mutation` starts the app again with the mutant, proves from
the app's own reported settings that exactly the declared change is live, runs every check, then starts the app as
configured and proves that it came back. A run without a mutant goes the same way first and once more at the end:
the first is what every cell is judged against, the last shows the false alarms. A provider fault is planted by the
harness's model proxy for the whole answer run, so every model call of that run gets it.

Every cell is judged by a rule written before the first matrix (see [METRICS.md](../METRICS.md), "Mutant
experiment"): **killed** when the whole interval of the change against the run without a mutant lies beyond the
check's smallest meaningful change, **survived** when it lies within it, **underpowered** when it holds both, so the
cases at hand cannot decide. A check that refuses to measure, or whose questions all fail, kills too. A question that
failed for a reason of the stack that the mutant does not declare is left out of both runs.

### What each check counts

<!-- [[[cog cog.out(report.sample_block(experiment)) ]]] -->

| Check | Stage it measures | Counted over |
|---|---|---|
| `retrieval_gold` | search | questions finding their section: 146 |
| `paraphrase` | search | pairs whose rewording loses the section: 76 |
| `paraphrase` | search | pairs with a stable top k: 76 |
| `search_relations` | search | 2 property tests pass or fail |
| `answer_run` | answer | answerable questions answered with the gold section: 82 |
| `answer_run` | answer | unanswerable questions declined: 30 |
| `answer_run` | answer | questions whose verdict changed between asks: 112 |
| `failure_labels` | answer | questions whose asks got a failure label: 112 |
|  |  | asks per question in the answer run: 2 |

<!-- [[[end]]] -->

Missing from the sample: a second course, other languages, questions written by learners, more than one answer
model. The two asks of one question are not independent, so the answer-run cells count questions, not asks.

## The kill matrix

"Expected kill" marks a cell in which the catalog expected a kill; "blind spot" one in which it expected a kill that
did not come; "unexpected kill" a kill nobody expected. A plain "survived" was not aimed at.

<!-- [[[cog cog.out(report.grid_block(experiment)) ]]] -->

| Mutant | Faulty part | `retrieval_gold` | `paraphrase` | `search_relations` | `answer_run` | `failure_labels` |
|---|---|---|---|---|---|---|
| `lowest_score_0_88` | search | killed (expected kill) | killed (expected kill) | survived | killed (expected kill) | survived |
| `lowest_score_0_86` | search | killed (unexpected kill) | killed (expected kill) | survived | killed (expected kill) | survived |
| `default_k_1` | search | killed (expected kill) | killed (unexpected kill) | survived | killed (expected kill) | survived |
| `lowest_score_0` | search | survived | underpowered | survived | survived (blind spot) | survived |
| `embedder_unreachable` | embedding | killed (expected kill) | killed (expected kill) | killed (unexpected kill) | killed (expected kill) | killed (expected kill) |
| `query_prefix_changed` | embedding | killed (expected kill) | killed (expected kill) | killed (expected kill) | killed (expected kill) | killed (expected kill) |
| `chunk_size_200` | ingestion | underpowered (expected kill) | killed (unexpected kill) | survived | underpowered (expected kill) | survived |
| `chunk_size_4000` | ingestion | survived | underpowered | survived | underpowered (expected kill) | survived |
| `ask_ungrounded` | answer | survived | survived | survived | killed (expected kill) | survived |
| `answer_model_nano` | answer | survived | survived | survived | survived (blind spot) | survived |
| `provider_rate_limited` | provider | survived | survived | survived | killed (expected kill) | killed (expected kill) |
| `provider_server_error` | provider | survived | survived | survived | killed (expected kill) | killed (expected kill) |
| `provider_empty_output` | provider | survived | survived | survived | killed (expected kill) | killed (expected kill) |
| `baseline` | no mutant | survived | survived | survived | survived | survived |

<!-- [[[end]]] -->

## The checks

<!-- [[[cog cog.out(report.summary_block(experiment)) ]]] -->

| Check | Expected kills | False alarms | Only this check | Expected, not killed | Model calls |
|---|---|---|---|---|---|
| `retrieval_gold` | 4 of 5 | 0 of 1 | - | `chunk_size_200` (underpowered) | none |
| `paraphrase` | 4 of 4 | 0 of 1 | `chunk_size_200` | - | none |
| `search_relations` | 1 of 1 | 0 of 1 | - | - | none |
| `answer_run` | 9 of 13 | 0 of 1 | `ask_ungrounded` | `lowest_score_0` (survived), `chunk_size_200` (underpowered), `chunk_size_4000` (underpowered), `answer_model_nano` (survived) | live |
| `failure_labels` | 5 of 5; labels named 4 of 4 | 0 of 1 | - | - | none (from the answer run) |

<!-- [[[end]]] -->

What each check cannot see, from this matrix:

- `retrieval_gold` counts whether a search finds the question's section. Passages cut to 200 characters still land in
  the right section, so the cut is seen only as an underpowered cell: the labels are sections, not passages.
- `paraphrase` is the most sensitive check of search: only it kills the 200-character cut, through the share of pairs
  whose top results stay the same.
- `search_relations` (two property tests of the search) kills only when the search breaks outright. Both its kills
  come from the tests' own guard (the stack is not healthy, or a search failed), not from a broken relation.
- `answer_run` is the only check that sees the ungrounded answer mode. It cannot see the model swap or the lowest
  score of 0 (see the families below).
- `failure_labels` names every planted fault of the stack with its own label; it sees nothing while the app keeps
  answering.

## What no check noticed, as families

<!-- [[[cog cog.out(report.families_block(experiment)) ]]] -->

| Check | Faulty part | Verdict | n | Mutants |
|---|---|---|---|---|
| `answer_run` | search | survived | 1 | `lowest_score_0` |
| `retrieval_gold` | ingestion | underpowered | 1 | `chunk_size_200` |
| `answer_run` | ingestion | underpowered | 2 | `chunk_size_200`, `chunk_size_4000` |
| `answer_run` | answer | survived | 1 | `answer_model_nano` |

<!-- [[[end]]] -->

- **The answer run and the lowest score of 0.** With no lowest score, the nine off-topic questions that the search
  normally turns away get passages, but the model declined all of them by itself: the decline moved from the search
  to the model, and the learner sees the same answer. To see it, the answer run needs off-topic questions close to the course, where the model is
  tempted to answer.
- **The answer run and the answer model.** The smaller model cited the same sections and declined the same
  questions as the configured one on every question. The answer run does not judge the answer text; a judge of the
  answer against its passages would.
- **Chunk sizes on the retrieval and answer runs.** The changes go in the expected direction for 200 characters
  (more questions lose their section than gain it), but too few to tell them from no change. More questions or
  repetitions would decide them; nothing depends on it now.

## Failure labels on the planted faults

Every run with a failure label to expect, and every other run whose asks got a label. A fault reaches an ask when
the ask calls the model; nine questions are declined before any model call.

<!-- [[[cog cog.out(report.labels_block(experiment)) ]]] -->

| Run | Label expected | Asks the fault reached | Labels the asks got |
|---|---|---|---|
| `lowest_score_0_88` | - | 224 | no label 223, no_response 1 |
| `query_prefix_changed` | search_failed | 224 | search_failed 224 |
| `ask_ungrounded` | - | 224 | no label 223, no_response 1 |
| `provider_rate_limited` | rate_limited | 206 | rate_limited 206, no label 18 |
| `provider_server_error` | provider_error | 206 | provider_error 206, no label 18 |
| `provider_empty_output` | unusable_reply | 206 | unusable_reply 206, no label 18 |

| Label | Precision (asks) | Recall (asks) |
|---|---|---|
| search_failed | 224 of 224 (1.00) | 224 of 224 (1.00) |
| rate_limited | 206 of 206 (1.00) | 206 of 206 (1.00) |
| provider_error | 206 of 206 (1.00) | 206 of 206 (1.00) |
| unusable_reply | 206 of 206 (1.00) | 206 of 206 (1.00) |

<!-- [[[end]]] -->

The one `no_response` in two runs is a request the app did not answer in time, a fault of the stack that neither
mutant declares; its question is left out of both runs of the comparison. The embedding server's outage has no
labels: the answer run refused to start, since the app reported itself degraded.

## Where the request traces point

Read from a second run of the same catalog on the same app commit, with the model proxy answering from its
recordings (no paid call): each ask names the first stage of its trace that differs from every ask of the same
question in the run without a mutant, and a mutant names the stage most of its differing asks name. Ingestion has
no span in a request, so its mutants are "not attributable" by design.

<!-- [[[cog cog.out(report.attribution_block(experiment)) ]]] -->

```text
stage named by the traces, experiment d2a03189; asks naming a stage without a mutant (noise): 0
  lowest_score_0_88        search     search        right             none 52, search 172
  lowest_score_0_86        search     search        right             none 106, search 118
  default_k_1              search     search        right             none 30, search 194
  lowest_score_0           search     search        right             none 186, search 38
  embedder_unreachable     embedding  embedding     right             embedding 224
  query_prefix_changed     embedding  embedding     right             embedding 224
  chunk_size_200           ingestion  search        not_attributable  none 10, search 214
  chunk_size_4000          ingestion  search        not_attributable  none 18, search 206
  ask_ungrounded           answer     embedding     wrong             embedding 224
  answer_model_nano        answer     not_measured  not_measured      none 18, not_measured 206
  provider_rate_limited    provider   provider      right             none 18, provider 206
  provider_server_error    provider   provider      right             none 18, provider 206
  provider_empty_output    provider   generate      wrong             generate 206, none 18
  baseline                 -          none          -                 none 224
attributed to the faulty part (mutants, 95% CI): search 4 of 4 (0.51-1.00); embedding 2 of 2 (0.34-1.00); ingestion n/a; answer 0 of 1 (0.00-0.79); provider 2 of 3 (0.21-0.94); all 8 of 10 (0.49-0.94)
```

<!-- [[[end]]] -->

- **Ungrounded mode → "embedding".** In this mode the app answers without passages, and its traces have no
  embedding or search span; the first stage that differs is the missing embedding call, not the answer step where
  the mode was chosen.
- **Empty reply of the provider → "generate".** The provider answered with status 200 and no text; the trace shows
  the failure where the app read the reply.
- **Smaller answer model → "not measured".** Its calls had no recordings, and a call without a recording is refused
  by the proxy; the model's name is in no span either.

## Cost

<!-- [[[cog cog.out(report.cost_block(experiment)) ]]] -->

| Experiment | Id | Paid, USD | From recordings at list price, USD | Asks | Paid asks without a price | Model calls by source |
|---|---|---|---|---|---|---|
| judged | `8f173c22` | 2.19 | 0.00 | 3136 | 1 | fault 1442, upstream 1960 |
| traces | `d2a03189` | 0.00 | 0.90 | 3360 | 0 | cassette 732, fault 1442, miss 1228 |

<!-- [[[end]]] -->

The judged experiment was estimated at USD 2.4 before it ran; the proxy's ledger gave USD 2.19 for it, the same as
the paid sum of its records. The traces' run paid nothing: its calls were answered from recordings, refused as
missing from them, or met a planted fault. The one paid ask without a price (`ask_ungrounded`, question QA-U09)
timed out in the app: its two calls reached the provider, and what they cost is not known.

## Not covered

<!-- [[[cog cog.out(report.not_covered_block(experiment)) ]]] -->

- Fault kinds of the proxy never planted: `spend_limit`, `truncated_body`.
- Faulty parts with no mutant: none.

<!-- [[[end]]] -->

Every part the catalog can name has a mutant. Parts it cannot name yet: the reranker (off in the configured app),
the database, the by-words arm of the hybrid search, the mapping of citations to sections, the prompt. Faults never
planted: a slow or timed-out provider or embedding server, a provider spend limit, a cut-off reply, defects in the
app's source code (the next experiment).

## Defects in the app's code

A second experiment, `63f17978` (2026-11-23, app `48066f3`, catalog `mutants/istqb_ctfl_source.toml`, USD 0),
planted two defects written into the app's code and ran them through the three checks without model calls. Every
run was valid: the patched file was in the app's container under the mutant, the original after the restore, and the
working tree was unchanged.

- **The course filter dropped from the search by meaning** survived every check, as written before the run: the 146
  gold questions found the same sections (134 of 146) and the 393 distinct passages they got were all of the asked
  course. The defect is live: asked of the first course, a question about the ferry course got that course's passage
  (similarity 0.872) and a question about the moth course five of its passages (0.855-0.863), where the configured
  app finds nothing. Questions about the course's own content cannot show a missing course filter; only questions
  aimed at another course's content can. The app's integration test of the search statement on a real PostgreSQL
  (`test_nearest_returns_the_k_closest_passages_of_the_requested_course_with_their_similarity`) fails on the same
  patch: it stores a second course with the same texts and expects the asked course's chunks only.
- **The range check of cited passage numbers removed** survived every check, as written before the run. No check
  here asks the model, and in the 5,212 answer cases of earlier records the model never cited a number outside the
  passages, so an answer run would not see it either. The app's unit test of the citations is the check for it.

Decision: both defects are caught below the harness, by the app's own tests of the code they change (the citation
mapping by a unit test, the course filter by an integration test), and the harness checks of quality are not the
place for them.

## Every killed or underpowered cell

The change each cell measured against the run without a mutant, with the run's id, so that every verdict can be
traced to its record.

<!-- [[[cog cog.out(report.margins_block(experiment)) ]]] -->

| Mutant | Run | Check | Verdict | What changed |
|---|---|---|---|---|
| `lowest_score_0_88` | `305489b3` | `retrieval_gold` | killed | questions finding their section: 134 -> 73 of 146; 61 changed (+0, -61); the true change may be from -73 to -50 of 146, all of it beyond the threshold of 7.3: noticed |
| `lowest_score_0_88` | `305489b3` | `paraphrase` | killed | pairs whose rewording loses the section: 10 -> 41 of 76; 39 changed (+35, -4); the true change may be from +21 to +41 of 76, all of it beyond the threshold of 3.8: noticed; pairs with a stable top k: 30 -> 3 of 76; 27 changed (+0, -27); the true change may be from -35 to -19 of 76, all of it beyond the threshold of 3.8: noticed |
| `lowest_score_0_88` | `305489b3` | `answer_run` | killed | answerable questions answered with the gold section: 79 -> 63 of 81; 16 changed (+0, -16); the true change may be from -24 to -10 of 81, all of it beyond the threshold of 4.0: noticed; 1 question(s) left out of both runs: QA-81 |
| `lowest_score_0_86` | `a1a083d2` | `retrieval_gold` | killed | questions finding their section: 134 -> 115 of 146; 19 changed (+0, -19); the true change may be from -28 to -12 of 146, all of it beyond the threshold of 7.3: noticed |
| `lowest_score_0_86` | `a1a083d2` | `paraphrase` | killed | pairs whose rewording loses the section: 10 -> 26 of 76; 18 changed (+17, -1); the true change may be from +9 to +24 of 76, all of it beyond the threshold of 3.8: noticed; pairs with a stable top k: 30 -> 20 of 76; 10 changed (+0, -10); the true change may be from -17 to -5 of 76, all of it beyond the threshold of 3.8: noticed |
| `lowest_score_0_86` | `a1a083d2` | `answer_run` | killed | answerable questions answered with the gold section: 80 -> 75 of 82; 5 changed (+0, -5); the true change may be from -11 to -2 of 82, all of it beyond the threshold of 4.1: noticed |
| `default_k_1` | `834ebb98` | `retrieval_gold` | killed | questions finding their section: 134 -> 116 of 146; 18 changed (+0, -18); the true change may be from -27 to -11 of 146, all of it beyond the threshold of 7.3: noticed |
| `default_k_1` | `834ebb98` | `paraphrase` | killed | pairs whose rewording loses the section: 10 -> 29 of 76; 19 changed (+19, -0); the true change may be from +12 to +27 of 76, all of it beyond the threshold of 3.8: noticed; pairs with a stable top k: 30 -> 13 of 76; 21 changed (+2, -19); the true change may be from -26 to -9 of 76, all of it beyond the threshold of 3.8: noticed |
| `default_k_1` | `834ebb98` | `answer_run` | killed | answerable questions answered with the gold section: 80 -> 75 of 82; 6 changed (+0, -6); the true change may be from -10.5 to -2 of 82, all of it beyond the threshold of 4.1: noticed |
| `lowest_score_0` | `1e897689` | `paraphrase` | underpowered | pairs whose rewording loses the section: 10 -> 9 of 76; 1 changed (+0, -1); the true change may be from -6 to +0 of 76, which holds both no change and the threshold of 3.8: too few cases to decide |
| `embedder_unreachable` | `1a21ebf9` | `retrieval_gold` | killed | by refusal: stack_not_ready: the app reports 'degraded', not answering: embedder |
| `embedder_unreachable` | `1a21ebf9` | `paraphrase` | killed | by refusal: stack_not_ready: the app reports 'degraded', not answering: embedder |
| `embedder_unreachable` | `1a21ebf9` | `search_relations` | killed | failed: test_a_higher_lowest_score_returns_no_passage_under_it_and_never_more_passages, test_fewer_hits_asked_are_the_beginning_of_more_hits_asked |
| `embedder_unreachable` | `1a21ebf9` | `answer_run` | killed | by refusal: stack_not_ready: the app reports 'degraded', not answering: embedder |
| `embedder_unreachable` | `1a21ebf9` | `failure_labels` | killed | by refusal: no_answer_record: 0 answer records in the run; the answer run made none, see its cell |
| `query_prefix_changed` | `f2639646` | `retrieval_gold` | killed | questions finding their section: by errors: 146 of 146 questions gave no result |
| `query_prefix_changed` | `f2639646` | `paraphrase` | killed | pairs whose rewording loses the section: by errors: 76 of 76 questions gave no result; pairs with a stable top k: by errors: 76 of 76 questions gave no result |
| `query_prefix_changed` | `f2639646` | `search_relations` | killed | failed: test_a_higher_lowest_score_returns_no_passage_under_it_and_never_more_passages, test_fewer_hits_asked_are_the_beginning_of_more_hits_asked |
| `query_prefix_changed` | `f2639646` | `answer_run` | killed | answerable questions answered with the gold section: by errors: 82 of 82 questions gave no result; unanswerable questions declined: by errors: 30 of 30 questions gave no result; questions whose verdict changed between asks: by errors: 112 of 112 questions gave no result |
| `query_prefix_changed` | `f2639646` | `failure_labels` | killed | questions whose asks got a failure label: 0 -> 112 of 112; 112 changed (+112, -0); the true change may be from +112 to +112 of 112, all of it beyond the threshold of 5.6: noticed; labels of 224 asks: search_failed 224; expected search_failed: named |
| `chunk_size_200` | `9f53152b` | `retrieval_gold` | underpowered | questions finding their section: 134 -> 132 of 146; 8 changed (+3, -5); the true change may be from -8 to +3 of 146, which holds both no change and the threshold of 7.3: too few cases to decide |
| `chunk_size_200` | `9f53152b` | `paraphrase` | killed | pairs with a stable top k: 30 -> 13 of 76; 21 changed (+2, -19); the true change may be from -25 to -9 of 76, all of it beyond the threshold of 3.8: noticed |
| `chunk_size_200` | `9f53152b` | `answer_run` | underpowered | answerable questions answered with the gold section: 80 -> 74.5 of 82; 11 changed (+2, -9); the true change may be from -11.5 to +0 of 82, which holds both no change and the threshold of 4.1: too few cases to decide |
| `chunk_size_4000` | `7b177bad` | `paraphrase` | underpowered | pairs whose rewording loses the section: 10 -> 14 of 76; 12 changed (+8, -4); the true change may be from -2 to +11 of 76, which holds both no change and the threshold of 3.8: too few cases to decide; pairs with a stable top k: 30 -> 38 of 76; 22 changed (+15, -7); the true change may be from -1 to +17 of 76, which holds both no change and the threshold of 3.8: too few cases to decide |
| `chunk_size_4000` | `7b177bad` | `answer_run` | underpowered | unanswerable questions declined: 30 -> 29.5 of 30; 1 changed (+0, -1); the true change may be from -2.5 to +0 of 30, which holds both no change and the threshold of 1.5: too few cases to decide; questions whose verdict changed between asks: 0 -> 2 of 112; 2 changed (+2, -0); the true change may be from +0 to +7 of 112, which holds both no change and the threshold of 5.6: too few cases to decide |
| `ask_ungrounded` | `c174deb7` | `answer_run` | killed | answerable questions answered with the gold section: 80 -> 0 of 82; 80 changed (+0, -80); the true change may be from -82 to -75 of 82, all of it beyond the threshold of 4.1: noticed; unanswerable questions declined: 29 -> 0 of 29; 29 changed (+0, -29); the true change may be from -29 to -29 of 29, all of it beyond the threshold of 1.5: noticed; 1 question(s) left out of both runs: QA-U09 |
| `provider_rate_limited` | `82eb4236` | `answer_run` | killed | answerable questions answered with the gold section: by errors: 82 of 82 questions gave no result; unanswerable questions declined: by errors: 21 of 30 questions gave no result; questions whose verdict changed between asks: by errors: 103 of 112 questions gave no result |
| `provider_rate_limited` | `82eb4236` | `failure_labels` | killed | questions whose asks got a failure label: 0 -> 103 of 112; 103 changed (+103, -0); the true change may be from +96 to +108 of 112, all of it beyond the threshold of 5.6: noticed; labels of 224 asks: rate_limited 206; expected rate_limited: named |
| `provider_server_error` | `2990446a` | `answer_run` | killed | answerable questions answered with the gold section: by errors: 82 of 82 questions gave no result; unanswerable questions declined: by errors: 21 of 30 questions gave no result; questions whose verdict changed between asks: by errors: 103 of 112 questions gave no result |
| `provider_server_error` | `2990446a` | `failure_labels` | killed | questions whose asks got a failure label: 0 -> 103 of 112; 103 changed (+103, -0); the true change may be from +96 to +108 of 112, all of it beyond the threshold of 5.6: noticed; labels of 224 asks: provider_error 206; expected provider_error: named |
| `provider_empty_output` | `f0dc9480` | `answer_run` | killed | answerable questions answered with the gold section: 80 -> 0 of 82; 80 changed (+0, -80); the true change may be from -82 to -75 of 82, all of it beyond the threshold of 4.1: noticed; unanswerable questions declined: 30 -> 9 of 30; 21 changed (+0, -21); the true change may be from -25 to -15 of 30, all of it beyond the threshold of 1.5: noticed |
| `provider_empty_output` | `f0dc9480` | `failure_labels` | killed | questions whose asks got a failure label: 0 -> 103 of 112; 103 changed (+103, -0); the true change may be from +96 to +108 of 112, all of it beyond the threshold of 5.6: noticed; labels of 224 asks: unusable_reply 206; expected unusable_reply: named |

<!-- [[[end]]] -->
