# Metamorphic relations

A metamorphic relation checks a system where there is no single right output: the input is changed in a
controlled way (the **transformation**), and a stated relation between the two outputs must hold (the **relation
type**: invariance, ranking stability, monotonicity, …). Each relation below names its transformation, what must
stay true, what variation is accepted, how it is judged, and its latest results. Formulas:
[METRICS.md](METRICS.md#metamorphic-relations-on-search).

| # | Transformation | Relation type | System part | Since |
|---|---|---|---|---|
| R1 | Reword a question without its course term | Invariance: the gold section stays in the top k | `/v1/search` | 2026-11-07 |
| R2 | Same as R1 | Ranking stability: the top k stays mostly the same | `/v1/search` | 2026-11-07 |
| R3 | Ask for more hits | Subset consistency: the smaller result is the beginning of the larger one | `/v1/search` | 2026-11-15 |
| R4 | Raise the lowest score | Monotonicity: no passage under the score and never more passages; by meaning alone, exactly the earlier passages that reach it | `/v1/search` | 2026-11-15 |

## R1, R2 — reworded question on search

**Transformation.** Answerable questions of the first course's Q&A set
([`datasets/istqb_ctfl/qa_v1.toml`](datasets/istqb_ctfl/qa_v1.toml)) are reworded the way a learner asks who does not
know the course's term yet: same meaning, same gold section, none of the question's key terms. "How does boundary
value analysis work?" becomes "How do I pick test inputs at the edges of the allowed ranges?". The rewordings are
in [`datasets/istqb_ctfl/paraphrase_v0.toml`](datasets/istqb_ctfl/paraphrase_v0.toml); a rewording that still
contains a key term is refused before the run. Of the 82 answerable questions, 6 are left out because they ask what
a held-out question asks (see "Before it is used"), so 76 pairs remain. In 59 the question contained a key term,
which the rewording removed; 17 were already in plain words, and the rewording only changed them. The results are
given for both groups.

**What must stay true, and what may vary.**

- **R1, answer kept (invariance).** When a question finds its gold section among the first k passages, its
  rewording finds it too. The rank and the scores may change. Judged on the primary pass (the app's own number of
  hits and lowest score, as a learner gets them) by the change of recall from the questions to the rewordings,
  question by question, with its 95 % interval: **holds** when the interval stays within the smallest meaningful
  change (0.05) or lies above zero, **violated** when the whole interval lies below zero, **not confirmed**
  otherwise.
- **R2, top k stable (ranking stability).** The two searches return mostly the same passages: a pair is stable
  when they share at least 2 of their 5 passages, in any order (calibrated, see below; 3 of 5 before the
  calibration). Judged on the diagnostic pass (no lowest score; the lowest score cuts the two lists to different
  lengths and would mix its own effect in) by the share of stable pairs: **holds** when the lower end of its Wilson
  interval reaches 0.7, **not confirmed** when only the share does, **violated** otherwise.

The rules (0.05, 3 of 5, 0.7) were written into the pair file before the first run. The overlap was then
calibrated on searches with known answers, not on the pairs' results, and set to 2 of 5; 0.05 and 0.7 stay as
judged.

**Calibration of the overlap.** Two groups of searches, both without a lowest score (76 pairs):

| Group | What it is | Shared passages of 5 → pairs |
|---|---|---|
| Same question, no change of meaning | each question against itself in lower case without punctuation | 2 → 2, 3 → 18, 4 → 39, 5 → 17 |
| Different questions, nearby topic | each question against the next question of its chapter from another section | 0 → 73, 1 → 2 |

No pair of different questions shared 2 passages, and every unchanged question did, so a pair that shares 2 or more
shares more than topic alone gives. The rule is the smallest overlap no pair of different questions reached (the
same, 2 of 5, on the first 82 pairs). The first group also shows the noise of the search itself: a change of letter
case and punctuation alone moves at least one of the 5 passages for 59 of 76 questions, and never lost the gold
section. Rerun:

```powershell
poetry run python -m evalharness.evaluation.calibrate_overlap datasets/istqb_ctfl/paraphrase_v0.toml
```

**What it catches.** A search that follows the course's words instead of the meaning: the gold-set questions
mostly use the course terms, so recall on them cannot see it. A lowest score set too high: a reworded question
matches its passage less closely and falls under it. The diagnostic pass tells the two apart.

**Before it is used.** No question or rewording may ask what a held-out question of the local question bank asks:
the held-out questions are kept for a final measurement, and these pairs will be used to choose search settings.
The pairs were written by someone who had seen the bank, so the check compares meaning, not words. Every text and
every held-out question is embedded by the stack's model, and a text is flagged when it is closer to a held-out
question than any two questions of the set from different chapters are to each other (a line measured on every
run). A flagged pair is read by a person: when it asks the same thing it is left out of the pair file, otherwise
it is listed under `cleared` with the reason.

On 2026-11-07, 164 texts of the first 82 pairs against 40 held-out questions, line 0.929: 7 flagged, all questions,
none of the rewordings. Read and left out: QA-39 (0.985, D-9), QA-21 (0.968, D-1), QA-13 (0.961, D-27), QA-20
(0.956, D-40), QA-75 (0.954, D-33), QA-76 (0.938, D-35). Cleared: QA-38 (0.943, D-9; it asks how the lifecycle
model changes testing, D-9 which practices fit every lifecycle). A check by four-word sequences, used first, had
found none of them (its highest share was QA-50 against D-15, 0.33). The line flags only what is closer than any two
different questions, so a retold question that is a little less close passes: QA-50 against D-15, which asks the
opposite of the same thing, scored 0.916.

```powershell
poetry run python -m evalharness.datasets.check_paraphrases datasets/istqb_ctfl/paraphrase_v0.toml datasets/istqb_ctfl/retrieval_gold_v0.toml
poetry run python -m evalharness.evaluation.paraphrase_run datasets/istqb_ctfl/paraphrase_v0.toml
```

### Results, 2026-11-07

First course, 76 pairs, k = 5, lowest score 0.83, embedding model `multilingual-e5-small`, stable = at least 2 of 5
shared (run `bd8fac87`). The search does not sample: a second run of the first pairs gave the same result for every
pair.

| | Recall of questions | Recall of rewordings | Kept | Lost | Gained | Stable |
|---|---|---|---|---|---|---|
| Primary pass, all 76 | 0.974 | 0.868 | 64 | 10 | 2 | 30 |
| — key term removed (59) | 0.983 | 0.864 | 50 | 8 | 1 | 21 |
| — no key term in question (17) | 0.941 | 0.882 | 14 | 2 | 1 | 9 |
| Diagnostic pass, all 76 | 0.974 | 0.882 | 65 | 9 | 2 | 30 |

- **R1 — violated.** Recall change −0.105, 95 % interval −0.197 to −0.026; 10 lost, 2 gained, McNemar p = 0.039.
  Without a lowest score: −0.092 (−0.184 to −0.013), 9 lost, p = 0.065; the verdict reads the interval, which lies
  below zero. So 9 of the 10 losses come from the embedding search itself and 1 from the lowest score: for QA-50 the
  rewording's best passage scored 0.8299 and the gold passage (second) 0.828, just under 0.83, so the learner would
  get no passage at all.
- **R2 — violated.** 30 of 76 pairs stable, 0.395 (0.292 to 0.507) against 0.7. The answer step reads these
  passages, so for most reworded questions it reads mostly different evidence: 46 of 76 rewordings share at most
  one passage with their question, as little as two different questions of one chapter.
- **Reading.** Search works on the course's words more than on the meaning; a learner who asks without the term
  loses the answer about one time in seven (10 of the 74 questions that found it) and gets other passages most of
  the time. Lost pairs: QA-01, 04, 18, 28, 48, 50, 64, 66, 77, 78. A lost pair means the gold section is not among the 5, not that no useful passage is: the
  gold label is one section per question, so a passage of a neighbouring section that also answers counts as a miss.
- **First runs, 82 pairs, before six were left out** (runs `81a46d7b`, `07568a28`, `55eb9301`): R1 −0.110 (−0.195
  to −0.037), 11 lost (also QA-13), 2 gained; R2 with 3 of 5 shared 22 of 82 = 0.268 (0.184 to 0.373), with 2 of 5
  32 of 82 = 0.390 (0.292 to 0.498). Both violated, as now.

### Results by search variant, 2026-11-14

The same 76 pairs under three search settings of one app commit, k = 5, lowest score 0.83: search by meaning
(`vector`, run `7f0b156c`, the same numbers as on 2026-11-07), by meaning and by words (`hybrid`, run `7da89ac5`),
and by meaning with a cross-encoder reranker (run `9642f7d2`). What was compared, and what would change the app's
default, was written down before the first of these runs: the recall of the rewordings, as one of four planned
comparisons, so each verdict reads a 98.75 % interval.

| Search | Recall of questions | Recall of rewordings | Kept | Lost | Gained | R1, change of recall (95 % interval) | R2, stable pairs (95 % interval) |
|---|---|---|---|---|---|---|---|
| `vector` | 0.974 | 0.868 | 64 | 10 | 2 | −0.105 (−0.197 to −0.026), violated | 30 of 76, 0.395 (0.292 to 0.507), violated |
| `hybrid` | 1.000 | 0.868 | 66 | 10 | 0 | −0.132 (−0.224 to −0.066), violated | 33 of 76, 0.434 (0.329 to 0.546), violated |
| `vector` with reranker | 1.000 | 0.855 | 65 | 11 | 0 | −0.145 (−0.237 to −0.079), violated | 40 of 76, 0.526 (0.416 to 0.635), violated |

- **`hybrid` against `vector`, rewordings:** change 0.000 (98.75 % interval −0.079 to +0.079), 3 gained (QA-28, 64,
  66), 3 lost (QA-15, 45, 80): underpowered.
- **Reranker against none, rewordings:** change −0.013 (−0.092 to +0.053), 2 gained (QA-66, 77), 3 lost (QA-30,
  45, 80): underpowered.
- **Reading.** Neither variant repairs the relation: each finds a few of the lost rewordings and loses as many
  others, so about one reworded question in seven still misses its section. Both find every question asked in
  the course's own words (76 of 76 against 74), which makes R1 look worse for them: the gap between the two
  wordings widens while the rewordings stay where they were. More pairs keep their passages with the reranker
  (40 against 30), but that number was not one of the planned comparisons; it is a lead for a later planned
  check, not a result. The default stays `vector` without the reranker.

**Planted defects (mutants)**, on the first 82 pairs with the first rule (3 of 5). Each one run once on the stack,
then the stack was restored.

| Mutant | How it was planted | R1 primary | R1 diagnostic | R2 | Seen? |
|---|---|---|---|---|---|
| Lowest score 0.86 instead of 0.83 | `APP_SEARCH__MIN_SCORE=0.86` for the app | −0.366 (−0.476 to −0.256), 31 lost | −0.098, unchanged | 0.268, unchanged | Yes: the loss triples, while recall of the questions only falls from 0.976 to 0.927; the diagnostic pass shows it is the threshold |
| Query sent without its `query: ` prefix | app image built from a copy with the prefix removed | −0.146 (−0.238 to −0.085), 12 lost | −0.146 | 0.220 (0.144 to 0.321) | No: within the intervals of the real app, whose relation already fails |

**Limits.** One transformation; the rewordings were written by one author for the set, not collected from
learners; 76 pairs give intervals of about ±8 to ±10 points; the first course only (its material stays local), so
this is not a CI check.

## R3, R4 — the limits of one search

**Transformation.** The question stays the same; only a limit of the request changes. For R3 the number of hits
asked for grows, with the same lowest score. For R4 the lowest score a passage must reach rises, with the same
number of hits. The questions are not a written set: [Hypothesis](https://hypothesis.readthedocs.io/) draws them,
each a run of one to twelve words from a stored passage of the course the stack mounts as its local course. Such a
question is close to some passages by meaning and shares words with them, on any course, and needs no gold
answer. For R4 the question is first searched without a lowest score; the two lowest scores are then drawn from
the middles between neighbouring scores of that result (and −1, and a value above the best score), so they cut
the result at different places and never sit exactly on a score.

**What must stay true, and what may vary.**

- **R3, more hits asked (subset consistency).** The hits of the smaller request are the first hits of the larger
  one: the same passages in the same order. It follows that a request never gets more hits than it asked for and
  that asking for more never loses a passage, so recall cannot fall. This holds on every search path: by meaning,
  by meaning and words, and with the reranker.
- **R4, higher lowest score (monotonicity).** On every search path each hit reaches the higher score, and there
  are never more hits than under the lower one. When the search ranks **by meaning alone**, the hits are exactly
  the hits under the lower score that reach the higher one, in the same order: the lowest score is a filter.
  With words (`hybrid`) or with the reranker only the first two conditions are judged, because there the third
  one is false by the way the search is built (below). Which form applies is read from each response (its
  strategy and whether a reranker ordered the hits).

A search that failed, or that the app answered on a weaker path because a model server was down, stops the test:
the relations are stated for the configured search.

**Why a higher lowest score is not a filter with words or a reranker.** In a hybrid search both rankings are
made only from the passages that reach the lowest score, and their places are then fused. When a passage falls
under a higher score, every passage behind it moves up one place in that ranking, the fused order changes, and
a passage that was outside the first k can enter them. With the reranker the lowest score chooses which passages
are reordered, with the same effect. Example on the demo course under `hybrid`, k = 5, question "Barges that are
not used are returned to":

| Lowest score | Hits (first 8 characters of the passage id, similarity) |
|---|---|
| −1 | `7f172213` 0.7965, `e6c1ec7e` 0.7747, `ad25c150` 0.7953, `26e95cc5` 0.7982, `f8633987` 0.7953 |
| 0.785 | `7f172213` 0.7965, `26e95cc5` 0.7982, **`bc3cc163` 0.8166**, `ad25c150` 0.7953, `f8633987` 0.7953 |

`e6c1ec7e` is under 0.785 and leaves; `bc3cc163`, closer to the question by meaning than any of the five, was
outside the first five before and enters. This is not rare: with the
first eight words of each of 120 passages of the first course as questions, k = 5 and every pair of neighbouring
lowest scores, the higher score returned a passage the lower one did not in 218 of 480 pairs, for 113 of the 120
questions (demo course: 8 of 48 pairs). So under `hybrid` the lowest score also changes the ranking, which is why
the comparison of 2026-11-14 found `hybrid` without a lowest score worse than with it.

**What it catches.** For R3: an order that depends on the number of hits asked. A hybrid search whose rankings
keep as many passages as hits are asked, or a reranker that gets only the hits asked, gives another list for
every number; each list looks normal alone. Also a request answered with more hits than it asked for. For R4: a
search path that does not apply the lowest score. Each path applies it in its own place, and a path that skips it
returns well-formed results with passages too many. Neither needs labelled answers, so both run in CI on the
invented course and can serve as oracles when defects are planted.

```powershell
poetry run pytest -m api tests/api/test_search_relations.py --hypothesis-show-statistics
```

### Results, 2026-11-15

First course (782 passages), 25 drawn examples per relation and search setting, one app commit. Each setting was
started by configuration, as for the comparison of 2026-11-14.

| Search | R3 | R4 | Form of R4 judged | Time of both tests |
|---|---|---|---|---|
| `vector` (the default) | holds | holds | by meaning alone: exactly the earlier hits | 5 s |
| `hybrid` | holds | holds | scores and number of hits | 5 s |
| `vector` with reranker | holds | holds | scores and number of hits | 190 s |

"Holds" means no drawn example broke the relation; it is not a proof for every question.

**Planted defects (mutants).** Each one a single changed line of the app, in a source file mounted over the file
inside the app's container (the image and the working tree stay as they are), with the search setting the defect
lives in. The stack was restored after each.

| Mutant | Search setting | Seen by |
|---|---|---|
| The lowest score is not applied to the nearest passages | `vector` | R4: a hit under the lowest score |
| The statement reads one passage more than asked | `vector` | R3: more hits than asked |
| Each ranking of the hybrid search keeps as many passages as hits are asked | `hybrid` | R3: another order |
| The hybrid statement ranks every passage of the course, without the lowest score | `hybrid` | R4: a hit under the lowest score |
| The reranker gets only as many passages as hits are asked | `vector` with reranker | R3: another order |

All five were seen, each by the relation named and not by the other one. Hypothesis then reduced the failing
input to the smallest one it could find: for the first, second and fifth mutant the question "1.1." (the start
of a section heading) with one hit asked.

**Limits.** 25 examples per run are a sample: a defect that shows for few questions can pass a run (Hypothesis
keeps the failing examples it found and tries them first next time). The questions are fragments of the course
text, not questions of learners. The strong form of R4 is judged only for the search by meaning. Two passages
with exactly the same similarity to a question have no fixed order in the search by meaning; no course holds two
passages with the same text today, so R3 has not met that case.
