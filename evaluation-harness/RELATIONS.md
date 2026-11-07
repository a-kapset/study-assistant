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

**Planted defects (mutants)**, on the first 82 pairs with the first rule (3 of 5). Each one run once on the stack,
then the stack was restored.

| Mutant | How it was planted | R1 primary | R1 diagnostic | R2 | Seen? |
|---|---|---|---|---|---|
| Lowest score 0.86 instead of 0.83 | `APP_SEARCH__MIN_SCORE=0.86` for the app | −0.366 (−0.476 to −0.256), 31 lost | −0.098, unchanged | 0.268, unchanged | Yes: the loss triples, while recall of the questions only falls from 0.976 to 0.927; the diagnostic pass shows it is the threshold |
| Query sent without its `query: ` prefix | app image built from a copy with the prefix removed | −0.146 (−0.238 to −0.085), 12 lost | −0.146 | 0.220 (0.144 to 0.321) | No: within the intervals of the real app, whose relation already fails |

**Limits.** One transformation; the rewordings were written by one author for the set, not collected from
learners; 76 pairs give intervals of about ±8 to ±10 points; the first course only (its material stays local), so
this is not a CI check.
