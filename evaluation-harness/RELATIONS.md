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

**Transformation.** Every answerable question of the first course's Q&A set (82 questions,
[`datasets/istqb_ctfl/qa_v1.toml`](datasets/istqb_ctfl/qa_v1.toml)) is reworded the way a learner asks who does not
know the course's term yet: same meaning, same gold section, none of the question's key terms. "How does boundary
value analysis work?" becomes "How do I pick test inputs at the edges of the allowed ranges?". The rewordings are
in [`datasets/istqb_ctfl/paraphrase_v0.toml`](datasets/istqb_ctfl/paraphrase_v0.toml); a rewording that still
contains a key term is refused before the run. 64 questions contained a key term, which the rewording removed; 18
were already in plain words, and the rewording only changed them. The results are given for both groups.

**What must stay true, and what may vary.**

- **R1, answer kept (invariance).** When a question finds its gold section among the first k passages, its
  rewording finds it too. The rank and the scores may change. Judged on the primary pass (the app's own number of
  hits and lowest score, as a learner gets them) by the change of recall from the questions to the rewordings,
  question by question, with its 95 % interval: **holds** when the interval stays within the smallest meaningful
  change (0.05) or lies above zero, **violated** when the whole interval lies below zero, **not confirmed**
  otherwise.
- **R2, top k stable (ranking stability).** The two searches return mostly the same passages: a pair is stable
  when they share at least 3 of their 5 passages, in any order. Judged on the diagnostic pass (no lowest score;
  the lowest score cuts the two lists to different lengths and would mix its own effect in) by the share of
  stable pairs: **holds** when the lower end of its Wilson interval reaches 0.7, **not confirmed** when only the
  share does, **violated** otherwise.

The rules (0.05, 3 of 5, 0.7) were written into the pair file before the first run.

**What it catches.** A search that follows the course's words instead of the meaning: the gold-set questions
mostly use the course terms, so recall on them cannot see it. A lowest score set too high: a reworded question
matches its passage less closely and falls under it. The diagnostic pass tells the two apart.

**Before it is used.** No question or rewording may copy a held-out question of the local question bank (they are
kept for a final measurement, and these pairs will be used to choose search settings). The check compares
four-word sequences: on 2026-11-07, of 164 texts against 40 held-out questions, the closest shared 0.33 of its
sequences (threshold 0.5), so none was close. It finds word copies only, not the same question in other words.

```powershell
poetry run python -m evalharness.datasets.check_paraphrases datasets/istqb_ctfl/paraphrase_v0.toml datasets/istqb_ctfl/retrieval_gold_v0.toml
poetry run python -m evalharness.evaluation.paraphrase_run datasets/istqb_ctfl/paraphrase_v0.toml
```

### Results, 2026-11-07

First course, 82 pairs, k = 5, lowest score 0.83, embedding model `multilingual-e5-small` (run `81a46d7b`; a
second run, `07568a28`, gave the same result for every pair, as expected from a search that does not sample).

| | Recall of questions | Recall of rewordings | Kept | Lost | Gained | Stable (≥ 3 of 5) |
|---|---|---|---|---|---|---|
| Primary pass, all 82 | 0.976 | 0.866 | 69 | 11 | 2 | 21 |
| — key term removed (64) | 0.984 | 0.859 | 54 | 9 | 1 | 15 |
| — no key term in question (18) | 0.944 | 0.889 | 15 | 2 | 1 | 6 |
| Diagnostic pass, all 82 | 0.976 | 0.878 | 70 | 10 | 2 | 22 |

- **R1 — violated.** Recall change −0.110, 95 % interval −0.195 to −0.037; 11 lost, 2 gained, McNemar p = 0.022.
  Without a lowest score: −0.098 (−0.183 to −0.024), 10 lost. So 10 of the 11 losses come from the embedding
  search itself and 1 from the lowest score: for QA-50 the rewording's best passage scored 0.8299 and the gold passage
  (second) 0.828, just under 0.83, so the learner would get no passage at all.
- **R2 — violated.** 22 of 82 pairs stable, 0.268 (0.184 to 0.373) against 0.7. The answer step reads these
  passages, so for most reworded questions it reads mostly different evidence.
- **Reading.** Search works on the course's words more than on the meaning; a learner who asks without the term
  loses the answer about one time in nine and gets other passages most of the time. Lost pairs: QA-01, 04, 13, 18,
  28, 48, 50, 64, 66, 77, 78. A lost pair means the gold section is not among the 5, not that no useful passage
  is: the gold label is one section per question, so a passage of a neighbouring section that also answers counts
  as a miss.

**Planted defects (mutants).** Each one run once on the stack, then the stack was restored.

| Mutant | How it was planted | R1 primary | R1 diagnostic | R2 | Seen? |
|---|---|---|---|---|---|
| Lowest score 0.86 instead of 0.83 | `APP_SEARCH__MIN_SCORE=0.86` for the app | −0.366 (−0.476 to −0.256), 31 lost | −0.098, unchanged | 0.268, unchanged | Yes: the loss triples, while recall of the questions only falls from 0.976 to 0.927; the diagnostic pass shows it is the threshold |
| Query sent without its `query: ` prefix | app image built from a copy with the prefix removed | −0.146 (−0.238 to −0.085), 12 lost | −0.146 | 0.220 (0.144 to 0.321) | No: within the intervals of the real app, whose relation already fails |

**Limits.** One transformation; the rewordings were written by one author for the set, not collected from
learners; 82 pairs give intervals of about ±8 points; the first course only (its material stays local), so this is
not a CI check.
