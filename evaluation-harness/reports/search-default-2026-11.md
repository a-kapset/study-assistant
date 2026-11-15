# Search default: decision of 2026-11-15

**Decision: the default stays search by meaning (`vector`) without the reranker.** Neither the hybrid search
(`hybrid`) nor the cross-encoder reranker replaces it: neither was shown to find more answers, and the reranker
is also four times over the time budget.

This note says what was measured, how sure the numbers are, what the decision costs and what stays open. The
numbers come from run records of the harness (their ids are at the end); the records hold results measured on a
course whose material stays local, so they are not in the repository.

## The rule, written before the first variant run

- **Two outcomes.** Recall@5 on the 146 exam questions of sets A–C, and recall@5 of the rewordings on 76 pairs
  (a question and its rewording without the course term; see [RELATIONS.md](../RELATIONS.md)).
- **Four planned comparisons.** `hybrid` against `vector`, then the reranker on top of the winner against the
  winner, each on both outcomes. Because four comparisons give four chances to look better by luck, each is read
  with a 98.75 % interval.
- **A variant replaces the default only when** one outcome is better (the whole interval of its change above
  zero) and the measured change is at least +0.05; the other outcome is not worse; the 95th percentile of the
  search stage is at most 1.0 s; and one run on the held-out set confirms it. Otherwise the default stays.
- **Looked at, never decisive:** MRR, slices, the pass without a lowest score, the lists of gained and lost
  questions.

The 1.0 s budget: a learner waits for the search in every mode before any model is called; a model call takes
about 1.2 s at the median and 2.1 s at the 95th percentile here, so a search over one second would become the
longest stage of an answer.

## Sample

One course (782 stored passages). 146 questions from its sample exams (sets A–C) with one answer section each;
76 pairs of a question and its rewording. One search setting differs per run; app commit, corpus, embedding model,
k = 5 and the lowest score 0.83 are the same in all of them, checked by the harness before two runs are compared.
Missing from the sample: questions written by learners, a second course, other languages.

## Results

| Search | Recall@5, sets A–C (95 % interval) | Change against the default (98.75 % interval) | Recall@5 of the rewordings | Change against the default (98.75 % interval) | Search stage, median / 95th percentile |
|---|---|---|---|---|---|
| `vector` (default) | 0.918 (0.862–0.952) | — | 0.868 | — | 0.029 s / 0.055 s |
| `hybrid` | 0.918 (0.862–0.952) | 0.000 (−0.041 to +0.041), 3 gained, 3 lost: no meaningful change | 0.868 | 0.000 (−0.079 to +0.079), 3 gained, 3 lost: underpowered | 0.032 s / 0.143 s |
| `vector` with reranker | 0.890 (0.829–0.931) | −0.027 (−0.089 to +0.027), 4 gained, 8 lost: underpowered | 0.855 | −0.013 (−0.092 to +0.053), 2 gained, 3 lost: underpowered | 1.915 s / 4.377 s |

"Underpowered" means the questions at hand cannot show a change of this size either way; it is not "the same".

**By the rule.** `hybrid`: no outcome is better, so it does not replace the default; its time is within the
budget. Reranker: no outcome is better, it lost more questions than it gained on both outcomes, and its search
stage is over the budget at the median already. No variant is to replace the default, so the confirming run on
the held-out set was not made: that set has been used once and stays for a decision that needs it.

### Where the time goes

146 measured searches per row, one at a time, after 10 warm-up requests; all rows in one session on one machine
(a laptop: Windows 11, 8 logical processors, the stack in Docker Desktop, models on the CPU). Seconds, median /
95th percentile. Stage times are read from each request's trace.

| Search | Whole search | Embedding call | Database | Reranker | Slowest search |
|---|---|---|---|---|---|
| `vector` (default) | 0.029 / 0.055 | 0.017 / 0.041 | 0.008 / 0.012 | — | 0.190 |
| `hybrid` | 0.032 / 0.143 | 0.015 / 0.058 | 0.013 / 0.081 | — | 0.420 |
| `vector` with reranker | 1.915 / 4.377 | 0.059 / 0.140 | 0.011 / 0.021 | 1.852 / 4.293 | 5.671 |
| by words, embedding server unreachable (connection refused at once) | 1.413 / 1.701 | 0.010 / 0.014 | 0.036 / 0.339 | — | 2.176 |
| by words, embedding server's container stopped (15 searches) | 13.318 / 13.500 | 11.922 / 11.972 | 0.028 / 0.189 | — | 13.615 |

Reading: the hybrid statement costs the database a few milliseconds at the median and more in the tail. With
the reranker almost the whole search is the reranker (1.85 of 1.92 s at the median): it reads up to 50 passages
per question on the CPU. The embedding call is also slower in that run; the cause was not looked for. Without
the embedding server the search by words itself is fast (the database column); the rest is waiting: the app
tries the embedding call three times with a pause between the tries (about 1.4 s in all), and with the container
stopped each try first waits about 4 s for a name that no longer resolves. Both rows are over the budget, which
was set for the configured search, not for an outage.

## How sure

- Recall: the intervals above. On 146 questions a change of +0.05 is 8 more questions found than lost; on 76
  pairs it is 4. Smaller effects cannot be shown with these sets.
- Latency: one machine, one caller, one session. Another machine gives other numbers (the wait for a stopped server,
  13 s here, is far shorter on the CI runner), so only the order of magnitude and the comparison between rows
  carry over. The 99th percentile is not given: over 146 requests it is one measurement. Nothing here says how the
  search behaves with many callers at once.
- Exploratory findings, not used for the decision: `hybrid` without a lowest score is worse than with it (0.863
  against 0.918 on sets A–C), because under `hybrid` the lowest score also changes the ranking; with the reranker
  more pairs keep mostly the same passages for a question and its rewording (40 of 76 against 30).

## What the decision leaves open

- **Reworded questions.** About one rewording in seven does not find its section (10 of 76 with the default),
  and neither variant repairs it: each gains a few and loses as many. The search follows the course's words more
  than the meaning.
- **12 of 146 exam questions** do not find their section among five passages.
- **When the embedding server is down** the search answers by words only, marked `degraded`: recall@5 0.781
  (0.707–0.840) against 0.918, and the app declines to write an answer from it. How long such a search takes
  depends on how the server is unreachable (table above).

## What would reopen it

A larger or different question set on which a variant is better by the same rule; for the reranker also a search
stage within the budget, which on this machine means reading far fewer passages per question or another model,
measured before and after on the same sets.

## Records

App commit `48066f35`. Retrieval runs: `5a627e06` (`vector`), `6ab96e3e` (`hybrid`), `f8895262` (reranker),
`8672a8d0` (by words). Paraphrase runs: `7f0b156c`, `7da89ac5`, `9642f7d2`. Latency runs: `a6887fd8`, `2b199141`,
`2240a16b`, `bb17066e` (by words, connection refused); the 15 searches with the container stopped were measured
once by hand with the same code and are not a record. How to make such runs and compare them: [README.md](../README.md), "Retrieval
evaluation".
