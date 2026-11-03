# Metrics

How the harness scores what the study assistant returns. Every metric below is computed by the harness's own
code; this page lists only metrics that exist in the code today. When a metric is added, dropped or computed
differently, this page changes in the same commit, and a dropped or changed definition is kept under
[Changes](#changes).

## Retrieval

Code: [`src/evalharness/oracles/retrieval.py`](src/evalharness/oracles/retrieval.py).

Each question of the gold set has one **answer section**: the course section its learning objective belongs to.
The harness sends the question to the search and scores the passages that come back against that section.

### The search result of one question

The search result of question $q$, $L_q$, is the list of the sections of the returned passages, in the order
the search returned them:

```math
L_q = (s_1, s_2, \dots, s_m), \qquad 0 \le m \le k
```

where

- $q$ — one question of the gold set;
- $s_i$ — the section of the passage at position $i$ of the list (position 1 is the first passage); a passage
  of text before the first heading has no section;
- $m$ — the number of passages returned; $m = 0$ when the threshold (`min_score`) removed all of them;
- $k$ — the number of places the search fills at most (the `k` of the request, 5 by default).

When the search gave no result at all (an error, a timeout), $L_q$ does not exist.

### Whether a passage counts, rel

Relevance, $\mathrm{rel}(s_i, g_q)$, says whether the passage at position $i$ counts for the answer section:

```math
\mathrm{rel}(s_i, g_q) =
\begin{cases}
1 & \text{if } s_i = g_q \text{ or } s_i \text{ starts with } g_q\text{.} \\
0 & \text{otherwise, and when } s_i \text{ is missing}
\end{cases}
```

where

- $s_i$ — the section of the passage at position $i$ of $L_q$;
- $g_q$ — the answer section of question $q$, for example `1.2`.

A subsection counts: `1.2.4` counts for `1.2`. A section that only starts with the same digits does not: `1.20`
does not count for `1.2`, and neither does the parent chapter `1`.

### Per question

**Verdict**, $v_q$ — what happened to question $q$, one of four:

| Verdict | When | What it usually points to |
|---|---|---|
| `failed` | $L_q$ does not exist | the app or the stack is broken |
| `empty` | $m = 0$ | the threshold (`min_score`) cut every passage off |
| `miss` | $m > 0$ and $\mathrm{rel}(s_i, g_q) = 0$ for every $i$ | ranking, embeddings or chunking put other sections first |
| `hit` | $\mathrm{rel}(s_i, g_q) = 1$ for at least one $i$ | — |

**Rank of the first relevant passage**, $r_q$ — the position of the first passage of $L_q$ that counts for
the answer section:

```math
r_q = \min \{\, i : \mathrm{rel}(s_i, g_q) = 1 \,\}
```

where

- $i$ — a position in $L_q$, from 1 to $m$;
- $\mathrm{rel}(s_i, g_q)$ — 1 when the passage at position $i$ counts for the answer section $g_q$.

$r_q$ exists only for a `hit`. Example: $L_q$ = (`3.1`, `1.2`, `1.2.4`) for $g_q$ = `1.2` gives $r_q = 2$.

**Reciprocal rank**, $\mathrm{RR}_q$ — how high the first relevant passage stands: 1 at the top, 1/2 in
second place, and so on:

```math
\mathrm{RR}_q =
\begin{cases}
1 / r_q & \text{if } v_q = \text{hit} \\
0 & \text{if } v_q \text{ is miss, empty or failed}
\end{cases}
```

where

- $r_q$ — the rank of the first relevant passage of question $q$;
- $v_q$ — the verdict of question $q$.

**Precision**, $P_q$ — the share of the $k$ places taken by passages of the answer section:

```math
P_q = \frac{1}{k} \sum_{i=1}^{m} \mathrm{rel}(s_i, g_q)
```

**Irrelevant**, $I_q$ — the share of the $k$ places taken by passages of other sections:

```math
I_q = \frac{1}{k} \sum_{i=1}^{m} \bigl(1 - \mathrm{rel}(s_i, g_q)\bigr)
```

where, in both formulas,

- $k$ — the number of places; the shares are counted over all $k$ places, not over the $m$ returned passages;
- $m$ — the number of passages in $L_q$;
- $\mathrm{rel}(s_i, g_q)$ — 1 when the passage at position $i$ counts for the answer section, otherwise 0.

A short list therefore does not look precise, and a place the threshold left empty is neither relevant nor
irrelevant. When $L_q$ does not exist (`failed`), $P_q = I_q = 0$.

### Per group of questions

The results are averaged over a group of questions $Q$, for example the questions of one sample exam.

**Recall at k**, $\mathrm{recall@}k$ — the share of questions whose answer section was found among the first
$k$ passages:

```math
\mathrm{recall@}k = \frac{1}{n} \sum_{q \in Q} \mathbf{1}[\, v_q = \text{hit} \,]
```

where

- $Q$ — the group of questions; $n$ — the number of questions in it;
- $v_q$ — the verdict of question $q$;
- $\mathbf{1}[\dots]$ — 1 when the condition in the brackets is true, otherwise 0.

Each question has exactly one answer section, so this is the same as the hit rate.

**Mean reciprocal rank**, $\mathrm{MRR}$ — the average of the reciprocal ranks:

```math
\mathrm{MRR} = \frac{1}{n} \sum_{q \in Q} \mathrm{RR}_q
```

where

- $n$ — the number of questions in the group $Q$;
- $\mathrm{RR}_q$ — the reciprocal rank of question $q$ (0 for miss, empty and failed).

**Mean precision**, $\bar{P}$, and **mean irrelevant**, $\bar{I}$ — the averages of the per-question shares:

```math
\bar{P} = \frac{1}{n} \sum_{q \in Q} P_q
\qquad
\bar{I} = \frac{1}{n} \sum_{q \in Q} I_q
```

where

- $n$ — the number of questions in the group $Q$;
- $P_q$, $I_q$ — the precision and the irrelevant share of question $q$.

**Failure counts** — how many questions of the group ended as `miss`, as `empty` and as `failed`. The number of
hits is $n$ minus these three counts.

In every formula of the group, $n$ counts all questions: a failed or empty search stays in the group with 0,
because a search that gives the user nothing must lower the result, never be left out of it. A group with no
questions is an error, not a score.

### Worked example

Answer section $g_q$ = `1.2`, $k = 5$:

| $L_q$ | $v_q$ | $r_q$ | $\mathrm{RR}_q$ | $P_q$ | $I_q$ |
|---|---|---|---|---|---|
| (`3.1`, `1.2`, `1.2.4`) | hit | 2 | 1/2 | 2/5 | 1/5 |
| (`1.2`, `3.1`) | hit | 1 | 1 | 1/5 | 1/5 |
| () — empty list | empty | — | 0 | 0 | 0 |
| does not exist | failed | — | 0 | 0 | 0 |

The last three as a group: $n = 3$, recall@5 = 1/3, MRR = 1/3, one `empty`, one `failed`, no `miss`.

### Reading the metrics together

- **MRR is never above recall@k.** $\mathrm{RR}_q \le 1$ for a hit and 0 otherwise. Their ratio,
  MRR / recall@k, is the mean reciprocal rank of the questions that were found.
- **Recall unchanged, MRR changed:** the same questions are found, but the answer section moved up or down
  inside the first $k$. This is a ranking change, not a coverage change.
- **Recall changed:** questions moved between `hit` and the three failure verdicts. The counts say where they
  went: to `failed` (something broke), to `empty` (the threshold is too strict) or to `miss` (other sections
  rank higher).
- **Precision and irrelevant are not opposites.** $\bar{P} + \bar{I}$ is the average share of filled places;
  $1 - \bar{P} - \bar{I}$ is the share the threshold left empty. Precision can fall because the threshold
  removes good passages (irrelevant falls too) or because other sections push in (irrelevant rises).
- **Precision has a low ceiling here.** An answer section may hold fewer than $k$ passages, so $\bar{P}$ well
  below 1 is normal; compare it between runs, not with 1.

### A run: two passes and the main result

Code: [`src/evalharness/evaluation/retrieval.py`](src/evalharness/evaluation/retrieval.py) (the run),
[`src/evalharness/evaluation/__main__.py`](src/evalharness/evaluation/__main__.py) (the printed summary).

A run sends every chosen question of the gold set to the search twice and scores both results with the
formulas above:

- **primary pass** — the question is sent as a user sends it, so the app's own $k$ and threshold
  (`min_score`) apply; this is the result a user gets;
- **diagnostic pass** — the same $k$ with `min_score` = −1. A cosine score is never below −1, so the
  threshold removes nothing and $L_q$ holds the plain first $k$ passages.

Each pass is summarised over all chosen questions and per sample exam, per chapter and per knowledge level
(K1–K3), each group with its $n$.

**Reading the two passes together.** Per question:

| Primary | Diagnostic | What it means |
|---|---|---|
| `hit` | `hit` | found, and the threshold kept it |
| `miss` or `empty` | `hit` | the answer section was among the first $k$, but its score was under the threshold: the threshold caused the miss |
| `miss` or `empty` | `miss` | the answer section was not among the first $k$ at all: the search itself missed |

So recall@k of the diagnostic pass is the highest value that lowering the threshold alone could reach. The
price of a lower threshold is that questions the course does not cover get passages instead of an empty
result.

**Main result**, $R$ — recall@k of the primary pass over the questions of the declared sets:

```math
R = \mathrm{recall@}k \text{ of the primary pass over } Q_{\text{main}}
```

where

- $Q_{\text{main}}$ — the questions of the sample exams named under `primary.sets` in the gold set's
  declaration;
- $k$ — the number of places the app used, as its response reports it;
- $\mathrm{recall@}k$ — the share of questions whose answer section was found among the first $k$ passages
  (formula above).

**Target verdict** — whether the main result reaches the target fixed in the declaration before the first run:

```math
\text{verdict} =
\begin{cases}
\text{met} & \text{if } R \ge \tau \\
\text{below} & \text{if } R < \tau
\end{cases}
```

where

- $R$ — the main result;
- $\tau$ — `primary.target` in the declaration (0.80 for the first course).

The verdict is printed only when the run took exactly the declared sets and the app used the declared $k$
(`primary.k`). With another $k$ the result is a different metric (recall@10 is not recall@5), so it is shown
without a verdict. "below" is the reason to improve retrieval next. With about 150 questions $R$ is precise only
to a few points, so a result just above or just below $\tau$ is not a real difference.

## Answers

Code: [`src/evalharness/oracles/answer.py`](src/evalharness/oracles/answer.py) (scores),
[`src/evalharness/evaluation/answer_run.py`](src/evalharness/evaluation/answer_run.py) (the run and the target
verdict).

A Q&A set lists questions a learner could ask. An **answerable** question has a **gold section**, the course
section that answers it; an **unanswerable** question has none: the course does not answer it, and the app must
decline. The harness asks every question through `/v1/ask` and scores facts of the response only: whether the
app declined and why, and which sections the cited passages come from. Whether the answer text is right is not
scored here: that needs a judge.

### Per question

**Verdict**, $v_q$ — what the app did with question $q$, one of six:

| Verdict | When | Good for |
|---|---|---|
| `cited_gold` | the app answered and a cited passage counts for $g_q$ | answerable |
| `cited_other` | the app answered and no cited passage counts for $g_q$ | — |
| `answered_unsupported` | the app answered a question without a gold section | — (a blocker) |
| `declined_no_passages` | the app declined because search found nothing close enough; the model was not called | unanswerable |
| `declined_by_model` | the app declined because the model said the passages do not answer the question | unanswerable |
| `failed` | `/v1/ask` gave no answer (an error response, a timeout, a recording that no longer matches) | — |

where

- $q$ — one question of the set;
- $g_q$ — the gold section of question $q$, for example `4.2`; a cited passage counts for it by the same rule
  $\mathrm{rel}$ as a search hit (the section itself or one of its subsections, see Retrieval above).

**Gold section found by search**, $f_q$ — the stage of question $q$: whether a search for the same question, with
the app's own settings, returns a passage that counts for $g_q$:

```math
f_q = \max_{i} \mathrm{rel}(s_i, g_q)
```

where

- $s_i$ — the section of the passage at position $i$ of that search's result;
- $\mathrm{rel}(s_i, g_q)$ — 1 when $s_i$ counts for the gold section $g_q$ of question $q$, otherwise 0;
- $f_q = 0$ when the search found nothing; $f_q$ does not exist for an unanswerable question or a failed search.

The search runs with the same number of hits and the same lowest score as `/v1/ask`, and it is exact, so it
returns the passages the answer step saw. A wrong verdict with $f_q = 0$ points to retrieval; one with $f_q = 1$
points to the answer step (the prompt or the model).

### Per run

**Answered share**, $\alpha$ — the share of the answerable questions that were answered instead of declined:

```math
\alpha = \frac{|A_{\text{ans}}|}{|A^{*}|}
```

where

- $A$ — the answerable questions of the set;
- $A^{*}$ — the questions of $A$ whose verdict $v_q$ is not `failed`;
- $A_{\text{ans}}$ — the questions of $A^{*}$ whose verdict $v_q$ is `cited_gold` or `cited_other`;
- $|\dots|$ — the number of questions in a set.

**Gold cited share**, $\gamma$ — among the answered answerable questions, the share whose citations include the
gold section:

```math
\gamma = \frac{1}{|A_{\text{ans}}|} \sum_{q \in A_{\text{ans}}} \mathbf{1}[\, v_q = \text{cited\_gold} \,]
```

where

- $A_{\text{ans}}$ — the answerable questions that were answered (see $\alpha$); $|A_{\text{ans}}|$ — their number;
- $v_q$ — the verdict of question $q$;
- $\mathbf{1}[\dots]$ — 1 when the condition in the brackets is true, otherwise 0.

$\gamma$ is counted over answered questions only, so it says how well the answers the app gave are supported,
apart from how often it declines ($\alpha$). It is an approximation of support: a passage of another section may
support an answer too, and a cited gold passage does not prove that every claim of the answer is in it.

**Unsupported answers**, $u$ — the number of unanswerable questions the app answered:

```math
u = \sum_{q \in U} \mathbf{1}[\, v_q = \text{answered\_unsupported} \,]
```

where

- $U$ — the unanswerable questions of the set;
- $v_q$ — the verdict of question $q$;
- $\mathbf{1}[\dots]$ — 1 when the condition in the brackets is true, otherwise 0.

A count, not a share: one answer that the course does not support is a blocker whatever $\alpha$ and $\gamma$
are.

A `failed` question stays out of every share and count above: it measured nothing about the answer. When
$A^{*}$ or $A_{\text{ans}}$ is empty, $\alpha$ or $\gamma$ does not exist and is shown as `n/a`.

Also reported, without a target: the **invalid citations**, the sum over all questions of the citations the
model gave to passages it was not shown (the app drops them, the learner does not see them), and the **cost**,
the sum of the listed price of every model call of the run, from the proxy's ledger (a replayed call shows the
cost of the reply it replays; a miss costs nothing).

### Validity and the target verdict

A run is **valid** when no question is `failed`. An invalid run gets no target verdict: it is reported with the
reasons of its failed questions (for example `llm_failed` after a recording no longer matches the request).

A valid run meets its targets when all three hold:

```math
\alpha \ge \alpha_{\min}, \qquad \gamma \ge \gamma_{\min}, \qquad u \le u_{\max}
```

where

- $\alpha$, $\gamma$, $u$ — the answered share, the gold cited share and the unsupported answers of the run;
- $\alpha_{\min}$, $\gamma_{\min}$, $u_{\max}$ — `answered_min`, `gold_cited_min` and `unsupported_answers_max`
  in the `[targets]` table of the set's file (0.85, 0.75 and 0 for the first course), written before the first
  run;
- a share that does not exist (`n/a`) does not meet its minimum.

With 20 answerable and 10 unanswerable questions a share is precise only to about ±20 points, so the targets
tell a working answer step from a broken one, not small differences.

## Changes

None yet.
