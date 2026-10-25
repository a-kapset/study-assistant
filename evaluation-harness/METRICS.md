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

## Changes

None yet.
