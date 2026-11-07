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

## Intervals and target verdicts

Code: [`src/evalharness/stats/intervals.py`](src/evalharness/stats/intervals.py) (computed with scipy).

A run measures a sample of questions. Another sample of similar questions would give a somewhat different
number, so every share and mean is reported with a 95 % confidence interval: the range in which the value over
all such questions plausibly lies. The intervals are not stored in run records; they are computed from the
per-question scores a record holds, so any record can be printed again with them.

**Wilson interval of a share**, $\mathrm{CI}_W(x, n)$ — used for every share of yes / no outcomes in this file:
recall@k, and the shares of the answer run below:

```math
\mathrm{CI}_W(x, n) =
\frac{\hat p + \frac{z^2}{2n}}{1 + \frac{z^2}{n}}
\pm \frac{z}{1 + \frac{z^2}{n}} \sqrt{\frac{\hat p\,(1 - \hat p)}{n} + \frac{z^2}{4n^2}},
\qquad \hat p = \frac{x}{n}
```

where

- $x$ — the questions with the outcome counted (e.g. `hit`); $n$ — all questions of the share;
- $\hat p$ — the measured share;
- $z$ — 1.96, the multiplier of a 95 % interval.

It stays inside 0..1 and is usable at 0 of $n$ and $n$ of $n$, where the simpler $\hat p \pm z\sqrt{\hat p(1-\hat
p)/n}$ breaks. Example: 19 of 20 gives 0.764–0.991.

**Bootstrap interval of a mean**, $\mathrm{CI}_B$ — used for MRR, whose values per question are not yes / no:

```math
\mathrm{CI}_B = \mathrm{BCa}_{95\%}\big(\overline{\mathrm{RR}}^{(1)}, \dots, \overline{\mathrm{RR}}^{(B)}\big),
\qquad \overline{\mathrm{RR}}^{(b)} = \frac{1}{n} \sum_{j=1}^{n} \mathrm{RR}_{q_j^{(b)}}
```

where

- $\mathrm{RR}_q$ — the reciprocal rank of question $q$ (see Per question above); $n$ — the questions of the group;
- $q_j^{(b)}$ — the $j$-th question of resample $b$, drawn with replacement from the $n$ questions;
- $B$ — 9 999 resamples, with a fixed seed, so the same record always prints the same interval;
- $\overline{\mathrm{RR}}^{(b)}$ — the mean reciprocal rank recomputed on resample $b$;
- $\mathrm{BCa}_{95\%}$ — the range that holds the middle 95 % of the $B$ values, with percentiles corrected for
  bias and skew (bias-corrected and accelerated, scipy's default method).

There is no interval when $n < 2$ or every $\mathrm{RR}_q$ is the same: resampling cannot change the mean then.

**Target verdict of a share**, $V(x, n, \tau)$ — whether a share reaches a minimum fixed before the run:

```math
V(x, n, \tau) =
\begin{cases}
\text{met} & \text{if } \ell \ge \tau \\
\text{not confirmed} & \text{if } \ell < \tau \le x / n \\
\text{below} & \text{if } x / n < \tau
\end{cases}
```

where

- $x$, $n$ — the questions with the outcome counted and all questions of the share;
- $\ell$ — the lower end of $\mathrm{CI}_W(x, n)$;
- $\tau$ — the minimum the share must reach.

"met" means even the lower end reaches the target; "not confirmed" means the measured share does, but the sample
is too small to tell it from a value under the target.

## A retrieval run: two passes and the main result

Code: [`src/evalharness/evaluation/retrieval.py`](src/evalharness/evaluation/retrieval.py) (the run),
[`src/evalharness/evaluation/__main__.py`](src/evalharness/evaluation/__main__.py) (the printed summary).

A run sends every chosen question of the gold set to the search twice and scores both results with the
formulas of [Retrieval](#retrieval):

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
  (see Per group of questions).

**Target verdict** — whether the main result reaches the target fixed in the declaration before the first run,
judged by its interval (see [Intervals and target verdicts](#intervals-and-target-verdicts) above):

```math
\text{verdict} = V(h, n, \tau)
```

where

- $V$ — the target verdict of a share (met, not confirmed or below), defined under Intervals above;
- $h$ — the questions of $Q_{\text{main}}$ whose verdict $v_q$ in the primary pass is `hit`, so $R = h / n$;
- $n$ — the number of questions in $Q_{\text{main}}$;
- $\tau$ — `primary.target` in the declaration (0.80 for the first course).

The verdict is printed only when the run took exactly the declared sets and the app used the declared $k$
(`primary.k`). With another $k$ the result is a different metric (recall@10 is not recall@5), so it is shown
without a verdict. With `--require-target` (CI) the command fails unless the verdict is "met". "below" is the
reason to improve retrieval next; "not confirmed" says the gold set is too small to tell.

Every row of the printed summary shows recall@k with its Wilson interval, and the main result's MRR is printed
with its bootstrap interval.

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

A valid run gets one verdict per target; the two shares are judged by their intervals (see
[Intervals and target verdicts](#intervals-and-target-verdicts) above), the unsupported answers by their count:

```math
V\bigl(|A_{\text{ans}}|,\ |A^{*}|,\ \alpha_{\min}\bigr), \qquad
V\bigl(c,\ |A_{\text{ans}}|,\ \gamma_{\min}\bigr), \qquad
\begin{cases}
\text{met} & \text{if } u \le u_{\max} \\
\text{below} & \text{if } u > u_{\max}
\end{cases}
```

where

- $V$ — the target verdict of a share (met, not confirmed or below), defined under Intervals above;
- $A^{*}$, $A_{\text{ans}}$ — the scored and the answered answerable questions (see $\alpha$); $|\dots|$ — their
  number;
- $c$ — the questions of $A_{\text{ans}}$ whose verdict $v_q$ is `cited_gold`, so $\gamma = c / |A_{\text{ans}}|$;
- $u$ — the unsupported answers of the run;
- $\alpha_{\min}$, $\gamma_{\min}$, $u_{\max}$ — `answered_min`, `gold_cited_min` and `unsupported_answers_max`
  in the `[targets]` table of the set's file (0.85, 0.75 and 0 for the first course), written before the first
  run;
- a share that does not exist (`n/a`) is "below" its minimum.

"all targets" in the printed summary is the worst of the three: "below" if any is below, else "not confirmed"
if any is not confirmed, else "met". The unsupported answers are printed with the Wilson interval of their share
of $U$: no unsupported answer among 10 unanswerable questions still allows a true share of up to 0.28.

### Repetitions and the flip rate

The answer model samples its reply, so the same question with the same passages may get another verdict when it
is asked again. A run with `--reps n` asks every question $n$ times (search once: it does not sample). Each
**repetition** $r = 1, \dots, n$ gets its own $\alpha$, $\gamma$, $u$, validity and target verdicts, computed as
above over the verdicts $v_{q,r}$ of that repetition; they are not added up, so one repetition's luck stays
visible.

**Flip rate**, $\phi$ — the share of questions whose verdict was not the same in every repetition:

```math
\phi = \frac{1}{|Q^{*}|} \sum_{q \in Q^{*}} \mathbf{1}\bigl[\, |\{ v_{q,1}, \dots, v_{q,n} \}| > 1 \,\bigr]
```

where

- $Q^{*}$ — the questions of the set, answerable and unanswerable, with no `failed` verdict in any repetition (a
  failed repetition measured nothing about the answer); $|Q^{*}|$ — their number;
- $v_{q,r}$ — the verdict of question $q$ in repetition $r$;
- $\{ v_{q,1}, \dots, v_{q,n} \}$ — the different verdicts question $q$ got; more than one means it flipped;
- $n$ — the number of repetitions;
- $\mathbf{1}[\dots]$ — 1 when the condition in the brackets is true, otherwise 0.

Every change of verdict counts, also between `cited_gold` and `cited_other` (the answer was given each time, but
its support changed). $\phi$ is printed with its Wilson interval and the list of flipped questions with their
verdicts in order; it has no target. It says how much of a single run is chance: a difference between two
versions smaller than what the same version shows between its own repetitions is not evidence of a change. A
flipped question is a borderline case, the first place to look.

## Comparison of two runs

Code: [`src/evalharness/stats/paired.py`](src/evalharness/stats/paired.py),
[`src/evalharness/evaluation/compare.py`](src/evalharness/evaluation/compare.py).

Two retrieval runs are compared only when they measured the same questions (same gold-set hash, course and exam
sets) and their recorded corpus snapshot, embedding space, k and lowest score differ in exactly the one factor
declared for the comparison, or in nothing when none is declared. Both runs answered the same questions, so the
comparison is made question by question, on the primary pass: a hard question is hard in both runs, and the
per-question differences carry much less noise than two separately measured numbers.

**Change of a metric**, $\Delta$ — the mean of the per-question differences, after minus before:

```math
\Delta = \frac{1}{n} \sum_{q=1}^{n} d_q, \qquad d_q = s_q^{\text{after}} - s_q^{\text{before}}
```

where

- $n$ — the questions of the comparison (the same in both runs);
- $s_q^{\text{before}}$, $s_q^{\text{after}}$ — the score of question $q$ in the run before and after the change:
  for recall 1 when the question is a `hit` and 0 otherwise, for MRR its reciprocal rank $\mathrm{RR}_q$;
- $d_q$ — the difference for question $q$; for recall it is $+1$ (gained), $-1$ (lost) or 0.

For recall, $\Delta$ equals $R^{\text{after}} - R^{\text{before}}$, where $R^{\text{before}}$ and $R^{\text{after}}$
are the recall@k of the primary pass of each run over these $n$ questions.

**Interval of the change**, $\mathrm{CI}_B(\Delta)$ — the bootstrap interval of a mean (see Intervals above), computed over the
$n$ differences $d_q$ instead of the scores:

```math
\mathrm{CI}_B(\Delta) = [\,\ell_\Delta,\ h_\Delta\,] = \mathrm{BCa}_{95\%}\big(\Delta^{(1)}, \dots, \Delta^{(B)}\big),
\qquad \Delta^{(b)} = \frac{1}{n} \sum_{j=1}^{n} d_{q_j^{(b)}}
```

where

- $d_q$ — the difference for question $q$ (see the change above); $n$ — the questions of the comparison;
- $q_j^{(b)}$ — the $j$-th question of resample $b$, drawn with replacement from the $n$ questions;
- $B$ — 9 999 resamples, with a fixed seed, so the same two records always give the same interval;
- $\Delta^{(b)}$ — the change recomputed on resample $b$;
- $\mathrm{BCa}_{95\%}$ — the middle 95 % of the $B$ values, as for the bootstrap interval of a mean above;
- $\ell_\Delta$, $h_\Delta$ — the lower and upper end of that range.

There is no interval when every $d_q$ is the same (in practice: no question changed).

**McNemar's exact test**, $p$ — printed for recall next to the counts of flipped questions:

```math
p = P\big(\, |X - \tfrac{m}{2}| \ge |g - \tfrac{m}{2}| \,\big), \qquad X \sim \mathrm{Binomial}(m, \tfrac{1}{2}), \qquad m = g + l
```

where

- $g$ — the questions gained (not a hit before, a hit after); $l$ — the questions lost (the other way round);
- $m$ — the questions that flipped; the $n - m$ unchanged questions say nothing about which run is better;
- $X$ — the number of gains among $m$ flips if the change did nothing and each flip went either way with equal
  chance (the null hypothesis);
- $p$ — the chance of a split at least as uneven as the one seen, under that hypothesis. None when $m = 0$.

Example: the second primary source of the first course, $g = 4$, $l = 7$, $p = 0.549$; the course sections, $g = 6$,
$l = 0$, $p = 2 \cdot (1/2)^6 = 0.031$. A large $p$ does not show that nothing changed: it only fails to show a
change. The verdict therefore reads the interval, not $p$; near $p = 0.05$ the two can disagree, and then the
interval decides.

**Verdict of a change**, $W(\Delta)$ — given for recall, which has a target; MRR gets $\Delta$ and its interval only:

```math
W(\Delta) =
\begin{cases}
\text{unchanged} & \text{if every } d_q = 0 \\
\text{better} & \text{if } \ell_\Delta > 0 \\
\text{worse} & \text{if } h_\Delta < 0 \\
\text{no meaningful change} & \text{if } -\delta < \ell_\Delta \le 0 \le h_\Delta < \delta \\
\text{underpowered} & \text{otherwise}
\end{cases}
```

where

- $d_q$ — the per-question differences of recall;
- $\ell_\Delta$, $h_\Delta$ — the lower and upper end of $\mathrm{CI}_B(\Delta)$; $\Delta$ itself is printed next to
  them, but the verdict reads only the two ends;
- $\delta$ — the smallest change worth acting on, written in the gold declaration before comparing
  (`[comparison] min_meaningful_change`; 0.05 for the first course, about 7 of 146 questions); it is not part of
  the gold-set hash, so it can be added without making earlier runs incomparable.

"underpowered" means the interval includes zero but also a change of at least $\delta$: these questions cannot
tell whether the change mattered. Example: the second primary source, $\Delta = -0.021$, interval $-0.068$ to
$+0.021$ with $\delta = 0.05$, is underpowered: a loss of up to 7 points is not ruled out.

**Reading a comparison.** Recall says whether more or fewer questions find their answer section; MRR adds how high
it stands. Read together, by their intervals:

- **Recall unchanged or no meaningful change, MRR interval wholly above zero:** the same questions are found, but
  the answer section stands higher among the first $k$ — a ranking gain, not a coverage gain.
- **Both intervals above zero:** more questions are found, and found ones stand higher. Example: the course
  sections of the first course, recall $+0.041$ ($+0.014$ to $+0.082$), MRR $+0.043$ ($+0.024$ to $+0.070$).
- **MRR interval includes zero:** no change of ranking is shown, whatever the mean moved. Example: the second
  primary source, MRR $+0.006$ ($-0.032$ to $+0.047$).
- **Recall better, MRR interval includes zero or lies below it:** a newly found question adds its
  $\mathrm{RR}_q > 0$ to MRR, so an MRR that does not rise with it means questions found before moved down.

## Metamorphic relations on search

Code: [`src/evalharness/relations/paraphrase.py`](src/evalharness/relations/paraphrase.py),
[`src/evalharness/evaluation/paraphrase_run.py`](src/evalharness/evaluation/paraphrase_run.py). What the relations
are, why, and their results: [RELATIONS.md](RELATIONS.md).

A pair is a question $q$ of a Q&A set and its rewording $q'$ without the question's key terms; both have the gold
section of $q$. Both are searched in two passes, as in a retrieval run: primary (the app's own $k$ and lowest score)
and diagnostic (the same $k$, no lowest score). Each search is scored like a retrieval question (see Per question
above): `hit` when a passage of the gold section is among the first $k$.

**Outcome of a pair**, $o_p$ — from the verdicts of its two searches in one pass:

```math
o_p =
\begin{cases}
\text{failed} & \text{if either search of } p \text{ failed} \\
\text{kept} & \text{if } h_q = 1 \text{ and } h_{q'} = 1 \\
\text{lost} & \text{if } h_q = 1 \text{ and } h_{q'} = 0 \\
\text{gained} & \text{if } h_q = 0 \text{ and } h_{q'} = 1 \\
\text{neither} & \text{if } h_q = 0 \text{ and } h_{q'} = 0
\end{cases}
```

where

- $p$ — one pair, made of the question $q$ and its rewording $q'$;
- $h_q$, $h_{q'}$ — 1 when the search of $q$ (of $q'$) is a `hit`, 0 when it is a `miss` or `empty`.

**Overlap of a pair**, $O_p$ — the share of the $k$ places taken by passages both searches returned:

```math
O_p = \frac{|C_q \cap C_{q'}|}{k}
```

where

- $C_q$, $C_{q'}$ — the sets of chunk ids returned for $q$ and for $q'$; order does not count, because the answer
  step reads all of them;
- $k$ — the number of hits the app returns (5), the same for both searches;
- $|C_q \cap C_{q'}|$ — the number of passages found by both. Not defined when a search failed.

**Relation R1, answer kept** — judged on the primary pass by the change of recall from the questions to their
rewordings, which is the change $\Delta$ of [Comparison of two runs](#comparison-of-two-runs) with "before" = the
questions and "after" = the rewordings:

```math
\Delta = \frac{1}{n} \sum_{p=1}^{n} \big(h_{q'} - h_q\big) = \frac{g - l}{n}
```

where

- $n$ — the pairs with no failed search;
- $h_q$, $h_{q'}$ — as in the outcome above;
- $g$ — the pairs `gained`, $l$ — the pairs `lost`.

$\Delta$ gets its bootstrap interval $[\ell_\Delta, h_\Delta]$, McNemar's exact $p$ on $g$ and $l$, and the verdict
$W(\Delta)$ with the smallest meaningful change $\delta$ from the pair file (`[relations] min_meaningful_change`,
0.05), all as defined there. The relation reads it as:

```math
R_1 =
\begin{cases}
\text{holds} & \text{if } W(\Delta) \in \{\text{unchanged},\ \text{no meaningful change},\ \text{better}\} \\
\text{violated} & \text{if } W(\Delta) = \text{worse} \\
\text{not confirmed} & \text{if } W(\Delta) = \text{underpowered}
\end{cases}
```

where

- $W(\Delta)$ — the verdict of a change, read from the two ends of its interval;
- "better" counts as holding: rewordings that find more answers do not break the relation.

The same $\Delta$ is printed for the diagnostic pass; the difference between the two passes is the share of losses
the lowest score causes.

**Relation R2, top k stable** — judged on the diagnostic pass by the share of stable pairs:

```math
\sigma = \frac{1}{n} \sum_{p=1}^{n} \mathbf{1}\big[\, O_p \ge \omega \,\big], \qquad R_2 = V(n\sigma,\ n,\ \sigma_{\min})
```

where

- $n$ — the pairs with no failed search;
- $O_p$ — the overlap of pair $p$;
- $\omega$ — the part of the $k$ passages a stable pair must share (`[relations] min_overlap`, 0.6: 3 of 5);
- $\mathbf{1}[\dots]$ — 1 when the condition in the brackets is true, otherwise 0;
- $n\sigma$ — the number of stable pairs;
- $\sigma_{\min}$ — the share of stable pairs the search must reach (`[relations] stable_share_min`, 0.7);
- $V$ — the target verdict of a share by its Wilson interval (see Intervals above), read as holds (met), not
  confirmed, violated (below).

The diagnostic pass is used because the lowest score cuts the two lists to different lengths: two searches that
return the same two passages would have $O_p = 0.4$ and look unstable for a reason that R1 already shows.

**Validity.** A run with a failed search in a pass gives that pass no verdict for either relation; it is printed
as invalid with the number of such pairs.

**Slices.** Every count is also given for the pairs whose question contained a key term (the rewording removed
it) and for the pairs whose question did not.

## Changes

- **2026-11-07 — metamorphic relations on search.** New: the outcome $o_p$ and overlap $O_p$ of a pair, and the
  relations R1 (answer kept) and R2 (top k stable). Nothing earlier changed.

- **2026-11-06 — answer runs may repeat every question.** An answer record holds the repetition of every case and
  one summary per repetition; the flip rate $\phi$ is new. A run without `--reps` is one repetition and reports as
  before.

- **2026-11-05 — target verdicts read the interval.** Before, a share met its target when the measured value
  reached it ($R \ge \tau$; $\alpha \ge \alpha_{\min}$, $\gamma \ge \gamma_{\min}$), with only "met" and "below".
  Reason: with 20 answerable questions, 19 of 20 (0.95) was "met" against 0.85 although its interval starts at
  0.764. Results reported before this date used the old rule; the answer run of 2026-11-03 ("all targets met")
  is "not confirmed" under the new one, the retrieval main results stay "met".
