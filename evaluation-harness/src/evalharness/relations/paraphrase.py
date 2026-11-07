"""The paraphrase relation on search: a question and its rewording ask the same thing, so search should treat them
alike.

Two relations are checked on every pair:

- answer kept (invariance): when the question finds its gold section among the first k passages, the rewording
  finds it too. Judged on the change of recall from the questions to the rewordings, question by question;
- top-k stable (ranking stability): the two searches return mostly the same passages. Judged on the share of
  pairs whose first k passages overlap by at least a declared part.

Neither relation needs to know the one right list of passages; that is what makes them useful when the gold label
of a question is coarse (one section) or missing.
"""

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

from evalharness.oracles.retrieval import CaseScore
from evalharness.stats.intervals import TargetVerdict, share_verdict
from evalharness.stats.paired import Change, FlipCount, change_verdict, flip_count, paired_change

# "kept": both searches found the gold section. "lost": the question did, the rewording did not. "gained": the other
# way round. "neither": both missed. "failed": one of the two searches gave no result, so the pair measured nothing.
type Found = Literal["kept", "lost", "gained", "neither", "failed"]

# "holds": the relation is shown to hold. "not_confirmed": the result points the right way but the interval is too
# wide to tell. "violated": the relation is shown not to hold.
type RelationVerdict = Literal["holds", "not_confirmed", "violated"]


class _Strict(BaseModel):
    """Base for the results: an unknown key is an error, and a result never changes after it is made."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class PairOutcome(_Strict):
    """How one pair came out: whether each side found the gold section, and how many passages they share."""

    found: Found
    # Passages found by both searches, as a share of the k places; None when a search failed.
    overlap: float | None


class RelationSummary(_Strict):
    """The counts of a group of pairs that the two verdicts are computed from."""

    pairs: int
    failed: int
    kept: int
    lost: int
    gained: int
    neither: int
    # Pairs whose overlap reaches the declared part.
    stable: int


def overlap(first: Sequence[str], second: Sequence[str], k: int) -> float:
    """Return the share of the k places taken by passages that both searches returned, in any order.

    Passages are named by their chunk ids. Order does not count: the answer step reads all k passages, so two
    lists with the same passages in another order give it the same evidence.
    """
    return len(set(first) & set(second)) / k


def pair_outcome(
    question: CaseScore, rewording: CaseScore, question_chunks: Sequence[str], rewording_chunks: Sequence[str], k: int
) -> PairOutcome:
    """Classify one pair from the scores of its two searches and the chunk ids they returned."""
    if question.verdict == "failed" or rewording.verdict == "failed":
        return PairOutcome(found="failed", overlap=None)
    before, after = question.verdict == "hit", rewording.verdict == "hit"
    found: Found = "kept" if before and after else "lost" if before else "gained" if after else "neither"
    return PairOutcome(found=found, overlap=overlap(question_chunks, rewording_chunks, k))


def summarise(outcomes: Sequence[PairOutcome], min_overlap: float) -> RelationSummary:
    """Count the outcomes of a group of pairs; an empty group is an error, never a result."""
    if not outcomes:
        raise ValueError("cannot summarise an empty group of pairs")
    return RelationSummary(
        pairs=len(outcomes),
        failed=sum(o.found == "failed" for o in outcomes),
        kept=sum(o.found == "kept" for o in outcomes),
        lost=sum(o.found == "lost" for o in outcomes),
        gained=sum(o.found == "gained" for o in outcomes),
        neither=sum(o.found == "neither" for o in outcomes),
        stable=sum(o.overlap is not None and o.overlap >= min_overlap for o in outcomes),
    )


def answer_kept(summary: RelationSummary) -> tuple[Change, FlipCount]:
    """Return the change of recall from the questions to their rewordings with its interval, and the McNemar count.

    Only pairs where both searches ran take part; each pair contributes a 1 or 0 for each side.
    """
    s = summary
    before = [True] * (s.kept + s.lost) + [False] * (s.gained + s.neither)
    after = [True] * s.kept + [False] * s.lost + [True] * s.gained + [False] * s.neither
    change = paired_change([float(x) for x in before], [float(x) for x in after])
    return change, flip_count(before, after)


def answer_kept_verdict(summary: RelationSummary, min_meaningful_change: float) -> RelationVerdict | None:
    """Judge the answer-kept relation; None when a pair failed, so the run measured less than it claims.

    Holds: no question lost more than chance and the declared smallest change (also when rewordings found more).
    Violated: the whole interval of the change lies below zero. Otherwise not confirmed.
    """
    if summary.failed:
        return None
    verdict = change_verdict(answer_kept(summary)[0], min_meaningful_change)
    if verdict == "worse":
        return "violated"
    return "not_confirmed" if verdict == "underpowered" else "holds"


def top_k_stable_verdict(summary: RelationSummary, stable_share_min: float) -> RelationVerdict | None:
    """Judge the top-k stable relation by the Wilson interval of the stable share; None when a pair failed."""
    if summary.failed:
        return None
    return _STABLE_VERDICTS[share_verdict(summary.stable, summary.pairs, stable_share_min)]


# A share judged against its target, read as a relation verdict.
_STABLE_VERDICTS: dict[TargetVerdict, RelationVerdict] = {
    "met": "holds",
    "not_confirmed": "not_confirmed",
    "below": "violated",
}
