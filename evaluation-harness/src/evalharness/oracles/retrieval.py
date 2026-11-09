"""Reference scores of a search against the course section that answers the question, and their summary.

A search that finds nothing gives the user nothing, so it scores zero and stays in every count. A search that
failed (the embedding server, the database or the app broke) measured nothing about retrieval: it is counted apart
and left out of the shares, and a run with such a failure gets no target verdict, so a broken stack can neither
lower the numbers nor pass for a better search.
"""

from collections.abc import Sequence
from statistics import fmean
from typing import Literal

from pydantic import BaseModel, ConfigDict

# "hit": a passage of the answer section is among the first k. "miss": the search found passages, none of them from
# that section. "empty": the search ran and found nothing above its lowest score. "failed": no search result at all.
type Verdict = Literal["hit", "miss", "empty", "failed"]


class _Strict(BaseModel):
    """Base for the scores: an unknown key is an error, and a score never changes after it is made."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class CaseScore(_Strict):
    """The scores of one question. Without a passage of the answer section in the first k, the rank, the
    reciprocal rank and the precision are empty or 0; ``irrelevant`` still counts the other sections found."""

    verdict: Verdict
    # Position of the first passage from the answer section, starting at 1; None when there is none in the first k.
    first_relevant_rank: int | None
    reciprocal_rank: float
    # Share of the k places taken by passages of the answer section, and by passages of other sections.
    precision: float
    irrelevant: float


class Summary(_Strict):
    """The mean scores of a group of questions and how many of them ended in each kind of failure."""

    n: int
    # Share of the questions whose search ran with the answer section among the first k (one answer section per
    # question). The shares are over the searches that ran; all four are None when none did.
    recall: float | None
    mrr: float | None
    precision: float | None
    irrelevant: float | None
    misses: int
    empty: int
    failed: int


def is_relevant(section_id: str | None, gold_section: str) -> bool:
    """Say whether a passage counts for the answer section: it comes from that section or one of its subsections."""
    return section_id is not None and (section_id == gold_section or section_id.startswith(f"{gold_section}."))


def score_case(gold_section: str, sections: Sequence[str | None] | None, k: int) -> CaseScore:
    """Score one search from the sections of its hits in the order they were returned; None means the search failed.

    The search returns at most k hits, fewer when the lowest score cuts some off; ``k`` is the number of places
    the shares are counted over. Positions come from the order of the list, which is the order every consumer
    reads the hits in; the rank field of the response states the same order and is checked by the app's own tests.
    """
    if sections is None:
        return CaseScore(verdict="failed", first_relevant_rank=None, reciprocal_rank=0.0, precision=0.0, irrelevant=0.0)
    relevant = [is_relevant(section, gold_section) for section in sections]
    first = relevant.index(True) + 1 if any(relevant) else None
    verdict: Verdict = "hit" if first is not None else ("miss" if sections else "empty")
    return CaseScore(
        verdict=verdict,
        first_relevant_rank=first,
        reciprocal_rank=1 / first if first is not None else 0.0,
        precision=sum(relevant) / k,
        irrelevant=(len(sections) - sum(relevant)) / k,
    )


def summarise(scores: Sequence[CaseScore]) -> Summary:
    """Average the scores of a group of questions over the searches that ran; an empty group is an error, never a
    score of 0 or 1."""
    if not scores:
        raise ValueError("cannot summarise an empty group of questions")
    ran = [score for score in scores if score.verdict != "failed"]
    return Summary(
        n=len(scores),
        recall=fmean(score.verdict == "hit" for score in ran) if ran else None,
        mrr=fmean(score.reciprocal_rank for score in ran) if ran else None,
        precision=fmean(score.precision for score in ran) if ran else None,
        irrelevant=fmean(score.irrelevant for score in ran) if ran else None,
        misses=sum(score.verdict == "miss" for score in scores),
        empty=sum(score.verdict == "empty" for score in scores),
        failed=sum(score.verdict == "failed" for score in scores),
    )
