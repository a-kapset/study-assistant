"""Exact scores of the answers to a Q&A set: did the app answer or decline, did it cite the gold section, and at
which stage a case went wrong.

Only facts of the response are scored; whether the answer text is right needs a judge and is not scored here. A
case whose ask failed is counted apart and left out of every share: it measured nothing about the answer.
"""

from collections import Counter
from collections.abc import Sequence
from statistics import fmean
from typing import Literal

from pydantic import BaseModel, ConfigDict

from evalharness.oracles.retrieval import is_relevant

# Why the app declined, as its response states it.
type AbstainReason = Literal["no_passages", "model_declined"]
# "cited_gold": answered and cited the gold section. "cited_other": answered without citing it. "answered_unsupported":
# answered a question the course does not answer. "declined_no_passages" / "declined_by_model": declined, by search
# finding nothing close enough or by the model. "failed": no answer at all.
type AnswerVerdict = Literal[
    "cited_gold", "cited_other", "answered_unsupported", "declined_no_passages", "declined_by_model", "failed"
]


class _Strict(BaseModel):
    """Base for the scores: an unknown key is an error, and a score never changes after it is made."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class AskReply(_Strict):
    """What the scores need from an answer: why it declined, if it did, and the sections of the passages it cites."""

    abstain_reason: AbstainReason | None
    cited_sections: tuple[str | None, ...]


class AnswerCaseScore(_Strict):
    """The scores of one question."""

    verdict: AnswerVerdict
    # Whether a search for the same question finds the gold section; None without a gold section or a search result.
    gold_retrieved: bool | None


class AnswerSummary(_Strict):
    """The results of a run over a Q&A set, in the terms its targets use."""

    answerable: int
    unanswerable: int
    failed: int
    # Share of answerable questions answered instead of declined; None when no answerable question was scored.
    answered: float | None
    # Share of answered answerable questions that cite the gold section; None when none was answered.
    gold_cited: float | None
    unsupported_answers: int
    verdicts: dict[AnswerVerdict, int]


def score_case(
    gold_section: str | None, reply: AskReply | None, search_sections: Sequence[str | None] | None
) -> AnswerCaseScore:
    """Score one question from its answer (None when the ask failed) and the sections a search for it found (None
    when that search failed)."""
    gold_retrieved = (
        None
        if gold_section is None or search_sections is None
        else any(is_relevant(section, gold_section) for section in search_sections)
    )
    return AnswerCaseScore(verdict=_verdict(gold_section, reply), gold_retrieved=gold_retrieved)


def _verdict(gold_section: str | None, reply: AskReply | None) -> AnswerVerdict:
    """Say what the app did with the question, judged against whether the course answers it."""
    if reply is None:
        return "failed"
    if reply.abstain_reason == "no_passages":
        return "declined_no_passages"
    if reply.abstain_reason == "model_declined":
        return "declined_by_model"
    if gold_section is None:
        return "answered_unsupported"
    cited = any(is_relevant(section, gold_section) for section in reply.cited_sections)
    return "cited_gold" if cited else "cited_other"


def summarise(gold_sections: Sequence[str | None], scores: Sequence[AnswerCaseScore]) -> AnswerSummary:
    """Sum up a run; ``gold_sections`` and ``scores`` are in the same order, one entry per question."""
    if len(gold_sections) != len(scores):
        raise ValueError(f"{len(gold_sections)} questions but {len(scores)} scores")
    answerable = [s.verdict for g, s in zip(gold_sections, scores, strict=True) if g is not None]
    scored = [verdict for verdict in answerable if verdict != "failed"]
    answered = [verdict for verdict in scored if verdict in ("cited_gold", "cited_other")]
    return AnswerSummary(
        answerable=len(answerable),
        unanswerable=len(scores) - len(answerable),
        failed=sum(score.verdict == "failed" for score in scores),
        answered=fmean(verdict in answered for verdict in scored) if scored else None,
        gold_cited=fmean(verdict == "cited_gold" for verdict in answered) if answered else None,
        unsupported_answers=sum(score.verdict == "answered_unsupported" for score in scores),
        verdicts=dict(Counter(score.verdict for score in scores)),
    )
