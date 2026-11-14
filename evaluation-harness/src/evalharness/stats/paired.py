"""Comparison of two runs on the same questions: how much a number changed, how sure we are, and what that means.

Both runs answered the same questions, so the comparison is made question by question: for every question the
score after the change minus the score before it. A hard question is hard in both runs, so these per-question
differences carry much less noise than two separately measured numbers. Their mean is the change of the metric,
and a bootstrap over them gives its interval.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Literal

from scipy.stats import binomtest

from evalharness.stats.intervals import CONFIDENCE, Interval, mean_interval

# "better" / "worse": the whole interval of the change lies above / below zero. "no_meaningful_change": the interval
# includes zero and stays closer to it than the smallest change worth acting on. "underpowered": the interval
# includes zero but also reaches a change worth acting on, so these questions cannot tell. "unchanged": no question
# scored differently.
type ChangeVerdict = Literal["better", "worse", "no_meaningful_change", "underpowered", "unchanged"]


@dataclass(frozen=True)
class Change:
    """The change of a mean over the same questions, and its interval; None when every question changed alike."""

    change: float
    interval: Interval | None


@dataclass(frozen=True)
class FlipCount:
    """The questions that changed their pass / fail result between two runs, and the exact McNemar test on them."""

    # Failed before and passed after, and the other way round.
    gained: int
    lost: int
    # The chance of a split at least this uneven if the change did nothing and every flip went either way with
    # equal chance; None when no question flipped.
    p_value: float | None


def planned_confidence(planned: int) -> float:
    """Return the level each interval needs when ``planned`` comparisons are judged together (Bonferroni).

    Every comparison has its own chance of a false "better" or "worse". With several comparisons planned for one
    decision, the allowed 5 % is shared equally among them, so all of them together still go wrong at most 5 % of
    the time: one comparison keeps 95 %, four need 98.75 % each.
    """
    if planned < 1:
        raise ValueError(f"{planned} planned comparisons: at least the one being made is planned")
    return 1 - (1 - CONFIDENCE) / planned


def paired_change(before: Sequence[float], after: Sequence[float], confidence: float = CONFIDENCE) -> Change:
    """Return the mean of the per-question differences ``after - before`` with its bootstrap interval.

    The two sequences hold the scores of the same questions in the same order. ``confidence`` is the level of the
    interval: 95 % for a single comparison, the level of ``planned_confidence`` for one of several planned ones.
    """
    if len(before) != len(after):
        raise ValueError(f"{len(before)} scores before and {len(after)} after: not the same questions")
    differences = [new - old for old, new in zip(before, after, strict=True)]
    return Change(change=fmean(differences), interval=mean_interval(differences, confidence))


def flip_count(before: Sequence[bool], after: Sequence[bool]) -> FlipCount:
    """Count the questions that passed in one run only, and test whether the split between the two directions is
    more uneven than chance (McNemar's test in its exact form: a binomial test on the flipped questions alone)."""
    if len(before) != len(after):
        raise ValueError(f"{len(before)} results before and {len(after)} after: not the same questions")
    gained = sum(not old and new for old, new in zip(before, after, strict=True))
    lost = sum(old and not new for old, new in zip(before, after, strict=True))
    flipped = gained + lost
    p_value = float(binomtest(gained, flipped).pvalue) if flipped else None
    return FlipCount(gained=gained, lost=lost, p_value=p_value)


def change_verdict(change: Change, min_meaningful: float) -> ChangeVerdict:
    """Judge a change by its interval against the smallest change worth acting on (from the gold declaration)."""
    if change.interval is None:
        # Every question changed by the same amount; with real question sets that amount is zero.
        if change.change == 0:
            return "unchanged"
        return "better" if change.change > 0 else "worse"
    low, high = change.interval.low, change.interval.high
    if low > 0:
        return "better"
    if high < 0:
        return "worse"
    if -min_meaningful < low and high < min_meaningful:
        return "no_meaningful_change"
    return "underpowered"
