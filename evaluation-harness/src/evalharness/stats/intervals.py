"""Confidence intervals of the numbers a run reports, and the verdict of a share against its target.

A run measures a sample of questions, so a share such as recall is an estimate: another set of similar questions
would give a somewhat different number. An interval says how far the true value may plausibly lie from the measured
one, so a target is judged against the whole range, not against one number.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Literal

from scipy.stats import binomtest, bootstrap

# The usual level in evaluation reports; every interval of a run uses it.
CONFIDENCE = 0.95
# A fixed seed: the same record printed twice shows the same bootstrap interval.
_BOOTSTRAP_SEED = 0
_BOOTSTRAP_RESAMPLES = 9999

# "met": even the lower end of the interval reaches the target. "not_confirmed": the measured share does, the lower
# end does not, so more questions are needed to tell. "below": the measured share is under the target.
type TargetVerdict = Literal["met", "not_confirmed", "below"]


@dataclass(frozen=True)
class Interval:
    """The lower and upper end of a confidence interval: 95 % unless the caller asked for another level."""

    low: float
    high: float


def share_interval(successes: int, n: int) -> Interval:
    """Return the Wilson interval of ``successes`` out of ``n``; it stays inside 0..1 and works for 0 and n too."""
    if n == 0:
        raise ValueError("a share of no questions has no interval")
    interval = binomtest(successes, n).proportion_ci(confidence_level=CONFIDENCE, method="wilson")
    return Interval(low=float(interval.low), high=float(interval.high))


def mean_interval(values: Sequence[float], confidence: float = CONFIDENCE) -> Interval | None:
    """Return a bootstrap interval of the mean of ``values`` (one value per question), or None when there is none.

    Questions are resampled with replacement and the mean is recomputed each time. With fewer than two values, or
    when all values are equal (e.g. every question of a small slice found its answer first), resampling cannot vary
    the mean, and an interval of zero width would claim a certainty the sample does not give. ``confidence`` is the
    level of the interval; only a comparison planned together with others asks for more than the usual 95 %.
    """
    if len(values) < 2 or min(values) == max(values):
        return None
    result = bootstrap(
        (list(values),),
        fmean,
        vectorized=False,
        n_resamples=_BOOTSTRAP_RESAMPLES,
        confidence_level=confidence,
        rng=_BOOTSTRAP_SEED,
    )
    return Interval(low=float(result.confidence_interval.low), high=float(result.confidence_interval.high))


def share_verdict(successes: int, n: int, minimum: float) -> TargetVerdict:
    """Judge ``successes`` out of ``n`` against a target the share must reach, by its Wilson interval."""
    if share_interval(successes, n).low >= minimum:
        return "met"
    return "not_confirmed" if successes / n >= minimum else "below"
