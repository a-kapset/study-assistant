"""Unit tests of the intervals and target verdicts a run reports: which method is used and how a verdict is read
from an interval. The interval formulas themselves are scipy's and are not re-tested here."""

import pytest

from evalharness.stats.intervals import TargetVerdict, mean_interval, share_interval, share_verdict


@pytest.mark.covers("US-012", "AC-2")
@pytest.mark.parametrize(
    ("successes", "n", "minimum", "verdict"),
    [
        pytest.param(19, 20, 0.85, "not_confirmed", id="share-above-target-lower-end-below"),
        pytest.param(17, 20, 0.85, "not_confirmed", id="share-exactly-at-target"),
        pytest.param(20, 20, 0.80, "met", id="lower-end-above-target"),
        pytest.param(15, 20, 0.85, "below", id="share-below-target"),
    ],
)
def test_a_share_meets_its_target_only_when_the_lower_end_reaches_it(
    successes: int, n: int, minimum: float, verdict: TargetVerdict
) -> None:
    """Judges shares of 20 questions against a target: one whose 95 % interval lies above the target, two whose
    measured share reaches the target (one of them exactly) while the lower end of the interval does not, one
    under the target.

    A pass means a target counts as met only when even the lower end of the interval reaches it, and a share that
    only looks good on this sample is reported as not confirmed. A failure means the verdict went back to the
    measured share alone: 19 of 20 would again be called "met" against 0.85 although another 20 similar questions
    could well give 0.80. CI and every target line of a report read this verdict, so nothing else would notice.
    """
    actual = share_verdict(successes, n, minimum)

    assert actual == verdict, f"{successes} of {n} against {minimum} gave {actual!r}, expected {verdict!r}"


@pytest.mark.covers("US-012", "AC-1")
def test_a_share_interval_is_the_wilson_interval() -> None:
    """Computes the interval of 19 of 20 and expects the Wilson interval's lower end, 0.764.

    A pass means the reports print the interval they name: "Wilson" stands next to every share. A failure means
    another method is used, most likely scipy's default "exact" (lower end 0.751) because the method argument was
    lost; the target verdicts would then rest on a different, wider interval than the one the report claims.
    """
    interval = share_interval(19, 20)

    assert interval.low == pytest.approx(0.7639, abs=1e-4), f"lower end {interval.low:.4f} is not Wilson's 0.7639"


@pytest.mark.covers("US-012", "AC-1")
@pytest.mark.parametrize(
    "values",
    [
        pytest.param([1.0] * 5, id="every-question-scored-the-same"),
        pytest.param([0.5], id="one-question"),
    ],
)
def test_a_mean_has_no_interval_when_resampling_cannot_vary_it(values: list[float]) -> None:
    """Asks for the bootstrap interval of five equal values (every question of a slice found its answer first, as
    in chapter 1 of the invented course) and of a single value; expects no interval in both cases.

    A pass means such a slice is printed as having no interval. A failure means the report either stops with an
    error (scipy refuses one value) or prints an interval of "nan" or of zero width, which claims a certainty the
    sample does not give.
    """
    interval = mean_interval(values)

    assert interval is None, f"values {values} got the interval {interval}"
