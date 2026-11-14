"""Unit tests of the comparison of two runs on the same questions: which numbers go into the change, the flip count
and the verdict. The bootstrap and binomial formulas themselves are scipy's and are not re-tested here."""

import pytest

from evalharness.stats.intervals import Interval
from evalharness.stats.paired import (
    Change,
    ChangeVerdict,
    change_verdict,
    flip_count,
    paired_change,
    planned_confidence,
)

# The S2b comparison with the second book: 146 questions, 4 found only after the change, 7 only before.
_BEFORE = [False] * 4 + [True] * 7 + [True] * 126 + [False] * 9
_AFTER = [True] * 4 + [False] * 7 + [True] * 126 + [False] * 9


@pytest.mark.covers("US-013", "AC-3")
@pytest.mark.parametrize(
    ("low", "high", "verdict"),
    [
        pytest.param(0.014, 0.082, "better", id="whole-interval-above-zero"),
        pytest.param(-0.082, -0.014, "worse", id="whole-interval-below-zero"),
        pytest.param(-0.03, 0.03, "no_meaningful_change", id="inside-the-smallest-meaningful-change"),
        pytest.param(-0.068, 0.021, "underpowered", id="reaches-a-meaningful-loss"),
        pytest.param(-0.02, 0.05, "underpowered", id="reaches-exactly-the-smallest-meaningful-change"),
    ],
)
def test_a_change_is_judged_by_its_interval_against_the_smallest_meaningful_change(
    low: float, high: float, verdict: ChangeVerdict
) -> None:
    """Judges five intervals of a change against a smallest meaningful change of 0.05: one above zero, one below,
    one that stays within 0.03 of zero, the interval of the S2b comparison with the second book (-0.068 to +0.021),
    and one that ends exactly at 0.05.

    A pass means a comparison says "underpowered" whenever these questions cannot rule out a change worth acting
    on, and "no meaningful change" only when they can. A failure most likely means the rule slipped back to "no
    significant difference means no change" (the mistake made by hand in S2b, where p = 0.55 was written down as
    "no measurable change"), or to judging by the width of the interval, which calls a clear gain underpowered.
    """
    actual = change_verdict(Change(change=(low + high) / 2, interval=Interval(low=low, high=high)), 0.05)

    assert actual == verdict, f"interval {low} to {high} against 0.05 gave {actual!r}, expected {verdict!r}"


@pytest.mark.covers("US-013", "AC-3")
def test_runs_with_no_question_changed_are_unchanged() -> None:
    """Compares two runs in which every question scored the same (a repeated run, as `cfea1cad` and `d6fa2ba5`)
    and expects the verdict "unchanged".

    A pass means a repeated run is reported as identical, not as a result the questions cannot judge. A failure
    means the case without an interval (every difference is zero) fell into another branch, so the check of a
    rebuilt stack against its baseline would print a confusing verdict.
    """
    change = paired_change([1.0, 0.0, 1.0], [1.0, 0.0, 1.0])

    assert change_verdict(change, 0.05) == "unchanged", f"identical runs gave {change_verdict(change, 0.05)!r}"


@pytest.mark.covers("US-013", "AC-2")
def test_flipped_questions_are_counted_in_their_direction() -> None:
    """Counts the flips of the S2b comparison with the second book (4 found only after, 7 only before) and expects
    4 gained and 7 lost.

    A pass means the report says which way the questions moved. A failure means the two directions are swapped
    or mixed, so a loss would be reported as a gain and the gained / lost lists would point at the wrong questions.
    """
    flips = flip_count(_BEFORE, _AFTER)

    assert (flips.gained, flips.lost) == (4, 7), f"gained {flips.gained}, lost {flips.lost}; expected 4 and 7"


@pytest.mark.covers("US-013", "AC-2")
def test_the_mcnemar_test_uses_only_the_flipped_questions() -> None:
    """Tests the same comparison (11 flipped questions out of 146) and expects McNemar's exact p of 4 against 7,
    0.549, the value computed by hand in S2b.

    A pass means the p value describes the questions that changed. A failure most likely means the test was run
    over all 146 questions, which makes any comparison look highly significant, because 135 unchanged questions
    then count as evidence.
    """
    flips = flip_count(_BEFORE, _AFTER)

    assert flips.p_value == pytest.approx(0.5488, abs=1e-4), f"p = {flips.p_value}, expected 0.549 from 4 vs 7"


@pytest.mark.covers("US-013", "AC-2")
def test_a_change_and_its_interval_come_from_the_per_question_differences() -> None:
    """Builds 146 questions where 6 are found only after the change and none only before (the S2b comparison with
    the course sections) and expects a change of +6/146 with an interval above zero and below three times the change.

    A pass means the change is after minus before and its interval is built from the per-question differences.
    A failure means the sign is reversed (a gain reported as a loss) or the interval was taken from the scores of
    one run, which says how good that run is, not how much it changed.
    """
    before = [0.0] * 6 + [1.0] * 131 + [0.0] * 9
    after = [1.0] * 6 + [1.0] * 131 + [0.0] * 9

    change = paired_change(before, after)

    assert change.change == pytest.approx(6 / 146), f"change {change.change:+.4f}, expected +{6 / 146:.4f}"
    assert change.interval is not None, "6 gained questions should give an interval"
    assert 0 < change.interval.low < change.change < change.interval.high < 3 * change.change, (
        f"interval {change.interval} is not around the per-question change {change.change:+.4f}"
    )


@pytest.mark.covers("US-013", "AC-5")
def test_one_of_several_planned_comparisons_is_judged_by_a_wider_interval() -> None:
    """Builds 146 questions where 7 are found only after the change and 1 only before, and judges the change twice:
    as the only comparison, and as one of four planned together. Expects "better" alone and "underpowered" as one
    of four.

    A pass means the number of planned comparisons really widens the interval the verdict reads (95 % alone,
    98.75 % for four). A failure means the level is not passed on to the interval, or is computed wrongly, so four
    comparisons are each judged as if they were the only one: with four chances the odds that one of them shows
    "better" by luck are about four times as high, and a search variant could be made the default on it.
    """
    before = [0.0] * 7 + [1.0] * 1 + [1.0] * 138
    after = [1.0] * 7 + [0.0] * 1 + [1.0] * 138

    alone = change_verdict(paired_change(before, after, planned_confidence(1)), 0.05)
    one_of_four = change_verdict(paired_change(before, after, planned_confidence(4)), 0.05)

    assert (alone, one_of_four) == ("better", "underpowered"), (
        f"7 gained and 1 lost of 146 gave {alone!r} alone and {one_of_four!r} as one of four planned comparisons"
    )
