"""Unit tests of the overlap calibration: which pairs form the two groups and how the value is read from them, on
small invented pairs instead of the running stack."""

import pytest

from evalharness.datasets.paraphrase_set import ParaphrasePair
from evalharness.evaluation.calibrate_overlap import calibrated_min_overlap, nearby_different, unchanged_meaning


def _pair(case_id: str, gold_section: str) -> ParaphrasePair:
    """Build a pair whose texts name the case, answered by ``gold_section``."""
    return ParaphrasePair(
        case_id=case_id, question=case_id, paraphrase=f"{case_id}?", gold_section=gold_section, term_removed=True
    )


@pytest.mark.covers("US-015", "AC-7")
def test_the_calibrated_overlap_is_one_more_than_any_pair_of_different_questions_shared() -> None:
    """Reads the value from different-question pairs that shared at most 1 of 5 passages, and from pairs that shared
    all 5.

    A pass means 2 of 5 (0.4) in the first case: the smallest overlap that topic alone never gave; and 1.0 in the
    second, the strictest rule, because no value separates them. A failure means the value is the highest overlap of
    different questions itself, so a pair that shares only as much as two different questions would count as stable.
    """
    values = (calibrated_min_overlap([0, 0, 1, 0], 5), calibrated_min_overlap([5, 2], 5))

    assert values == (0.4, 1.0), f"calibrated values {values}"


@pytest.mark.covers("US-015", "AC-7")
def test_each_question_is_paired_with_the_next_question_of_its_chapter_from_another_section() -> None:
    """Pairs four questions: two of section 4.2, one of 4.3 and one of 5.1, in that order.

    A pass means both 4.2 questions get the 4.3 one (never each other), 4.3 gets the first 4.2 question by going
    round, and 5.1 has no partner. A failure means two questions of the same section, which may ask nearly the same
    thing, or questions of distant chapters, which share nothing anyway, enter the group of nearby different
    questions; the calibrated value would then be too high or too low.
    """
    pairs = [_pair("QA-1", "4.2"), _pair("QA-2", "4.2"), _pair("QA-3", "4.3"), _pair("QA-4", "5.1")]

    partners = [(a.case_id, b.case_id) for a, b in nearby_different(pairs)]

    assert partners == [("QA-1", "QA-3"), ("QA-2", "QA-3"), ("QA-3", "QA-1")], f"partners {partners}"


@pytest.mark.covers("US-015", "AC-7")
def test_the_unchanged_question_differs_only_in_letter_case_and_punctuation() -> None:
    """Changes "What is Test-Driven Development (TDD)?" into the form used as "the same question".

    A pass means only capitals and punctuation go, and the hyphen inside a word stays: "what is test-driven
    development tdd". A failure means the group meant to change nothing also drops or joins words, and its overlap
    would no longer show the noise of the search alone.
    """
    text = unchanged_meaning("What is Test-Driven Development (TDD)?")

    assert text == "what is test-driven development tdd", f"unchanged form {text!r}"
