"""Unit tests of the held-out copy check of a paraphrase set, on made-up questions instead of the local bank."""

import pytest

from evalharness.datasets.check_paraphrases import find_copies

HELD_OUT = [("D-1", "Which of the following is a typical objective of testing a new release of a system?")]


@pytest.mark.covers("US-015", "AC-6")
@pytest.mark.parametrize(
    ("text", "named"),
    [
        ("Which is a typical objective of testing a new release of a system?", True),
        ("What do we usually want to achieve by testing a new version?", False),
        ("What is the difference between the objective of testing and debugging?", False),
    ],
)
def test_a_near_copy_of_a_held_out_question_is_named_and_a_question_on_the_same_topic_is_not(
    text: str, named: bool
) -> None:
    """Checks three texts against one held-out question: a copy with two words left out, the same topic in other
    words, and a question that shares a few common words.

    A pass means only the near copy is named. A failure means either a copied held-out question slips into the pairs
    (the pairs then leak the held-out set into any setting chosen with them), or ordinary phrasing is flagged and
    the check stops being run because it always complains.
    """
    problems = find_copies([("QA-1 rewording", text)], HELD_OUT)

    assert bool(problems) == named, f"{text!r}: problems {problems}"
