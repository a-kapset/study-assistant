"""Unit tests of the held-out check of a paraphrase set: the line measured from questions of different chapters and
the texts flagged above it, on small hand-made unit vectors instead of the embedding server."""

import math

import pytest

from evalharness.datasets.check_paraphrases import different_question_line, find_close


def _unit(angle_degrees: float) -> list[float]:
    """Return a unit vector in the plane at ``angle_degrees``; two such vectors have the cosine of the angle between."""
    angle = math.radians(angle_degrees)
    return [math.cos(angle), math.sin(angle)]


@pytest.mark.covers("US-015", "AC-6")
def test_the_line_is_the_highest_similarity_of_two_questions_from_different_chapters() -> None:
    """Builds three questions: two of chapter 1 almost the same (5 degrees apart) and one of chapter 2, 40 degrees
    from the first and 35 from the second.

    A pass means the line is cos 35 degrees, the closest pair across chapters, and the near-identical pair of the same
    chapter is not taken into account. A failure means questions of one chapter, which may really ask nearly the same
    thing, set the line; it would then be so high that no retold held-out question is ever flagged.
    """
    questions = [("1", _unit(0)), ("1", _unit(5)), ("2", _unit(40))]

    line = different_question_line(questions)

    assert line == pytest.approx(math.cos(math.radians(35))), f"line {line}"


@pytest.mark.covers("US-015", "AC-6")
def test_a_text_is_flagged_only_when_it_is_closer_to_a_held_out_question_than_the_line() -> None:
    """Checks two texts against one held-out question with the line at cos 20 degrees: one text 10 degrees away
    (closer than any two different questions) and one exactly 20 degrees away (as close as two different questions).

    A pass means only the first is flagged, closest first, with the held-out id. A failure means either a held-out
    question that came back in other words is not named, or a text merely as close as two different questions is
    flagged and the list fills with pairs that are not the same question.
    """
    held_out = [("D-1", _unit(0))]
    texts = [("QA-1 question", _unit(10)), ("QA-2 question", _unit(20))]

    close = find_close(texts, held_out, math.cos(math.radians(20)) + 1e-9)

    assert [(c.label, c.held_out_id) for c in close] == [("QA-1 question", "D-1")], f"flagged {close}"


@pytest.mark.covers("US-015", "AC-6")
def test_a_text_cleared_by_reading_is_not_flagged_again_for_that_held_out_question_only() -> None:
    """Checks one text that is close to two held-out questions, after a reading cleared it for D-1 only.

    A pass means it is still flagged for D-2. A failure means either the check fails on every run for a pair a person
    already read and found different (so it is no longer run), or a clearing for one held-out question also hides the
    same text's closeness to another one, which nobody has read.
    """
    held_out = [("D-1", _unit(0)), ("D-2", _unit(2))]
    texts = [("QA-1 question", _unit(1))]

    close = find_close(texts, held_out, math.cos(math.radians(20)), cleared={("QA-1 question", "D-1")})

    assert [(c.label, c.held_out_id) for c in close] == [("QA-1 question", "D-2")], f"flagged {close}"
