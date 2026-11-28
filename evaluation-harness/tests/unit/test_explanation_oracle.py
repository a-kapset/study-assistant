"""The rule that finds which options an explanation names: the api tests of explanations trust it as their
reference, so a mistake here would pass an explanation that names nothing or fail one that names everything."""

import pytest

from evalharness.oracles.explanation import options_not_named


@pytest.mark.parametrize(
    "text",
    [
        "Option b is correct because the handbook says so.",
        'The right choice is option "b".',
        "The correct answer is **option b**: see the passage.",
        "Options b and option d are both listed, but only OPTION (b) is right.",
        "You selected option c; the key is option\nb.",
    ],
    ids=["plain", "quoted", "markdown", "upper case and brackets", "line break"],
)
def test_an_option_written_as_the_word_option_and_its_id_is_named(text: str) -> None:
    """Gives texts that name option b in the forms a model really writes: plain, in quotes, in markdown stars, in
    brackets with the word in capitals, and across a line break. Expects b to count as named in each.

    What a failure tells: the rule is too strict and would fail good explanations of the running app for their
    punctuation alone.
    """
    assert options_not_named(text, ["b"]) == [], f"option b was not found in {text!r}"


@pytest.mark.parametrize(
    "text",
    [
        "The second answer is correct because the handbook says so.",
        "b is correct.",
        "A keeper has the option between two lamps.",
        "Option be careful: this is not an id.",
        "Option a is correct.",
    ],
    ids=["no id at all", "id without the word", "the word without an id", "id inside a longer word", "another id"],
)
def test_text_that_does_not_write_option_and_the_id_names_nothing(text: str) -> None:
    """Gives texts that do not name option b: no id, a bare letter, the word "option" in its everyday sense, the
    letter as the start of a longer word, and another option's id. Expects b reported as not named in each.

    What a failure tells: the rule is too loose. A bare letter must not count: "a" is also an English word, so
    almost any text would then "name" option a, and the api test could never fail.
    """
    assert options_not_named(text, ["b"]) == ["b"], f"option b was wrongly found in {text!r}"


def test_every_id_is_judged_on_its_own_and_missing_ones_come_sorted() -> None:
    """Gives a text that names options a and c and asks about d, a, c and b. Expects exactly b and d back, in
    sorted order, each once.

    What a failure tells: one named option hid a missing one, or the report of missing ids is not stable, so a
    failure message of the api test would name the wrong options.
    """
    text = "Option a and option c are correct; the others are not."

    assert options_not_named(text, ["d", "a", "c", "b", "d"]) == ["b", "d"], "expected only b and d as not named"
