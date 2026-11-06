"""Unit tests of the Q&A set check: which key terms are found where, on made-up passages instead of the stored
course."""

import pytest

from evalharness.datasets.check_qa_set import find_problems
from evalharness.datasets.qa_set import QaCase, QaSet, QaTargets

TARGETS = QaTargets(answered_min=0.85, gold_cited_min=0.75, unsupported_answers_max=0)


def _set(case: QaCase) -> QaSet:
    """Build a set of the fake course that holds ``case`` only."""
    return QaSet(course_id="evalharness_course", targets=TARGETS, cases=(case,))


@pytest.mark.covers("US-011", "AC-10")
@pytest.mark.parametrize(
    ("section", "text", "passes"),
    [
        ("4.2", "Boundary value analysis uses the edges.", True),
        ("4.2.2", "Boundary value analysis uses the edges.", True),
        ("4.2", "BOUNDARY VALUE\nanalysis is a technique.", True),
        ("4.20", "Boundary value analysis uses the edges.", False),
        ("5.1", "Boundary value analysis uses the edges.", False),
        ("4.2", "Equivalence partitioning only.", False),
    ],
)
def test_a_term_of_an_answerable_question_must_occur_in_its_gold_section(section: str, text: str, passes: bool) -> None:
    """Checks the answerable question with gold section 4.2 and the key term "boundary value analysis" against one
    passage: in 4.2, in its subsection 4.2.2, in 4.2 across a line break in capitals, in 4.20 (which only looks like
    a subsection), in 5.1, and a passage of 4.2 without the term.

    A pass means a question passes only when its gold section holds its term, the way the answer oracle counts a
    citation of the section or a subsection. A failure means the check passes a wrong gold section (a correct answer
    would then score as citing another section) or rejects a right one because a PDF line break split the term.
    """
    case = QaCase(
        id="QA-1",
        question="How does boundary value analysis work?",
        gold_section="4.2",
        key_terms=("boundary value analysis",),
    )

    problems = find_problems(_set(case), [(section, text)])

    assert (problems == []) == passes, f"passage of {section} {text!r}: problems {problems}"


@pytest.mark.covers("US-011", "AC-10")
@pytest.mark.parametrize(
    ("term", "section", "text", "passes"),
    [
        ("mutation testing", "6.2", "Test automation has benefits and risks.", True),
        ("mutation testing", "6.2", "Mutation testing is not described here.", False),
        ("mutation testing", None, "MUTATION\n  TESTING appears in a passage without a section.", False),
        ("pact", "1.4", "The impact of a change is analysed.", True),
        ("pact", "2.2", "Pacts between services are not tested here.", False),
    ],
)
def test_a_term_of_an_unanswerable_question_must_occur_nowhere_in_the_course(
    term: str, section: str | None, text: str, passes: bool
) -> None:
    """Checks an unanswerable question against one passage: "mutation testing" in a passage without it, in a
    passage of 6.2 with it, and across a line break in capitals in a passage without a section; "pact" inside the
    word "impact" and at the start of "Pacts".

    A pass means a question counts as unanswerable only when no passage of the course mentions its term as a word.
    A failure means either a question the course does answer stays in the set as unanswerable, so a correct answer
    counts as an unsupported answer, which fails the whole run; or a term is found inside another word, so a good
    question is thrown out (the first check of v1 rejected "pact" for every passage that says "impact").
    """
    case = QaCase(id="QA-U1", question="A question the course does not answer?", key_terms=(term,))

    problems = find_problems(_set(case), [(section, text)])

    assert (problems == []) == passes, f"passage of {section} {text!r}: problems {problems}"


@pytest.mark.covers("US-011", "AC-10")
def test_a_question_without_key_terms_is_a_problem() -> None:
    """Checks an answerable question that lists no key terms against a passage of its gold section.

    A pass means every question of a set is checked. A failure means a question without terms passes silently, so
    the check proves nothing about it while its summary says every case passed.
    """
    case = QaCase(id="QA-1", question="What is testing?", gold_section="1.1")

    problems = find_problems(_set(case), [("1.1", "Testing is a set of activities.")])

    assert problems == ["QA-1: no key terms"], f"problems: {problems}"
