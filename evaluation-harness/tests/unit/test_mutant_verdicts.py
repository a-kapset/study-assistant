"""Unit tests of the verdict of one cell of the kill matrix: made-up results of a check on the reference run and on a
mutant run; never the stack."""

import pytest

from evalharness.mutation.record import TestOutcome
from evalharness.mutation.verdicts import CellVerdict, Measured, judge, relation_cell, signal_cell

# The question set of the first course has 146 questions; the smallest change worth acting on is 0.05 (about 7).
QUESTIONS = 146
MIN_MEANINGFUL = 0.05


def _answers(found: int, total: int = QUESTIONS, failed: int = 0) -> dict[str, bool | None]:
    """Return a signal over ``total`` questions: the first ``found`` found, then ``failed`` without a result."""
    values: dict[str, bool | None] = {}
    for index in range(total):
        values[f"q{index:03}"] = None if index >= total - failed else index < found
    return values


def _moved(lost: int, gained: int, found: int = 120, total: int = QUESTIONS) -> dict[str, bool | None]:
    """Return the reference of ``_answers(found)`` with ``lost`` found questions missed and ``gained`` missed ones
    found."""
    values = _answers(found, total)
    for index in range(lost):
        values[f"q{index:03}"] = False
    for index in range(found, found + gained):
        values[f"q{index:03}"] = True
    return values


@pytest.mark.covers("US-019", "AC-6")
@pytest.mark.parametrize(
    ("mutant", "reference", "expected"),
    [
        pytest.param(_moved(lost=19, gained=0), _answers(120), "killed", id="19 questions lost"),
        pytest.param(_moved(lost=0, gained=19), _answers(120), "killed", id="19 questions gained"),
        pytest.param(_moved(lost=6, gained=0), _answers(120), "survived", id="a real change under the smallest one"),
        pytest.param(_moved(lost=2, gained=2), _answers(120), "survived", id="changes that cancel out"),
        pytest.param(_answers(120), _answers(120), "survived", id="nothing changed"),
        pytest.param(
            _moved(lost=2, gained=1, found=6, total=10), _answers(6, total=10), "underpowered", id="too few questions"
        ),
    ],
)
def test_a_signal_kills_only_beyond_the_smallest_meaningful_change_in_either_direction(
    mutant: dict[str, bool | None], reference: dict[str, bool | None], expected: CellVerdict
) -> None:
    """The found / not found answers of a mutant run are compared with the reference question by question. A loss
    or a gain of 19 of 146 questions must kill; a change of 6 questions (under the smallest change worth acting on),
    changes that cancel out and no change must survive; 3 changes among 10 questions must be underpowered.

    A pass says the rule of the matrix: a kill needs an interval off zero and a change of at least 0.05, either way.
    A failure points to signal_cell in verdicts.py. A one-sided rule would let a mutant that "improves" a number live;
    a rule without the smallest change would kill on noise; reading underpowered as survived would hide a cell that
    cannot tell.
    """
    cell = signal_cell(reference, mutant, MIN_MEANINGFUL)

    assert cell.verdict == expected, f"verdict {cell.verdict}, expected {expected}: {cell.why}"


@pytest.mark.covers("US-019", "AC-6")
def test_a_check_that_refuses_on_the_mutant_has_noticed_it() -> None:
    """The check measured the reference and refused to measure the mutant, as a retrieval run refuses a degraded
    stack. The cell must be killed and say it was by refusal.

    A pass says that the loudest mutants, the ones that make a check stop, count as noticed. A failure points to
    judge in verdicts.py. Without the rule a mutant that takes a model server away would read as not seen.
    """
    reference = Measured(signals={"found": _answers(120)}, min_meaningful=MIN_MEANINGFUL)
    mutant = Measured(refusal="stack_not_ready: GET /ready says degraded")

    cell = judge(reference, mutant)

    assert cell.verdict == "killed" and "by refusal" in cell.why, f"cell: {cell}"


@pytest.mark.covers("US-019", "AC-6")
@pytest.mark.parametrize(
    ("reference", "mutant", "expected"),
    [
        pytest.param({"t": "passed"}, {"t": "failed"}, "killed", id="failed on the mutant only"),
        pytest.param({"t": "passed"}, {"t": "passed"}, "survived", id="passed on both"),
        pytest.param({"t": "passed"}, {"t": "error"}, "invalid", id="could not run on the mutant"),
        pytest.param({"t": "failed"}, {"t": "failed"}, "invalid", id="failed on the reference too"),
    ],
)
def test_a_test_kills_only_when_it_failed_on_the_mutant_and_passed_on_the_reference(
    reference: dict[str, TestOutcome], mutant: dict[str, TestOutcome], expected: CellVerdict
) -> None:
    """The outcome of one relation test on the reference and on the mutant. Only "passed, then failed" may kill; an
    error on the mutant and a failure on both runs must make the cell invalid.

    A pass says that a kill by a test is a change the mutant caused. A failure points to relation_cell in verdicts.py.
    Without it a fixture that could not reach the app would count as a kill, and a test red everywhere would look
    like a perfect killer.
    """
    cell = relation_cell(reference, mutant)

    assert cell.verdict == expected, f"verdict {cell.verdict}, expected {expected}: {cell.why}"


@pytest.mark.covers("US-019", "AC-6")
@pytest.mark.parametrize(
    ("mutant", "reference", "expected"),
    [
        pytest.param(_answers(110, failed=10), _answers(120), "killed", id="10 of 146 searches failed"),
        pytest.param(_answers(118, failed=2), _answers(120), "invalid", id="2 of 146 searches failed"),
        pytest.param(_answers(120), _answers(118, failed=2), "invalid", id="the reference had failed searches"),
    ],
)
def test_failed_searches_kill_only_above_five_percent_and_never_count_as_misses(
    mutant: dict[str, bool | None], reference: dict[str, bool | None], expected: CellVerdict
) -> None:
    """Some searches gave no result. Ten of 146 on the mutant must kill by errors; two must make the cell invalid;
    failed searches on the reference must make it invalid.

    A pass says that a search without a result is never read as "not found". A failure points to signal_cell in
    verdicts.py. Counting failures as misses would show a broken stack as worse retrieval; killing on two failures
    would read a slip of the network as a defect.
    """
    cell = signal_cell(reference, mutant, MIN_MEANINGFUL)

    assert cell.verdict == expected, f"verdict {cell.verdict}, expected {expected}: {cell.why}"


def _failed_ids(values: dict[str, bool | None]) -> frozenset[str]:
    """Return the questions of a signal that gave no result."""
    return frozenset(case_id for case_id, value in values.items() if value is None)


@pytest.mark.covers("US-019", "AC-6")
@pytest.mark.parametrize(
    ("reference", "mutant", "not_declared", "expected"),
    [
        pytest.param(
            _answers(119, failed=1), _moved(lost=19, gained=0), "reference", "killed", id="a slip on the reference"
        ),
        pytest.param(_answers(120), _answers(118, failed=2), "mutant", "survived", id="slips on the mutant"),
        pytest.param(_answers(120), _answers(110, failed=10), "none", "killed", id="failures the mutant declares"),
    ],
)
def test_failures_the_mutant_does_not_declare_are_left_out_of_both_runs(
    reference: dict[str, bool | None], mutant: dict[str, bool | None], not_declared: str, expected: CellVerdict
) -> None:
    """Judges a cell whose failed questions are marked as not declared by the mutant (a timeout, a real provider
    error) on the reference or on the mutant, or are not marked (the mutant's own failures). Expects one slip on
    the reference not to stop a loss of 19 questions from killing, two slips on an otherwise equal mutant to
    survive, and ten failures the mutant declares to kill by errors as before.

    What a pass tells: a question that failed for a reason of the stack is taken out of both runs and the rest is
    judged. A failure points to judge in verdicts.py. Worth having because without it one timeout on the
    reference leaves a whole column unreadable, and an outage during a mutant's run would kill that mutant.
    """
    measured = {
        "reference": Measured(signals={"found": reference}, min_meaningful=MIN_MEANINGFUL),
        "mutant": Measured(signals={"found": mutant}, min_meaningful=MIN_MEANINGFUL),
    }
    if not_declared != "none":
        side = measured[not_declared]
        failed = _failed_ids(reference if not_declared == "reference" else mutant)
        measured[not_declared] = Measured(signals=side.signals, min_meaningful=MIN_MEANINGFUL, left_out=failed)

    cell = judge(measured["reference"], measured["mutant"])

    assert cell.verdict == expected, f"verdict {cell.verdict}, expected {expected}: {cell.why}"


@pytest.mark.covers("US-019", "AC-6")
def test_a_cell_with_more_than_five_percent_of_failures_not_declared_is_invalid() -> None:
    """Judges a cell where 10 of 146 questions of the mutant run failed for a reason the mutant does not declare.
    Expects the cell invalid and to say the stack was not healthy.

    What a pass tells: a run in which the stack failed often is repeated, not judged on what was left. A failure
    points to judge in verdicts.py. Worth having because an outage over part of a run would otherwise leave a
    remnant of questions that may differ from the whole set, and the cell would read as a result.
    """
    mutant = _answers(110, failed=10)
    reference = Measured(signals={"found": _answers(120)}, min_meaningful=MIN_MEANINGFUL)
    measured = Measured(signals={"found": mutant}, min_meaningful=MIN_MEANINGFUL, left_out=_failed_ids(mutant))

    cell = judge(reference, measured)

    assert cell.verdict == "invalid" and "not healthy" in cell.why, f"cell: {cell}"
