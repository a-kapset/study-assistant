"""The verdict of one cell of the kill matrix: did a check notice a mutant?

A check noticed a mutant when its result on the app with the mutant differs clearly from its result on the app as
configured, measured in the same experiment. The check's own verdict (a target met, a relation held) is not used: a
check that is red without any mutant would otherwise "notice" every mutant, and a target can be too coarse to show a
real loss.

- A check of numbers (the retrieval run, the paraphrase run, the answer run) gives a value from 0 to 1 per question
  or pair: yes / no, or the share of the repeated asks of a question that came out so. The mutant's values minus the
  reference's, question by question, give a change and its interval. The mutant is killed when the
  whole interval lies on one side of zero and the change is at least the smallest change worth acting on, in either
  direction: a mutant that makes a number better also changed what the app does.
- A check made of tests is killed by a test that failed on the mutant and passed on the reference.
- A check that refuses to measure on the mutant noticed it: it saw that the app is not in a state it can measure.
- A question that failed in either run for a reason of the stack the mutant does not declare (no response, a real
  provider error, a search without its model server) measured nothing about the mutant: it is left out of both runs.
  When such questions are more than a few, the stack was not healthy and the cell cannot be read.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from evalharness.mutation.record import TestOutcome
from evalharness.stats.paired import change_verdict, paired_change

# "killed": the check noticed the mutant. "survived": it did not. "underpowered": the change could be worth acting on
# or nothing at all; these questions cannot tell. "invalid": the cell cannot be read (the run is not valid, or the
# reference cannot be compared). "not_applicable": the run made no such check.
type CellVerdict = Literal["killed", "survived", "underpowered", "invalid", "not_applicable"]

# Above this share of failed questions a mutant is killed by its errors; at or under it, a few failed questions look
# like a slip of the network, and the cell is invalid until the run is repeated. Questions that failed for a reason
# the mutant does not declare are left out up to the same share; above it the stack itself was not healthy.
FAILED_SHARE_KILLS = 0.05

# Which verdict of several signals of one check stands for the check: a kill by any signal is a kill.
_RANK: dict[CellVerdict, int] = {"killed": 4, "invalid": 3, "underpowered": 2, "survived": 1, "not_applicable": 0}


@dataclass(frozen=True)
class Cell:
    """The verdict of one cell, and in a few words why."""

    verdict: CellVerdict
    why: str
    # For a mutant that names the failure label it expects: whether the labels of the run name it; None otherwise.
    named: bool | None = None


@dataclass(frozen=True)
class Measured:
    """What one check gave in one run, ready to be compared with another run of the same check.

    Exactly one of the three is set: the refusal of a check that did not measure; the signals of a check of numbers
    (signal name → question or pair id → a value from 0 to 1, None for a question that gave no result); the outcome
    per test of a check made of tests.
    """

    refusal: str | None = None
    signals: Mapping[str, Mapping[str, float | None]] | None = None
    tests: Mapping[str, TestOutcome] | None = None
    # The smallest change of a signal worth acting on, from the file that declares the questions.
    min_meaningful: float = 0.0
    # Why the result cannot stand for the state it claims (e.g. answers served from recordings); None when it can.
    problem: str | None = None
    # A note added to the cell's reason, and whether the failure labels name the expected one (None: not asked).
    note: str = ""
    named: bool | None = None
    # Questions or pairs that failed for a reason of the stack the mutant does not declare: left out of the comparison.
    left_out: frozenset[str] = frozenset()


def judge(reference: Measured, mutant: Measured) -> Cell:
    """Return the verdict of one cell from the check's result on the reference run and on the mutant run."""
    for run, measured in (("the mutant run", mutant), ("the reference", reference)):
        if measured.problem is not None:
            return Cell("invalid", f"{run}: {measured.problem}")
    cell = _judge(reference, mutant)
    if not mutant.note:
        return cell
    return Cell(cell.verdict, f"{cell.why}; {mutant.note}", mutant.named)


def _judge(reference: Measured, mutant: Measured) -> Cell:
    """Judge a cell whose two results are both usable, by the kind of the check."""
    if mutant.refusal is not None:
        return Cell("killed", f"by refusal: {mutant.refusal}")
    if reference.refusal is not None:
        return Cell("invalid", f"the reference refused: {reference.refusal}")
    if mutant.tests is not None or reference.tests is not None:
        return relation_cell(reference.tests or {}, mutant.tests or {})
    if mutant.signals is None or reference.signals is None:
        return Cell("invalid", "the two runs did not measure the same thing")
    left_out = reference.left_out | mutant.left_out
    cases = {case_id for values in (*reference.signals.values(), *mutant.signals.values()) for case_id in values}
    if cases and len(left_out) / len(cases) > FAILED_SHARE_KILLS:
        why = f"{len(left_out)} of {len(cases)} questions failed for a reason the mutant does not declare"
        return Cell("invalid", f"{why}: the stack was not healthy; repeat the run")
    cells = {
        name: signal_cell(
            _without(reference.signals.get(name, {}), left_out), _without(values, left_out), mutant.min_meaningful
        )
        for name, values in mutant.signals.items()
    }
    verdict = max((cell.verdict for cell in cells.values()), key=_RANK.__getitem__)
    why = "; ".join(f"{name}: {cell.why}" for name, cell in cells.items() if cell.verdict == verdict)
    if left_out:
        why += f"; {len(left_out)} question(s) left out of both runs: {', '.join(sorted(left_out))}"
    return Cell(verdict, why)


def _without(values: Mapping[str, float | None], left_out: frozenset[str]) -> dict[str, float | None]:
    """Return the values of a signal without the questions left out of the comparison."""
    return {case_id: value for case_id, value in values.items() if case_id not in left_out}


def signal_cell(
    reference: Mapping[str, float | None], mutant: Mapping[str, float | None], min_meaningful: float
) -> Cell:
    """Judge one signal of a check of numbers, question by question, by the rule of the module."""
    if set(reference) != set(mutant):
        return Cell("invalid", "the runs measured different questions")
    reference_failed = sum(value is None for value in reference.values())
    if reference_failed:
        return Cell("invalid", f"{reference_failed} question(s) of the reference gave no result")
    failed = sum(value is None for value in mutant.values())
    if failed:
        share = failed / len(mutant)
        if share > FAILED_SHARE_KILLS:
            return Cell("killed", f"by errors: {failed} of {len(mutant)} questions gave no result")
        return Cell("invalid", f"{failed} of {len(mutant)} questions gave no result; repeat the run")
    ids = sorted(reference)
    before = [float(reference[case_id] or 0.0) for case_id in ids]
    after = [float(mutant[case_id] or 0.0) for case_id in ids]
    change = paired_change(before, after)
    verdict = change_verdict(change, min_meaningful)
    n = len(ids)
    # Everything below is said in cases (questions or pairs), not in shares, so a reader sees the counts the verdict
    # rests on: how many cases changed each way, how far the true change may reach, and the threshold.
    up = sum(new > old for old, new in zip(before, after, strict=True))
    down = sum(new < old for old, new in zip(before, after, strict=True))
    threshold = min_meaningful * n
    # A share of repeated asks makes half questions possible, so the totals are written as short as they allow.
    if up + down == 0:
        return Cell("survived", f"{sum(before):g} -> {sum(after):g} of {n}: nothing changed")
    seen = f"{sum(before):g} -> {sum(after):g} of {n}; {up + down} changed (+{up}, -{down})"
    if change.interval is None:
        low = high = change.change * n
    else:
        low, high = change.interval.low * n, change.interval.high * n
    # The interval's ends are means of the per-case values, so in cases they are whole numbers, or halves when every
    # question was asked twice; rounding to a half removes the float noise of the bootstrap.
    reach = f"the true change may be from {round(low * 2) / 2:+g} to {round(high * 2) / 2:+g} of {n}"
    if verdict in ("better", "worse") and abs(change.change) >= min_meaningful:
        return Cell("killed", f"{seen}; {reach}, all of it beyond the threshold of {threshold:.1f}: noticed")
    if verdict == "underpowered":
        why = f"{reach}, which holds both no change and the threshold of {threshold:.1f}: too few cases to decide"
        return Cell("underpowered", f"{seen}; {why}")
    if verdict in ("better", "worse"):
        why = f"a real change of {change.change * n:+.1f}, but smaller than the threshold of {threshold:.1f}"
        return Cell("survived", f"{seen}; {reach}; {why}: not noticed")
    return Cell("survived", f"{seen}; {reach}, all of it within the threshold of {threshold:.1f}: not noticed")


def relation_cell(reference: Mapping[str, TestOutcome], mutant: Mapping[str, TestOutcome]) -> Cell:
    """Judge a check made of tests: killed by a test that failed on the mutant and passed on the reference.

    A test that did not run on the mutant (error, skipped, missing), or did not pass on the reference, says nothing
    about the mutant: such a test makes the cell invalid unless another test killed it.
    """
    killers = sorted(
        name for name, outcome in mutant.items() if outcome == "failed" and reference.get(name) == "passed"
    )
    if killers:
        return Cell("killed", f"failed: {', '.join(killers)}")
    unusable = sorted(
        name
        for name in reference.keys() | mutant.keys()
        if reference.get(name) != "passed" or mutant.get(name) not in ("passed", "failed")
    )
    if unusable:
        return Cell("invalid", f"not comparable: {', '.join(unusable)}")
    return Cell("survived", f"{len(mutant)} test(s) passed on both")
