"""The verdict of one cell of the kill matrix: did a check notice a mutant?

A check noticed a mutant when its result on the app with the mutant differs clearly from its result on the app as
configured, measured in the same experiment. The check's own verdict (a target met, a relation held) is not used: a
check that is red without any mutant would otherwise "notice" every mutant, and a target can be too coarse to show a
real loss.

- A check of numbers (the retrieval run, the paraphrase run) gives a yes / no per question or pair. The mutant's
  values minus the reference's, question by question, give a change and its interval. The mutant is killed when the
  whole interval lies on one side of zero and the change is at least the smallest change worth acting on, in either
  direction: a mutant that makes a number better also changed what the app does.
- A check made of tests is killed by a test that failed on the mutant and passed on the reference.
- A check that refuses to measure on the mutant noticed it: it saw that the app is not in a state it can measure.
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

# Above this share of failed searches a mutant is killed by its errors; at or under it, a few failed searches look
# like a slip of the network, and the cell is invalid until the run is repeated.
FAILED_SHARE_KILLS = 0.05

# Which verdict of several signals of one check stands for the check: a kill by any signal is a kill.
_RANK: dict[CellVerdict, int] = {"killed": 4, "invalid": 3, "underpowered": 2, "survived": 1, "not_applicable": 0}


@dataclass(frozen=True)
class Cell:
    """The verdict of one cell, and in a few words why."""

    verdict: CellVerdict
    why: str


@dataclass(frozen=True)
class Measured:
    """What one check gave in one run, ready to be compared with another run of the same check.

    Exactly one of the three is set: the refusal of a check that did not measure; the signals of a check of numbers
    (signal name → question or pair id → yes / no, None for a search that gave no result); the outcome per test of a
    check made of tests.
    """

    refusal: str | None = None
    signals: Mapping[str, Mapping[str, bool | None]] | None = None
    tests: Mapping[str, TestOutcome] | None = None
    # The smallest change of a signal worth acting on, from the file that declares the questions.
    min_meaningful: float = 0.0


def judge(reference: Measured, mutant: Measured) -> Cell:
    """Return the verdict of one cell from the check's result on the reference run and on the mutant run."""
    if mutant.refusal is not None:
        return Cell("killed", f"by refusal: {mutant.refusal}")
    if reference.refusal is not None:
        return Cell("invalid", f"the reference refused: {reference.refusal}")
    if mutant.tests is not None or reference.tests is not None:
        return relation_cell(reference.tests or {}, mutant.tests or {})
    if mutant.signals is None or reference.signals is None:
        return Cell("invalid", "the two runs did not measure the same thing")
    cells = {
        name: signal_cell(reference.signals.get(name, {}), values, mutant.min_meaningful)
        for name, values in mutant.signals.items()
    }
    verdict = max((cell.verdict for cell in cells.values()), key=_RANK.__getitem__)
    why = "; ".join(f"{name}: {cell.why}" for name, cell in cells.items() if cell.verdict == verdict)
    return Cell(verdict, why)


def signal_cell(reference: Mapping[str, bool | None], mutant: Mapping[str, bool | None], min_meaningful: float) -> Cell:
    """Judge one yes / no signal of a check of numbers, question by question, by the rule of the module."""
    if set(reference) != set(mutant):
        return Cell("invalid", "the runs measured different questions")
    reference_failed = sum(value is None for value in reference.values())
    if reference_failed:
        return Cell("invalid", f"{reference_failed} search(es) of the reference gave no result")
    failed = sum(value is None for value in mutant.values())
    if failed:
        share = failed / len(mutant)
        if share > FAILED_SHARE_KILLS:
            return Cell("killed", f"by errors: {failed} of {len(mutant)} searches gave no result")
        return Cell("invalid", f"{failed} of {len(mutant)} searches gave no result; repeat the run")
    ids = sorted(reference)
    before = [float(bool(reference[case_id])) for case_id in ids]
    after = [float(bool(mutant[case_id])) for case_id in ids]
    change = paired_change(before, after)
    verdict = change_verdict(change, min_meaningful)
    n = len(ids)
    # Everything below is said in cases (questions or pairs), not in shares, so a reader sees the counts the verdict
    # rests on: how many cases changed each way, how far the true change may reach, and the threshold.
    up = sum(new > old for old, new in zip(before, after, strict=True))
    down = sum(new < old for old, new in zip(before, after, strict=True))
    threshold = min_meaningful * n
    if up + down == 0:
        return Cell("survived", f"{sum(before):.0f} -> {sum(after):.0f} of {n}: nothing changed")
    seen = f"{sum(before):.0f} -> {sum(after):.0f} of {n}; {up + down} changed (+{up}, -{down})"
    if change.interval is None:
        low = high = change.change * n
    else:
        low, high = change.interval.low * n, change.interval.high * n
    # The interval's ends are means of whole cases, so they are whole numbers of cases.
    reach = f"the true change may be from {low:+.0f} to {high:+.0f} of {n}"
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
