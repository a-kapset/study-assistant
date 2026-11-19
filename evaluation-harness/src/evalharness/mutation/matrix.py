"""Command that builds the kill matrix of one experiment from its mutant run records: which check noticed which mutant.

    poetry run python -m evalharness.mutation.matrix <experiment id>

Run it from evaluation-harness/, like the mutant runs (the records name their dataset files relative to it). Every
cell is a mutant and a check, judged against the experiment's first run without a mutant (the reference) by the rule
of verdicts.py. A cell is marked by what the catalog expected before the first run: an expected kill, a blind spot (a
kill was expected and did not come), an unexpected kill, or not aimed at. Later runs without a mutant are judged the
same way, and a kill there is a false alarm. Under the matrix, every check is summed up by the mutants it was expected
to notice and did, its false alarms, the mutants only it noticed, and whether it costs model calls. Exit code 0 when
the matrix was built; 2 when the records cannot make one matrix.
"""

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

from evalharness.datasets.paraphrase_set import load_paraphrase_set
from evalharness.datasets.retrieval_gold import load_declaration
from evalharness.evaluation.paraphrase_record import ParaphraseRunRecord
from evalharness.evaluation.record import PassResult, RunRecord
from evalharness.mutation.catalog import CheckName, Component
from evalharness.mutation.record import RECORD_FILE, CheckResult, MutantRunRecord
from evalharness.mutation.verdicts import Cell, Measured, judge

RUNS_DIR = Path("runs/mutants")

type Stage = Literal["ingestion", "embedding", "search", "answer"]
# The stage each check measures: all checks without model calls look at what a search returns.
CHECK_STAGE: dict[CheckName, Stage] = {"retrieval_gold": "search", "paraphrase": "search", "search_relations": "search"}
# What one run of each check costs: "none" without model calls.
CHECK_COST: dict[CheckName, str] = {"retrieval_gold": "none", "paraphrase": "none", "search_relations": "none"}
# A request passes the stages in this order; a fault of an earlier stage reaches the later ones.
_STAGE_ORDER: tuple[Stage, ...] = ("ingestion", "embedding", "search", "answer")
_COMPONENT_STAGE: dict[Component, Stage] = {
    "ingestion": "ingestion",
    "embedding": "embedding",
    "search": "search",
    "answer": "answer",
    "provider": "answer",
}
_CHECK_ORDER: tuple[CheckName, ...] = get_args(CheckName.__value__)

# Reads what one check gave in one run from its result, given the folder of that run.
type Reader = Callable[[CheckName, CheckResult, Path], Measured]


class MatrixRefused(ValueError):
    """The records cannot make one matrix; the message says why."""


@dataclass(frozen=True)
class Run:
    """One mutant run of the experiment: its record and the folder that holds the records of its checks."""

    record: MutantRunRecord
    folder: Path


@dataclass(frozen=True)
class MarkedCell:
    """A cell's verdict with what the catalog expected of it."""

    cell: Cell
    # The catalog named this check among the mutant's expected killers.
    planned: bool
    # "expected kill", "blind spot", "unexpected kill", "not aimed at", "false alarm", or "" when the verdict says
    # neither kill nor survival.
    mark: str
    # For a kill: "at the faulty stage" when the check measures the stage the mutant makes faulty, "inherited" when
    # the fault comes from an earlier stage; "" otherwise.
    stage: str


@dataclass(frozen=True)
class Row:
    """One run of the matrix and its cells."""

    run: Run
    # A run without a mutant after the reference: its kills are false alarms.
    no_mutant: bool
    cells: dict[CheckName, MarkedCell]


@dataclass(frozen=True)
class CheckSummary:
    """What one check caught over the whole matrix."""

    expected_killed: int
    expected: int
    false_alarms: int
    no_mutant_runs: int
    # Mutants this check killed and no other check did.
    only_this: tuple[str, ...]
    cost: str


@dataclass(frozen=True)
class Matrix:
    """The kill matrix of one experiment."""

    experiment_id: str
    reference: Run
    checks: tuple[CheckName, ...]
    rows: tuple[Row, ...]
    summaries: dict[CheckName, CheckSummary]


def main(argv: Sequence[str]) -> int:
    """Run the command and return its exit code."""
    parser = argparse.ArgumentParser(prog="python -m evalharness.mutation.matrix")
    parser.add_argument("experiment_id", help="the id the mutant run command printed")
    parser.add_argument(
        "--runs-dir", type=Path, default=RUNS_DIR, help=f"where the mutant runs are (default {RUNS_DIR})"
    )
    args = parser.parse_args(argv)
    try:
        matrix = build_matrix(load_experiment(args.runs_dir, args.experiment_id), read_measured)
    except MatrixRefused as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    print(render(matrix))
    return 0


def load_experiment(runs_dir: Path, experiment_id: str) -> list[Run]:
    """Read every mutant run of the experiment, in the order they started; an id may be given by its beginning."""
    runs = []
    for path in sorted(runs_dir.glob(f"*/{RECORD_FILE}")):
        record = MutantRunRecord.model_validate_json(path.read_bytes())
        if record.experiment_id is not None and record.experiment_id.startswith(experiment_id):
            runs.append(Run(record, path.parent))
    if not runs:
        raise MatrixRefused(f"no mutant run of experiment {experiment_id!r} under {runs_dir}")
    if len({run.record.experiment_id for run in runs}) > 1:
        raise MatrixRefused(f"{experiment_id!r} is the beginning of more than one experiment id")
    return sorted(runs, key=lambda run: run.record.started_at)


def build_matrix(runs: Sequence[Run], read: Reader) -> Matrix:
    """Judge every mutant and check of the runs against the first valid run without a mutant.

    Refused when the runs do not share one catalog version, harness version and first state of the app (they would
    not be one experiment), or when the first run without a mutant is missing or not valid.
    """
    first = runs[0].record
    for field in ("experiment_id", "catalog_sha256", "harness_commit", "harness_dirty", "app_settings"):
        values = {repr(getattr(run.record, field)) for run in runs}
        if len(values) > 1:
            raise MatrixRefused(f"the runs differ in {field}: they are not one experiment")
    no_mutant = [run for run in runs if run.record.mutant.kind == "baseline"]
    if not no_mutant:
        raise MatrixRefused("no run without a mutant: there is nothing to judge the mutants against")
    reference = no_mutant[0]
    if reference.record.problem() is not None:
        raise MatrixRefused(f"the reference run is not valid: {reference.record.problem()}")
    present = {name for run in runs for name in run.record.checks}
    checks = tuple(name for name in _CHECK_ORDER if name in present)
    rows = tuple(_row(run, reference, checks, read) for run in runs if run is not reference)
    summaries = {name: _summary(name, rows) for name in checks}
    return Matrix(first.experiment_id or "", reference, checks, rows, summaries)


def _row(run: Run, reference: Run, checks: Sequence[CheckName], read: Reader) -> Row:
    """Judge and mark every cell of one run."""
    record = run.record
    no_mutant = record.mutant.kind == "baseline"
    cells = {}
    for name in checks:
        cell = _cell(name, run, reference, read)
        planned = name in record.mutant.expected_killers
        cells[name] = MarkedCell(cell, planned, _mark(cell, planned, no_mutant), _stage(cell, name, record))
    return Row(run, no_mutant, cells)


def _cell(name: CheckName, run: Run, reference: Run, read: Reader) -> Cell:
    """Judge one check of one run against the reference; an invalid run gives invalid cells only."""
    problem = run.record.problem()
    if problem is not None:
        return Cell("invalid", f"run {problem}")
    result = run.record.checks.get(name)
    if result is None:
        return Cell("not_applicable", "the run made no such check")
    reference_result = reference.record.checks.get(name)
    if reference_result is None:
        return Cell("invalid", "the reference made no such check")
    return judge(read(name, reference_result, reference.folder), read(name, result, run.folder))


def _mark(cell: Cell, planned: bool, no_mutant: bool) -> str:
    """Name a kill or a survival by what was expected of the cell."""
    if no_mutant:
        return "false alarm" if cell.verdict == "killed" else ""
    if cell.verdict == "killed":
        return "expected kill" if planned else "unexpected kill"
    if cell.verdict == "survived":
        return "blind spot" if planned else "not aimed at"
    return ""


def _stage(cell: Cell, name: CheckName, record: MutantRunRecord) -> str:
    """Say whether a kill was made at the stage the mutant makes faulty or downstream of it."""
    component = record.mutant.component
    if cell.verdict != "killed" or component is None:
        return ""
    faulty, measured = _STAGE_ORDER.index(_COMPONENT_STAGE[component]), _STAGE_ORDER.index(CHECK_STAGE[name])
    if faulty == measured:
        return "at the faulty stage"
    return "inherited" if faulty < measured else "before the faulty stage"


def _summary(name: CheckName, rows: Sequence[Row]) -> CheckSummary:
    """Sum up one check: expected kills made, false alarms, mutants only it killed."""
    mutant_rows = [row for row in rows if not row.no_mutant]
    expected = [row for row in mutant_rows if row.cells[name].planned]
    only_this = tuple(
        row.run.record.mutant_id
        for row in mutant_rows
        if row.cells[name].cell.verdict == "killed"
        and all(cell.cell.verdict != "killed" for other, cell in row.cells.items() if other != name)
    )
    no_mutant_rows = [row for row in rows if row.no_mutant]
    return CheckSummary(
        expected_killed=sum(row.cells[name].cell.verdict == "killed" for row in expected),
        expected=len(expected),
        false_alarms=sum(row.cells[name].cell.verdict == "killed" for row in no_mutant_rows),
        no_mutant_runs=len(no_mutant_rows),
        only_this=only_this,
        cost=CHECK_COST[name],
    )


def read_measured(name: CheckName, result: CheckResult, folder: Path) -> Measured:
    """Read what one check gave in one run from the records in the run's folder."""
    if result.status == "refused":
        return Measured(refusal=result.refusal or "refused")
    if result.tests is not None:
        return Measured(tests=result.tests)
    if result.result_file is None:
        raise MatrixRefused(f"{folder.name}: {name} ran but left no result file")
    content = (folder / result.result_file).read_bytes()
    if name == "retrieval_gold":
        run = RunRecord.model_validate_json(content)
        comparison = load_declaration(Path(run.declaration_file)).comparison
        if comparison is None:
            raise MatrixRefused(f"{run.declaration_file} declares no [comparison] min_meaningful_change")
        found = {case.case_id: _found(case.passes["primary"]) for case in run.cases}
        # The signal's name says what is counted, so a cell reads without the code: "+3, -5" are questions that
        # began and stopped finding their section.
        signals = {"questions finding their section": found}
        return Measured(signals=signals, min_meaningful=comparison.min_meaningful_change)
    if name == "paraphrase":
        pairs = ParaphraseRunRecord.model_validate_json(content)
        rules = load_paraphrase_set(Path(pairs.pair_file)).rules
        lost: dict[str, bool | None] = {}
        stable: dict[str, bool | None] = {}
        for pair in pairs.pairs:
            primary = pair.passes["primary"]
            measured = primary.outcome.found != "failed" and not _weaker(primary.question, primary.rewording)
            overlap = primary.outcome.overlap
            lost[pair.case_id] = primary.outcome.found == "lost" if measured else None
            stable[pair.case_id] = (overlap is not None and overlap >= rules.min_overlap) if measured else None
        signals = {"pairs whose rewording loses the section": lost, "pairs with a stable top k": stable}
        return Measured(signals=signals, min_meaningful=rules.min_meaningful_change)
    raise MatrixRefused(f"{folder.name}: no reader for the check {name}")


def _found(result: PassResult) -> bool | None:
    """Whether a search found the answer section; None when it gave no result or was answered on a weaker path,
    since then it did not measure the configured search."""
    if result.score.verdict == "failed" or result.degraded_reason is not None:
        return None
    return result.score.verdict == "hit"


def _weaker(*results: PassResult) -> bool:
    """Whether any of the searches was answered on a weaker path."""
    return any(result.degraded_reason is not None for result in results)


def render(matrix: Matrix) -> str:
    """Write the matrix as plain text: one block per run, then one line per check."""
    reference = matrix.reference.record
    lines = [
        f"kill matrix of experiment {matrix.experiment_id[:8]}  course {reference.course_id}  "
        f"catalog {reference.catalog_file} ({reference.catalog_sha256[:12]})",
        f"reference: {reference.mutant_id} {reference.run_id[:8]}  app {_app_commit(reference)}",
        "",
    ]
    for row in matrix.rows:
        record = row.run.record
        title = "no mutant (false alarms)" if row.no_mutant else f"{record.mutant.component}"
        lines.append(f"{record.mutant_id} [{title}] {record.run_id[:8]}")
        for name in matrix.checks:
            marked = row.cells[name]
            # Every cell says whether a kill was expected, whatever its verdict: an underpowered or invalid cell
            # that was expected to kill is a cell to repeat or look at, not one to skip.
            expected = "-" if row.no_mutant else ("yes" if marked.planned else "no")
            notes = ", ".join(note for note in (marked.mark, marked.stage) if note)
            lines.append(f"  {name:<17} expected {expected:<4} {marked.cell.verdict:<14} {notes:<38} {marked.cell.why}")
    lines.append("")
    for name, summary in matrix.summaries.items():
        lines.append(
            f"{name:<17} expected kills {summary.expected_killed} of {summary.expected}; "
            f"false alarms {summary.false_alarms} of {summary.no_mutant_runs}; "
            f"only this check: {', '.join(summary.only_this) or '-'}; model calls: {summary.cost}"
        )
    return "\n".join(lines)


def _app_commit(record: MutantRunRecord) -> str:
    """The app commit the experiment ran on, cut to 12 characters."""
    commit = record.app_settings.get("git_commit")
    return str(commit)[:12] if commit is not None else "unknown"


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
