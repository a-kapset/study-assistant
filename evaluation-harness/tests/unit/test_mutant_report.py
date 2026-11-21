"""Unit tests of the experiment report's blocks: matrices and asks made up by hand, never the records of a real run
or the stack."""

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from evalharness.evaluation.answer_record import AnswerCaseRecord
from evalharness.failures import FailureLabel
from evalharness.mutation.catalog import CheckName, Component, Mutant
from evalharness.mutation.matrix import CheckSummary, MarkedCell, Matrix, Row, Run
from evalharness.mutation.record import CheckResult, MutantRunRecord, Proof
from evalharness.mutation.report import Experiment, cost, families, grid_block, label_rows, label_scores
from evalharness.mutation.verdicts import Cell, CellVerdict
from evalharness.oracles.answer import AnswerCaseScore
from evalharness.proxy.faults import FaultKind
from evalharness.proxy.ledger import Source

CHECKS: tuple[CheckName, ...] = ("retrieval_gold", "paraphrase", "answer_run")
NO_MUTANT = Mutant(kind="baseline", expected_symptom="evalharness")


def _mutant(component: Component, killers: tuple[CheckName, ...] = ()) -> Mutant:
    """Return a configuration mutant of ``component`` that expects ``killers``."""
    return Mutant(
        kind="configuration",
        component=component,
        settings={"APP_SEARCH__MIN_SCORE": "0.9"},
        expected_symptom="evalharness",
        expected_killers=killers,
        origin="invented",
    )


def _provider_fault(fault: FaultKind, label: FailureLabel) -> Mutant:
    """Return a provider fault that expects the answer run and the failure labels to notice it with ``label``."""
    return Mutant(
        kind="provider_fault",
        component="provider",
        fault=fault,
        expected_symptom="evalharness",
        expected_killers=("answer_run", "failure_labels"),
        expected_label=label,
        origin="invented",
    )


def _run(mutant_id: str, mutant: Mutant) -> Run:
    """Return one valid run of ``mutant``; the folder is never read by the blocks under test."""
    record = MutantRunRecord(
        run_id=f"evalharness_{mutant_id}",
        experiment_id="evalharness_experiment",
        started_at=datetime(2026, 11, 21, tzinfo=UTC),
        harness_commit="commit",
        harness_dirty=False,
        catalog_file="catalog.toml",
        catalog_sha256="evalharness_sha",
        course_id="evalharness_course",
        mutant_id=mutant_id,
        mutant=mutant,
        app_settings={"git_commit": "app"},
        applied=Proof(changed={}, problem=None),
        checks={name: CheckResult(status="ran", result_file=f"{name}.json") for name in CHECKS},
        restored=Proof(changed={}, problem=None),
    )
    return Run(record, Path(f"evalharness_{mutant_id}"))


def _row(mutant_id: str, mutant: Mutant, verdicts: Sequence[CellVerdict]) -> Row:
    """Return a row of the matrix whose cells have ``verdicts`` in the order of CHECKS; a cell is planned when the
    mutant names its check among its expected killers. Marks are left empty: the blocks must not depend on them."""
    cells = {
        name: MarkedCell(Cell(verdict, "evalharness"), name in mutant.expected_killers, "", "")
        for name, verdict in zip(CHECKS, verdicts, strict=True)
    }
    return Row(_run(mutant_id, mutant), mutant.kind == "baseline", cells)


def _matrix(rows: Sequence[Row]) -> Matrix:
    """Return a matrix of ``rows`` judged against a run without a mutant; the summaries are not read here."""
    summary = CheckSummary(0, 0, 0, 0, (), "none")
    return Matrix(
        "evalharness_experiment", _run("baseline", NO_MUTANT), CHECKS, tuple(rows), dict.fromkeys(CHECKS, summary)
    )


def _ask(
    case_id: str,
    calls: Sequence[Source] = ("upstream",),
    label: FailureLabel | None = None,
    cost_usd: float | None = 0.0,
) -> AnswerCaseRecord:
    """Build the record of one ask of ``case_id`` with the sources of its model calls, its label and its cost."""
    return AnswerCaseRecord(
        case_id=case_id,
        repetition=1,
        gold_section="4.2",
        search_request_id=f"evalharness_s_{case_id}",
        ask_request_id=f"evalharness_a_{case_id}",
        ask_error=None,
        failure_label=label,
        abstain_reason=None,
        cited_sections=(),
        invalid_citations=0,
        search_sections=("4.2",),
        model_calls=tuple(calls),
        cost_usd=cost_usd,
        score=AnswerCaseScore(verdict="failed" if label else "cited_gold", gold_retrieved=True),
    )


@pytest.mark.covers("US-019", "AC-10")
def test_the_grid_says_a_kill_was_expected_in_a_cell_whatever_its_verdict() -> None:
    """One search mutant whose expected killers are the retrieval and the answer checks: the retrieval cell is
    underpowered, the answer cell survived, the paraphrase cell (not expected) survived. Its line in the grid must
    read, in the matrix's column order: "underpowered (expected kill)", "survived (blind spot)", "survived".

    A pass says that a reader of the report sees every cell in which a kill was expected and did not come, also one
    with too few cases to decide. A failure points to grid_block in report.py. Without it an underpowered cell that
    was meant to kill looks like any cell of a check that was never aimed at the mutant, and nobody repeats it.
    """
    mutant = _mutant("search", ("retrieval_gold", "answer_run"))
    row = _row("low_score", mutant, ("underpowered", "survived", "survived"))
    row = Row(
        row.run, row.no_mutant, {**row.cells, "answer_run": MarkedCell(Cell("survived", ""), True, "blind spot", "")}
    )
    experiment = Experiment((), _matrix([row]), (), None)

    line = next(line for line in grid_block(experiment).splitlines() if "`low_score`" in line)

    cells = [cell.strip() for cell in line.strip("|").split("|")][2:]
    expected = ["underpowered (expected kill)", "survived", "survived (blind spot)"]
    assert cells == expected, f"the grid line does not show what each cell was expected to do: {cells}"


@pytest.mark.covers("US-019", "AC-10")
def test_families_hold_the_expected_cells_that_survived_or_were_underpowered() -> None:
    """Two ingestion mutants expected by the retrieval check: one survived it, one was underpowered on it; a third
    ingestion mutant survived it too; a search mutant was killed by it as expected, one was invalid on it, and the run
    without a mutant survived. The families must be: retrieval × ingestion × survived (the two survivors) and
    retrieval × ingestion × underpowered (one); a survival of a check nobody expected is no family.

    A pass says that the report's list of what no check noticed holds exactly the misses the catalog did not plan
    for. A failure points to families in report.py: an underpowered cell dropped (it is a miss to look at), a kill or
    an invalid run counted as a miss, or a check's survival on a mutant it was never aimed at counted as one.
    """
    retrieval: tuple[CheckName, ...] = ("retrieval_gold",)
    rows = [
        _row("short", _mutant("ingestion", retrieval), ("survived", "killed", "survived")),
        _row("long", _mutant("ingestion", retrieval), ("underpowered", "survived", "survived")),
        _row("tiny", _mutant("ingestion", retrieval), ("survived", "survived", "survived")),
        _row("low_score", _mutant("search", retrieval), ("killed", "survived", "survived")),
        _row("broken", _mutant("search", retrieval), ("invalid", "invalid", "invalid")),
        _row("baseline", NO_MUTANT, ("survived", "survived", "survived")),
    ]

    found = {(f.check, f.component, f.verdict): f.mutants for f in families(_matrix(rows))}

    expected = {
        ("retrieval_gold", "ingestion", "survived"): ("short", "tiny"),
        ("retrieval_gold", "ingestion", "underpowered"): ("long",),
    }
    assert found == expected, f"the families are not the expected cells that were not killed: {found}"


@pytest.mark.covers("US-019", "AC-10")
def test_label_precision_counts_every_run_and_recall_only_the_asks_the_fault_reached() -> None:
    """A provider fault expecting `rate_limited`: three asks got the fault and the label, one ask was declined before
    any model call. Another run, without that fault, has one ask wrongly labelled `rate_limited`. The score of the
    label must be 3 right, 4 given in all runs, 3 reached: precision 3 of 4, recall 3 of 3.

    A pass says that precision counts the labels given where the fault was not planted, and that recall leaves out
    the asks the fault never reached. A failure points to label_rows or label_scores in report.py. Without the first
    a label given everywhere would read as perfect; without the second the asks declined before the model (nine
    questions in our runs) would read as missed labels.
    """
    fault = [_ask("q1", ("fault",), "rate_limited"), _ask("q2", ("fault",), "rate_limited")]
    fault += [_ask("q3", ("fault",), "rate_limited"), _ask("q4", ())]
    other = [_ask("q1", ("upstream",), "rate_limited"), _ask("q2")]
    rows = label_rows(
        [
            ("provider_rate_limited", _provider_fault("rate_limited", "rate_limited"), fault),
            ("low_score", _mutant("search"), other),
        ]
    )

    scores = label_scores(rows)

    assert scores == {"rate_limited": (3, 4, 3)}, f"(right, given, reached) of the label are wrong: {scores}"


@pytest.mark.covers("US-019", "AC-10")
def test_only_asks_that_went_to_the_provider_count_as_paid_and_only_they_can_lack_a_price() -> None:
    """Five asks: one answered by the provider for USD 0.01, one answered from a recording at a listed price of
    USD 0.02, one that met a planted fault (no price), one declined before any model call, and one that went to the
    provider and has no price. The cost must be USD 0.01 paid, USD 0.02 from recordings, one paid ask without a price.

    A pass says that the report's cost is what the provider was paid, and that an unknown price is shown as unknown.
    A failure points to cost in report.py. The first version of the report summed every ask and gave the free run
    answered from recordings a cost of USD 0.90, and counted 412 planted faults as asks of unknown cost.
    """
    cases = [
        _ask("q1", ("upstream",), cost_usd=0.01),
        _ask("q2", ("cassette",), cost_usd=0.02),
        _ask("q3", ("fault",), "rate_limited", cost_usd=None),
        _ask("q4", ()),
        _ask("q5", ("upstream", "upstream"), "no_response", cost_usd=None),
    ]

    spent = cost(cases)

    found = (round(spent.paid_usd, 6), round(spent.replayed_usd, 6), spent.unpriced)
    assert found == (0.01, 0.02, 1), f"(paid, from recordings, paid asks without a price) are wrong: {found}"
