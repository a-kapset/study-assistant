"""Unit tests of the kill matrix: made-up mutant run records and a fake reader of their check results; never the
stack or files."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from evalharness.mutation.catalog import CheckName, Component, Mutant
from evalharness.mutation.matrix import Matrix, MatrixRefused, Reader, Run, build_attribution, build_matrix
from evalharness.mutation.record import CheckResult, MutantRunRecord, Proof
from evalharness.mutation.verdicts import Measured

CHECKS: tuple[CheckName, ...] = ("retrieval_gold", "paraphrase")
START = datetime(2026, 11, 19, tzinfo=UTC)
FOUND = {f"q{index}": index < 100 for index in range(146)}
LOST_19 = {f"q{index}": 19 <= index < 100 for index in range(146)}
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


def _run(order: int, mutant_id: str, mutant: Mutant, applied: str | None = None, **record: object) -> Run:
    """Return one run of the experiment, started ``order`` minutes after the first, with a result of every check."""
    fields: dict[str, object] = {
        "run_id": f"evalharness_{order}",
        "experiment_id": "evalharness_experiment",
        "started_at": START + timedelta(minutes=order),
        "harness_commit": "commit",
        "harness_dirty": False,
        "catalog_file": "catalog.toml",
        "catalog_sha256": "evalharness_sha",
        "course_id": "evalharness_course",
        "mutant_id": mutant_id,
        "mutant": mutant,
        "app_settings": {"git_commit": "app"},
        "applied": Proof(changed={}, problem=applied),
        "checks": {name: CheckResult(status="ran", result_file=f"{name}.json") for name in CHECKS},
        "restored": Proof(changed={}, problem=None),
    }
    fields.update(record)
    return Run(MutantRunRecord.model_validate(fields), Path(f"evalharness_{order}"))


def _reader(killed_by: dict[str, set[CheckName]]) -> Reader:
    """Return a reader under which each run in ``killed_by`` lost 19 of 146 questions on the named checks, and every
    other result equals the reference."""

    def read(name: CheckName, result: CheckResult, folder: Path, mutant: Mutant) -> Measured:
        """Give the made-up answers of one check in one run."""
        values = LOST_19 if name in killed_by.get(folder.name, set()) else FOUND
        return Measured(signals={"found": values}, min_meaningful=0.05)

    return read


def _matrix(runs: list[Run], killed_by: dict[str, set[CheckName]]) -> Matrix:
    """Build the matrix of ``runs`` with the made-up answers."""
    return build_matrix(runs, _reader(killed_by))


@pytest.mark.covers("US-019", "AC-6")
def test_every_cell_of_a_run_without_proof_is_invalid() -> None:
    """A mutant run whose mutant was never applied, while the reader would show a lost 19 questions for it. Every
    cell of that run must be invalid and carry no mark of a kill or of a blind spot.

    A pass says that a run the app never really ran is kept out of the verdicts. A failure points to the validity
    check of matrix.py. Without it a mutant that was not applied would read as a blind spot of every check that was
    expected to see it.
    """
    runs = [
        _run(0, "baseline", NO_MUTANT),
        _run(1, "low_score", _mutant("search", CHECKS), applied="not changed: ['search.min_score']"),
    ]

    row = _matrix(runs, {"evalharness_1": set(CHECKS)}).rows[0]

    for name, marked in row.cells.items():
        assert marked.cell.verdict == "invalid" and marked.mark == "", f"{name}: {marked}"


@pytest.mark.covers("US-019", "AC-6")
@pytest.mark.parametrize(
    "runs",
    [
        pytest.param(
            [_run(0, "baseline", NO_MUTANT), _run(1, "low", _mutant("search"), catalog_sha256="other")],
            id="another catalog version",
        ),
        pytest.param(
            [_run(0, "baseline", NO_MUTANT), _run(1, "low", _mutant("search"), harness_commit="other")],
            id="another harness commit",
        ),
        pytest.param(
            [_run(0, "baseline", NO_MUTANT), _run(1, "low", _mutant("search"), app_settings={"git_commit": "x"})],
            id="another first state of the app",
        ),
        pytest.param([_run(0, "low", _mutant("search"))], id="no run without a mutant"),
        pytest.param(
            [_run(0, "baseline", NO_MUTANT, applied="changed but not declared"), _run(1, "low", _mutant("search"))],
            id="a reference that is not valid",
        ),
    ],
)
def test_runs_that_are_not_one_experiment_with_a_valid_reference_make_no_matrix(runs: list[Run]) -> None:
    """Runs that differ in catalog version, harness commit or the app's first settings, or a set without a valid
    run without a mutant, must be refused.

    A pass says that every cell of a matrix is judged against a reference of the same experiment. A failure points
    to build_matrix in matrix.py. Without it a cell could compare a mutant with a baseline of another day, another
    catalog or another app, and show that difference as a kill.
    """
    with pytest.raises(MatrixRefused):
        _matrix(runs, {})


@pytest.mark.covers("US-019", "AC-6")
def test_cells_are_marked_by_what_the_catalog_expected_and_by_the_stage_of_the_kill() -> None:
    """A search mutant expected by the retrieval check and killed by both checks; an ingestion mutant expected by
    the retrieval check and killed by none. The marks must be: expected kill at the faulty stage, unexpected kill,
    blind spot, not aimed at; and a kill of a search check on an ingestion mutant must read as inherited.

    A pass says that the matrix reads the expectations written before the run and the stage of each check. A failure
    points to the marks of matrix.py. Without them a blind spot would look like "not aimed at" and nobody would see
    that a check missed what it was meant to catch.
    """
    runs = [
        _run(0, "baseline", NO_MUTANT),
        _run(1, "low_score", _mutant("search", ("retrieval_gold",))),
        _run(2, "short", _mutant("ingestion", ("retrieval_gold",))),
        _run(3, "long", _mutant("ingestion")),
    ]

    rows = _matrix(runs, {"evalharness_1": set(CHECKS), "evalharness_3": {"paraphrase"}}).rows

    seen = {
        (row.run.record.mutant_id, name): (cell.mark, cell.stage) for row in rows for name, cell in row.cells.items()
    }
    assert seen[("low_score", "retrieval_gold")] == ("expected kill", "at the faulty stage"), seen
    assert seen[("low_score", "paraphrase")] == ("unexpected kill", "at the faulty stage"), seen
    assert seen[("short", "retrieval_gold")] == ("blind spot", ""), seen
    assert seen[("short", "paraphrase")] == ("not aimed at", ""), seen
    assert seen[("long", "paraphrase")] == ("unexpected kill", "inherited"), seen


@pytest.mark.covers("US-019", "AC-7")
def test_the_summary_counts_expected_kills_false_alarms_and_kills_no_other_check_made() -> None:
    """The paraphrase check kills both mutants and the second run without a mutant; the retrieval check kills only
    the first mutant, which both were expected to notice. The summary must give the retrieval check 1 of 1 expected,
    no false alarm and no kill of its own; the paraphrase check 1 of 1 expected, 1 false alarm of 1, and the second
    mutant as a kill only it made.

    A pass says that a check is summed up by what it was meant to catch, what it raised without a mutant and what
    only it catches. A failure points to the summary of matrix.py. Without the false alarms a check that fires on
    everything would look best.
    """
    runs = [
        _run(0, "baseline", NO_MUTANT),
        _run(1, "low_score", _mutant("search", CHECKS)),
        _run(2, "one_hit", _mutant("search")),
        _run(3, "baseline", NO_MUTANT),
    ]
    killed: dict[str, set[CheckName]] = {
        "evalharness_1": set(CHECKS),
        "evalharness_2": {"paraphrase"},
        "evalharness_3": {"paraphrase"},
    }

    summaries = _matrix(runs, killed).summaries

    retrieval, paraphrase = summaries["retrieval_gold"], summaries["paraphrase"]
    assert (retrieval.expected_killed, retrieval.expected, retrieval.false_alarms, retrieval.only_this) == (
        1,
        1,
        0,
        (),
    ), f"retrieval check: {retrieval}"
    assert (paraphrase.expected_killed, paraphrase.false_alarms, paraphrase.no_mutant_runs, paraphrase.only_this) == (
        1,
        1,
        1,
        ("one_hit",),
    ), f"paraphrase check: {paraphrase}"


@pytest.mark.covers("US-019", "AC-9")
@pytest.mark.parametrize(
    ("changed", "field"),
    [
        pytest.param({"catalog_sha256": "other"}, "catalog", id="another-catalog-version"),
        pytest.param({"app_settings": {"git_commit": "other"}}, "app commit", id="another-app-commit"),
    ],
)
def test_the_traces_of_another_catalog_or_app_are_not_put_next_to_the_matrix(
    changed: dict[str, Any], field: str
) -> None:
    """What it checks: the stages named by an attribution experiment are joined to a matrix only when that
    experiment ran the same catalog version on the same app commit; otherwise the join is refused, naming the field.

    What it tells: a pass means each attribution line belongs to the mutant of its matrix row. A failure means a
    mutant id could carry the traces of another mutant with the same id.

    Why it is worth having: the attribution runs as its own experiment, joined by mutant id, and the catalog was
    changed on 2026-11-19 with new mutants and expectations; the same id can name another setting after such a
    change.
    """
    matrix = _matrix([_run(0, "baseline", NO_MUTANT), _run(1, "low", _mutant("search"))], {})
    attribution_runs = [_run(0, "baseline", NO_MUTANT, **changed), _run(1, "low", _mutant("search"), **changed)]

    with pytest.raises(MatrixRefused) as refused:
        build_attribution(attribution_runs, matrix)

    assert field in str(refused.value), f"refused for another reason: {refused.value}"
