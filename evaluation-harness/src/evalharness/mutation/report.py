"""The parts of an experiment report that come from the mutant run records, as Markdown.

The report itself is a Markdown file under reports/: its question, decision and the causes of what the checks missed
are written by hand, and every number sits in a block that cog fills by calling the functions here with the ids of the
experiment's runs. Run from evaluation-harness/, where the records are:

    poetry run cog -r -U reports/<report>.md         # fill the blocks again from the records
    poetry run cog --check -U reports/<report>.md    # fail when the records no longer give the same blocks

The records hold course material and stay on the machine that made them, so only that machine can rebuild a report.
"""

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import get_args

from evalharness.evaluation.answer_record import AnswerCaseRecord, AnswerRunRecord
from evalharness.failures import FailureLabel
from evalharness.mutation.catalog import CheckName, Component, Mutant
from evalharness.mutation.matrix import (
    CHECK_STAGE,
    RUNS_DIR,
    AttributionTable,
    Matrix,
    Run,
    build_attribution,
    build_matrix,
    has_stages,
    load_experiment,
    read_measured,
    render_attribution,
)
from evalharness.proxy.faults import FaultKind


@dataclass(frozen=True)
class Experiment:
    """A judged experiment with its matrix, and the experiment its traces were read from (None when there is none)."""

    runs: tuple[Run, ...]
    matrix: Matrix
    attribution_runs: tuple[Run, ...]
    attribution: AttributionTable | None


@dataclass(frozen=True)
class Family:
    """Mutants a check was expected to notice and did not, of one faulty part and one verdict."""

    check: CheckName
    component: Component
    # "survived": the check saw no change; "underpowered": too few cases to say either way.
    verdict: str
    mutants: tuple[str, ...]


@dataclass(frozen=True)
class LabelRow:
    """How the asks of one run were labelled, against the label its mutant expects (None: it expects none)."""

    mutant_id: str
    expected: FailureLabel | None
    # Asks the planted fault reached: for a provider fault the asks whose model calls got it, otherwise every ask.
    reached: int
    labels: Counter[FailureLabel | None]


@dataclass(frozen=True)
class Cost:
    """What the model calls of an experiment's answer runs cost by their records.

    A record prices every answered call at the listed price, also one the proxy answered from a recording; only the
    asks whose calls went to the provider were paid for. A planted fault is answered by the proxy and costs nothing.
    """

    # The asks whose calls went to the provider.
    paid_usd: float
    # The asks answered from recordings, at the listed price: what the run would have cost without them.
    replayed_usd: float
    asks: int
    # Asks with a call to the provider and no price: what they cost is unknown, not zero.
    unpriced: int
    calls: Counter[str]


@dataclass(frozen=True)
class NotCovered:
    """What a catalog does not plant: fault kinds of the proxy and faulty parts with no mutant."""

    fault_kinds: tuple[FaultKind, ...]
    components: tuple[Component, ...]


@cache
def load(experiment_id: str, attribution_id: str | None = None, runs_dir: Path = RUNS_DIR) -> Experiment:
    """Read the runs of a judged experiment and, when given, of the experiment its traces are read from; built once
    per report, since every block of one report asks for the same experiment."""
    runs = tuple(load_experiment(runs_dir, experiment_id))
    matrix = build_matrix(runs, read_measured)
    attribution_runs = runs if attribution_id is None else tuple(load_experiment(runs_dir, attribution_id))
    table = build_attribution(attribution_runs, matrix) if has_stages(attribution_runs) else None
    return Experiment(runs, matrix, attribution_runs, table)


def families(matrix: Matrix) -> list[Family]:
    """Group the cells in which a kill was expected and did not come, by check, faulty part and verdict; a cell that
    was invalid or not applicable says nothing about the check and is left out."""
    grouped: dict[tuple[CheckName, Component, str], list[str]] = {}
    for row in matrix.rows:
        component = row.run.record.mutant.component
        if row.no_mutant or component is None:
            continue
        for name, marked in row.cells.items():
            if marked.planned and marked.cell.verdict in ("survived", "underpowered"):
                grouped.setdefault((name, component, marked.cell.verdict), []).append(row.run.record.mutant_id)
    return [Family(name, component, verdict, tuple(ids)) for (name, component, verdict), ids in grouped.items()]


def label_rows(runs: Iterable[tuple[str, Mutant, Sequence[AnswerCaseRecord]]]) -> list[LabelRow]:
    """Count the labels of every run's asks, and how many asks its planted fault reached."""
    rows = []
    for mutant_id, mutant, cases in runs:
        planted = mutant.kind == "provider_fault"
        reached = sum("fault" in case.model_calls for case in cases) if planted else len(cases)
        rows.append(LabelRow(mutant_id, mutant.expected_label, reached, Counter(case.failure_label for case in cases)))
    return rows


def label_scores(rows: Sequence[LabelRow]) -> dict[FailureLabel, tuple[int, int, int]]:
    """Per expected label: asks labelled with it in runs that expect it, all asks labelled with it in any run (the
    denominator of precision), and asks the fault reached in runs that expect it (the denominator of recall)."""
    scores = {}
    for label in dict.fromkeys(row.expected for row in rows if row.expected is not None):
        right = sum(row.labels[label] for row in rows if row.expected == label)
        given = sum(row.labels[label] for row in rows)
        reached = sum(row.reached for row in rows if row.expected == label)
        scores[label] = (right, given, reached)
    return scores


def cost(cases: Iterable[AnswerCaseRecord]) -> Cost:
    """Sum the cost of the asks that went to the provider apart from the asks answered from recordings, and count the
    sources of their model calls. An ask with a call to the provider and no price is counted apart, so an unknown
    price never reads as a free call; an ask with a call to the provider is counted as paid as a whole."""
    paid, replayed, asks, unpriced, calls = 0.0, 0.0, 0, 0, Counter[str]()
    for case in cases:
        asks += 1
        calls.update(case.model_calls)
        to_provider = "upstream" in case.model_calls
        if case.cost_usd is None:
            unpriced += to_provider
        elif to_provider:
            paid += case.cost_usd
        else:
            replayed += case.cost_usd
    return Cost(paid, replayed, asks, unpriced, calls)


def not_covered(mutants: Iterable[Mutant]) -> NotCovered:
    """Return the fault kinds of the proxy and the faulty parts that no mutant of the catalog plants."""
    planted = list(mutants)
    kinds = tuple(kind for kind in get_args(FaultKind.__value__) if all(m.fault != kind for m in planted))
    parts = tuple(part for part in get_args(Component.__value__) if all(m.component != part for m in planted))
    return NotCovered(kinds, parts)


def answer_cases(run: Run) -> tuple[AnswerCaseRecord, ...]:
    """Return the asks of a run's answer record; none when the run made no answer run or it was refused."""
    result = run.record.checks.get("answer_run")
    if result is None or result.status != "ran" or result.result_file is None:
        return ()
    return AnswerRunRecord.model_validate_json((run.folder / result.result_file).read_bytes()).cases


# The blocks of the report. Each returns Markdown with an empty line before and after it; cog writes it unchanged.


def configuration_block(experiment: Experiment) -> str:
    """The experiment's ids, the app and harness versions, the catalog and the models, and how many runs were valid."""
    reference = experiment.matrix.reference.record
    settings = reference.app_settings
    embedding = settings.get("embedding")
    llm = settings.get("llm")
    runs = experiment.runs
    valid = sum(run.record.problem() is None for run in runs)
    rows = [
        ("Judged experiment", f"`{experiment.matrix.experiment_id}`"),
        ("Course", f"`{reference.course_id}`"),
        ("Catalog", f"`{reference.catalog_file}` (SHA-256 `{reference.catalog_sha256[:12]}`)"),
        ("App commit", f"`{str(settings.get('git_commit'))[:12]}`"),
        (
            "Harness commit",
            f"`{reference.harness_commit[:12]}`" + (" with changes not committed" * reference.harness_dirty),
        ),
        ("Embedding model", f"`{embedding.get('model') if isinstance(embedding, dict) else None}`"),
        ("Answer model", f"`{llm.get('answer') if isinstance(llm, dict) else None}`"),
        ("Runs", f"{len(runs)}, valid {valid}; first {_time(runs[0])}, last {_time(runs[-1])}"),
    ]
    if experiment.attribution_runs is not experiment.runs:
        first = experiment.attribution_runs[0]
        valid = sum(run.record.problem() is None for run in experiment.attribution_runs)
        rows.append(
            (
                "Traces read from",
                f"`{first.record.experiment_id}`: {len(experiment.attribution_runs)} runs, valid {valid}",
            )
        )
    return _table(("Item", "Value"), rows)


def sample_block(experiment: Experiment) -> str:
    """What each check counts in the reference run, and over how many cases."""
    reference = experiment.matrix.reference
    rows: list[tuple[str, str, str]] = []
    for name in experiment.matrix.checks:
        measured = read_measured(name, reference.record.checks[name], reference.folder, reference.record.mutant)
        if measured.tests is not None:
            rows.append((f"`{name}`", CHECK_STAGE[name], f"{len(measured.tests)} property tests pass or fail"))
            continue
        for signal, values in (measured.signals or {}).items():
            rows.append((f"`{name}`", CHECK_STAGE[name], f"{signal}: {len(values)}"))
    cases = answer_cases(reference)
    if cases:
        asks = Counter(case.case_id for case in cases)
        rows.append(("", "", f"asks per question in the answer run: {max(asks.values())}"))
    return _table(("Check", "Stage it measures", "Counted over"), rows)


def grid_block(experiment: Experiment) -> str:
    """One row per run and one column per check: the verdict, and what was expected of the cell when it matters."""
    matrix = experiment.matrix
    rows = []
    for row in matrix.rows:
        record = row.run.record
        part = "no mutant" if row.no_mutant else str(record.mutant.component)
        cells = []
        for name in matrix.checks:
            marked = row.cells[name]
            # "not aimed at" is the usual case and is left out so that the cells that matter stand out.
            note = "" if marked.mark in ("", "not aimed at") else f" ({marked.mark})"
            if marked.planned and marked.cell.verdict == "underpowered":
                note = " (expected kill)"
            cells.append(f"{marked.cell.verdict}{note}")
        rows.append((f"`{record.mutant_id}`", part, *cells))
    return _table(("Mutant", "Faulty part", *(f"`{name}`" for name in matrix.checks)), rows)


def summary_block(experiment: Experiment) -> str:
    """Per check: expected kills, false alarms, kills no other check made, blind spots, and whether it pays for model
    calls."""
    matrix = experiment.matrix
    missed: dict[CheckName, list[str]] = {name: [] for name in matrix.checks}
    for family in families(matrix):
        missed[family.check].extend(f"`{mutant}` ({family.verdict})" for mutant in family.mutants)
    rows = []
    for name, summary in matrix.summaries.items():
        labels = f"; labels named {summary.named_right} of {summary.named_expected}" if summary.named_expected else ""
        rows.append(
            (
                f"`{name}`",
                f"{summary.expected_killed} of {summary.expected}{labels}",
                f"{summary.false_alarms} of {summary.no_mutant_runs}",
                ", ".join(f"`{mutant}`" for mutant in summary.only_this) or "-",
                ", ".join(missed[name]) or "-",
                summary.cost,
            )
        )
    header = ("Check", "Expected kills", "False alarms", "Only this check", "Expected, not killed", "Model calls")
    return _table(header, rows)


def families_block(experiment: Experiment) -> str:
    """The cells in which a kill was expected and did not come, grouped by check, faulty part and verdict."""
    rows = [
        (f"`{family.check}`", family.component, family.verdict, str(len(family.mutants)), _ids(family.mutants))
        for family in families(experiment.matrix)
    ]
    return _table(("Check", "Faulty part", "Verdict", "n", "Mutants"), rows)


def labels_block(experiment: Experiment) -> str:
    """How the asks of every run with a failure label to expect were labelled, the false labels of the other runs,
    and precision and recall per expected label over asks."""
    # Every valid run, the reference too: a label given where no fault was planted is a false label.
    runs = [run for run in experiment.runs if run.record.problem() is None]
    rows = label_rows((run.record.mutant_id, run.record.mutant, answer_cases(run)) for run in runs)
    no_record = [run.record.mutant_id for run in runs if run.record.mutant.expected_label and not answer_cases(run)]
    table = []
    for row in rows:
        given = ", ".join(f"{label or 'no label'} {n}" for label, n in sorted(row.labels.items(), key=_label_key))
        if row.expected is not None or any(label is not None for label in row.labels):
            table.append((f"`{row.mutant_id}`", row.expected or "-", str(row.reached), given))
    lines = [_table(("Run", "Label expected", "Asks the fault reached", "Labels the asks got"), table)]
    scores = [
        (label, f"{right} of {given}" + _share(right, given), f"{right} of {reached}" + _share(right, reached))
        for label, (right, given, reached) in label_scores(rows).items()
    ]
    lines.append(_table(("Label", "Precision (asks)", "Recall (asks)"), scores))
    if no_record:
        lines.append(_block(f"No answer record, so no labels: {_ids(no_record)}."))
    return "\n" + "".join(line.strip("\n") + "\n\n" for line in lines)


def attribution_block(experiment: Experiment) -> str:
    """The stage the request traces name for each mutant, as the matrix command prints it."""
    if experiment.attribution is None:
        return _block("No traces in the records.")
    return _block(f"```text\n{render_attribution(experiment.attribution)}\n```")


def cost_block(experiment: Experiment) -> str:
    """What the model calls of the judged experiment and of the traces' experiment cost, by their records."""
    sources = [("judged", experiment.runs)]
    if experiment.attribution_runs is not experiment.runs:
        sources.append(("traces", experiment.attribution_runs))
    rows = []
    for name, runs in sources:
        spent = cost(case for run in runs for case in answer_cases(run))
        calls = ", ".join(f"{source} {n}" for source, n in sorted(spent.calls.items())) or "none"
        rows.append(
            (
                name,
                f"`{(runs[0].record.experiment_id or '')[:8]}`",
                f"{spent.paid_usd:.2f}",
                f"{spent.replayed_usd:.2f}",
                str(spent.asks),
                str(spent.unpriced),
                calls,
            )
        )
    header = (
        "Experiment",
        "Id",
        "Paid, USD",
        "From recordings at list price, USD",
        "Asks",
        "Paid asks without a price",
    )
    return _table((*header, "Model calls by source"), rows)


def not_covered_block(experiment: Experiment) -> str:
    """The proxy's fault kinds and the faulty parts that no mutant of the experiment plants."""
    gaps = not_covered(run.record.mutant for run in experiment.runs)
    kinds = ", ".join(f"`{kind}`" for kind in gaps.fault_kinds) or "none"
    parts = ", ".join(f"`{part}`" for part in gaps.components) or "none"
    return _block(f"- Fault kinds of the proxy never planted: {kinds}.\n- Faulty parts with no mutant: {parts}.")


def margins_block(experiment: Experiment) -> str:
    """Every cell that was killed or underpowered, with its run id and the change it measured against the threshold."""
    rows = []
    for row in experiment.matrix.rows:
        for name, marked in row.cells.items():
            if marked.cell.verdict in ("killed", "underpowered"):
                record = row.run.record
                rows.append(
                    (
                        f"`{record.mutant_id}`",
                        f"`{record.run_id[:8]}`",
                        f"`{name}`",
                        marked.cell.verdict,
                        marked.cell.why,
                    )
                )
    return _table(("Mutant", "Run", "Check", "Verdict", "What changed"), rows)


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Write a Markdown table; a bar inside a cell is escaped so it does not split the cell."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in rows]
    return _block("\n".join(lines))


def _block(text: str) -> str:
    """Put an empty line before and after a block, so that the comment lines of cog around it are not read as part
    of it (a table would take the next line as a row)."""
    return "\n" + text.strip("\n") + "\n\n"


def _ids(ids: Sequence[str]) -> str:
    """Write mutant ids as code, separated by commas."""
    return ", ".join(f"`{mutant_id}`" for mutant_id in ids)


def _share(part: int, whole: int) -> str:
    """Write a share after its count, or nothing when there is nothing to divide by."""
    return f" ({part / whole:.2f})" if whole else ""


def _label_key(item: tuple[FailureLabel | None, int]) -> tuple[int, str]:
    """Order labels by how many asks got them, most first, then by name."""
    label, n = item
    return -n, label or ""


def _time(run: Run) -> str:
    """When a run started, to the minute, in UTC."""
    return f"{run.record.started_at:%Y-%m-%d %H:%M} UTC"
