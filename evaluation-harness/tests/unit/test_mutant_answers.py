"""Unit tests of the answer side of the mutant experiment: how an answer run's record becomes cells of the kill
matrix, how a provider fault is planted in the asks, and how the catalog describes such mutants. Made-up records and
files written by the tests, never the stack."""

from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import pytest

from evalharness.evaluation.answer_record import AnswerCaseRecord, AnswerRunRecord
from evalharness.evaluation.answer_run import ask_suffix
from evalharness.failures import FailureLabel
from evalharness.mutation.catalog import CatalogError, Mutant, load_catalog
from evalharness.mutation.matrix import answer_problem, answer_signals, label_note, not_declared_asks
from evalharness.mutation.oracles import failure_label_check
from evalharness.mutation.verdicts import Measured, judge, signal_cell
from evalharness.oracles.answer import AnswerCaseScore, AnswerVerdict
from evalharness.proxy.faults import FaultKind, fault_of
from evalharness.proxy.keys import is_fresh, repetition_of
from evalharness.proxy.ledger import Source

NO_MUTANT = Mutant(kind="baseline", expected_symptom="evalharness")
RATE_LIMITED = Mutant(
    kind="provider_fault",
    component="provider",
    fault="rate_limited",
    expected_symptom="evalharness",
    expected_killers=("answer_run", "failure_labels"),
    expected_label="rate_limited",
    origin="invented",
)


def _ask(
    case_id: str,
    repetition: int,
    verdict: AnswerVerdict,
    gold: str | None = "1.1",
    calls: tuple[Source, ...] = ("upstream",),
    label: FailureLabel | None = None,
) -> AnswerCaseRecord:
    """Build one ask of an answer run's record with the verdict, model calls and failure label given."""
    return AnswerCaseRecord(
        case_id=case_id,
        repetition=repetition,
        gold_section=gold,
        search_request_id="evalharness_search",
        ask_request_id=f"evalharness_{case_id}_{repetition}",
        ask_error=None,
        failure_label=label,
        abstain_reason=None,
        cited_sections=(),
        invalid_citations=0,
        search_sections=(),
        model_calls=calls,
        cost_usd=None,
        score=AnswerCaseScore(verdict=verdict, gold_retrieved=None),
    )


def _record(*asks: AnswerCaseRecord) -> AnswerRunRecord:
    """Build an answer run's record holding ``asks``."""
    return AnswerRunRecord(
        run_id="evalharness_run",
        started_at=datetime(2026, 11, 19, tzinfo=UTC),
        harness_commit="commit",
        harness_dirty=False,
        qa_file="qa.toml",
        qa_sha256="evalharness_sha",
        course_id="evalharness_course",
        corpus_snapshot="evalharness_snapshot",
        app_settings={},
        models_returned=(),
        repetitions=2,
        summaries=(),
        cases=asks,
    )


def _by_case(*asks: AnswerCaseRecord) -> dict[str, list[AnswerCaseRecord]]:
    """Group asks by their question, as the matrix reads them."""
    grouped: defaultdict[str, list[AnswerCaseRecord]] = defaultdict(list)
    for ask in asks:
        grouped[ask.case_id].append(ask)
    return dict(grouped)


@pytest.mark.covers("US-019", "AC-8")
@pytest.mark.parametrize("fault", [None, "rate_limited", "empty_output"])
@pytest.mark.parametrize("repetition", [1, 2, 12])
def test_every_mark_of_an_ask_is_read_back_by_the_proxy(repetition: int, fault: FaultKind | None) -> None:
    """Marks a request id the way an answer run of the experiment marks its asks (fresh, a repetition, maybe a
    fault) and reads it back the way the proxy does.

    What it checks: the proxy sees the call as fresh, as the right repetition and with the right fault.
    What it tells: a pass means the runner and the proxy agree on all three marks together. A failure points to
    the order of the marks in ask_suffix: the proxy reads the repetition and the fault only from the end.
    Why it is worth having: a fault read as no fault makes the provider mutants pay for real calls and survive; a
    repetition read wrongly mixes the two asks of a question in the ledger. Both stay silent.
    """
    request_id = "3f2a" + ask_suffix(repetition, fresh=True, fault=fault)

    assert is_fresh(request_id), f"{request_id!r} is not read as fresh"
    assert fault_of(request_id) == fault, f"{request_id!r} read as fault {fault_of(request_id)!r}, not {fault!r}"
    if fault is None:
        assert repetition_of(request_id) == repetition, f"{request_id!r} read as repetition {repetition_of(request_id)}"


@pytest.mark.covers("US-019", "AC-1")
@pytest.mark.parametrize(
    ("entry", "reason"),
    [
        pytest.param('fault = "rate_limited"', "names its fault", id="a provider fault with a setting"),
        pytest.param("", "names its fault", id="a provider fault without a fault"),
    ],
)
def test_a_provider_fault_must_name_its_fault_and_change_no_setting(tmp_path: Path, entry: str, reason: str) -> None:
    """Writes a catalog with a provider-fault mutant that also changes a setting, or that names no fault.

    What it checks: the catalog is refused with the reason.
    What it tells: a pass means every provider mutant plants exactly one fault and nothing else. A failure points to
    the validator of the mutant entry.
    Why it is worth having: a provider mutant without a fault would run as the app without a mutant and read as a
    blind spot of the answer checks; one with a setting too would make it unclear which change a check noticed.
    """
    settings = 'settings = { APP_SEARCH__MIN_SCORE = "0.5" }' if entry else ""
    path = tmp_path / "catalog.toml"
    path.write_text(
        f"""
course_id = "evalharness_course"
[checks]
answer_run = {{ qa = "qa.toml", repetitions = 2 }}
[mutants.baseline]
kind = "baseline"
expected_symptom = "evalharness"
[mutants.fault]
kind = "provider_fault"
component = "provider"
{entry}
{settings}
expected_symptom = "evalharness"
expected_killers = ["answer_run"]
origin = "invented"
""",
        encoding="utf-8",
    )

    with pytest.raises(CatalogError, match=reason):
        load_catalog(path)


@pytest.mark.covers("US-019", "AC-6")
def test_answer_signals_are_shares_of_the_asks_and_a_failed_ask_gives_no_value() -> None:
    """Reads four questions asked twice: one cites its gold section once of two, one the course does not answer is
    declined by search once and by the model once, one is answered the same way twice, one failed in one ask.

    What it checks: each signal is the share of the question's asks (a half here), both ways of declining count as
    declined, a verdict that differs between the asks counts as a change, and a failed ask leaves the question with
    no value in every signal.
    What it tells: a pass means the answer cells compare what the two asks really did. A failure points to
    answer_signals in matrix.py.
    Why it is worth having: a failed ask counted as a wrong answer would turn a stack error into a "killed" cell; a
    decline by search counted as an answer would hide the mutant that lets off-topic passages through.
    """
    signals = answer_signals(
        _by_case(
            _ask("QA-01", 1, "cited_gold"),
            _ask("QA-01", 2, "cited_other"),
            _ask("QA-U1", 1, "declined_no_passages", gold=None),
            _ask("QA-U1", 2, "declined_by_model", gold=None),
            _ask("QA-02", 1, "cited_gold"),
            _ask("QA-02", 2, "cited_gold"),
            _ask("QA-03", 1, "cited_gold"),
            _ask("QA-03", 2, "failed"),
        )
    )

    gold = signals["answerable questions answered with the gold section"]
    declined = signals["unanswerable questions declined"]
    unstable = signals["questions whose verdict changed between asks"]
    assert gold == {"QA-01": 0.5, "QA-02": 1.0, "QA-03": None}, f"wrong gold shares: {gold}"
    assert declined == {"QA-U1": 1.0}, f"both ways of declining must count: {declined}"
    assert unstable == {"QA-01": 1.0, "QA-U1": 1.0, "QA-02": 0.0, "QA-03": None}, f"wrong changes: {unstable}"


@pytest.mark.covers("US-019", "AC-6")
def test_a_question_that_moved_by_half_an_ask_moves_the_signal_by_a_half() -> None:
    """Judges a signal over 112 questions where 20 questions went from both asks right to one of two.

    What it checks: the change is 10 questions (20 halves), shown as such, and is judged beyond the threshold.
    What it tells: a pass means the rule reads shares, not yes / no. A failure points to signal_cell in verdicts.py.
    Why it is worth having: a share of 0.5 read as "yes" hides every question that became unstable on the mutant;
    the reference and the mutant would look the same.
    """
    reference: dict[str, float | None] = {f"q{index:03}": 1.0 for index in range(112)}
    mutant: dict[str, float | None] = {key: (0.5 if index < 20 else 1.0) for index, key in enumerate(reference)}

    cell = signal_cell(reference, mutant, 0.05)

    assert "112 -> 102 of 112" in cell.why, f"the halves were not counted: {cell.why}"
    assert cell.verdict == "killed", f"a loss of 10 of 112 must be noticed, got {cell.verdict}: {cell.why}"


@pytest.mark.covers("US-019", "AC-2")
@pytest.mark.parametrize(
    ("mutant", "asks", "problem"),
    [
        pytest.param(NO_MUTANT, (_ask("QA-01", 1, "cited_gold", calls=("cassette",)),), "recordings", id="replayed"),
        pytest.param(NO_MUTANT, (_ask("QA-01", 1, "cited_gold", calls=("miss",)),), "recordings", id="replay miss"),
        pytest.param(
            RATE_LIMITED, (_ask("QA-01", 1, "failed", calls=("upstream",)),), "not live", id="fault not planted"
        ),
        pytest.param(RATE_LIMITED, (_ask("QA-01", 1, "failed", calls=()),), "not live", id="no model call at all"),
        pytest.param(
            RATE_LIMITED,
            (
                _ask("QA-01", 1, "failed", calls=("fault",)),
                _ask("QA-U1", 1, "declined_no_passages", gold=None, calls=()),
            ),
            None,
            id="fault live, one ask declined before the model",
        ),
        pytest.param(
            NO_MUTANT,
            (_ask("QA-01", 1, "cited_gold"), _ask("QA-U1", 1, "declined_no_passages", calls=())),
            None,
            id="fresh",
        ),
        pytest.param(RATE_LIMITED, (_ask("QA-01", 1, "failed", calls=("fault", "fault")),), None, id="fault live"),
    ],
)
def test_an_answer_run_counts_only_with_fresh_answers_or_a_live_fault(
    mutant: Mutant, asks: tuple[AnswerCaseRecord, ...], problem: str | None
) -> None:
    """Checks answer runs whose model calls were served from recordings, or whose provider fault answered no call
    or not every call, against runs that are as they claim, one of them with an ask declined before the model.

    What it checks: the first kind gets a problem that names the cause, the second none.
    What it tells: a pass means an answer cell is judged only on answers the model gave in this run, and a provider
    mutant only when its fault answered every model call it made; an ask the app declines before calling the model
    does not count against the fault. A failure points to answer_problem in matrix.py.
    Why it is worth having: replayed answers make the reference and the run at the end identical, so no noise and no
    false alarm could ever show; a fault that did not reach the calls would read as a blind spot of the checks. The
    first experiment showed the other side: a rule asking a call of every ask made all three provider mutants
    invalid, because 9 questions are declined by search before the model.
    """
    found = answer_problem(_record(*asks), mutant)

    if problem is None:
        assert found is None, f"a run as it claims was rejected: {found}"
    else:
        assert found is not None and problem in found, f"expected a problem about {problem!r}, got {found!r}"


@pytest.mark.covers("US-019", "AC-2")
def test_a_cell_with_a_problem_is_invalid_whatever_its_signals_say() -> None:
    """Judges a mutant whose signals show a large change while its answers were replayed.

    What it checks: the cell is invalid and says why, not killed.
    What it tells: a pass means a proof that failed always wins over the numbers. A failure points to judge in
    verdicts.py.
    Why it is worth having: without it a run on recordings of another configuration would count as a kill.
    """
    values: dict[str, float | None] = {f"q{index}": 1.0 for index in range(40)}
    reference = Measured(signals={"s": values}, min_meaningful=0.05)
    mutant = Measured(
        signals={"s": dict.fromkeys(values, 0.0)}, min_meaningful=0.05, problem="3 of 80 calls from recordings"
    )

    cell = judge(reference, mutant)

    assert cell.verdict == "invalid", f"a run that failed its proof was judged {cell.verdict}: {cell.why}"


@pytest.mark.covers("US-019", "AC-7")
@pytest.mark.parametrize(
    ("labels", "named"),
    [
        pytest.param(("rate_limited", "rate_limited"), True, id="every failed ask labelled as expected"),
        pytest.param(("rate_limited", "provider_error"), False, id="one ask labelled otherwise"),
        pytest.param((None, None), False, id="no ask labelled"),
    ],
)
def test_the_labels_name_the_fault_only_when_every_labelled_ask_carries_it(
    labels: tuple[FailureLabel | None, ...], named: bool
) -> None:
    """Counts the labels of a provider mutant's asks: all as expected, one other, or none at all.

    What it checks: the labels "name" the fault only in the first case, and the note gives the counts.
    What it tells: a pass means the matrix says plainly when the failure labels blame the wrong cause. A failure
    points to label_note in matrix.py.
    Why it is worth having: the labels decide whether a failed case counts as a quality loss; a label that names the
    wrong cause on part of the asks would otherwise be hidden behind the most common one.
    """
    asks = [_ask("QA-01", index + 1, "failed", calls=("fault",), label=label) for index, label in enumerate(labels)]

    note, result = label_note(asks, RATE_LIMITED)

    assert result is named, f"labels {labels} were judged named={result}: {note}"


@pytest.mark.covers("US-019", "AC-5")
async def test_the_failure_labels_refuse_when_the_answer_run_left_no_record(tmp_path: Path) -> None:
    """Runs the failure-label check in a run folder where the answer run wrote nothing (it refused).

    What it checks: the check is recorded as refused with the reason, not as a result without a file.
    What it tells: a pass means a mutant that makes the answer run refuse is also seen by this column, as a refusal.
    A failure points to failure_label_check in oracles.py.
    Why it is worth having: a result without a file would stop the matrix command for the whole experiment.
    """
    result = await failure_label_check(tmp_path, NO_MUTANT)

    assert result.status == "refused", f"the check without an answer record was {result.status}"
    assert result.refusal is not None and "no_answer_record" in result.refusal, f"no reason given: {result.refusal}"


@pytest.mark.covers("US-019", "AC-6")
@pytest.mark.parametrize(
    ("mutant", "calls", "label", "left_out"),
    [
        pytest.param(NO_MUTANT, (), "no_response", True, id="no response on a run without a mutant"),
        pytest.param(NO_MUTANT, ("upstream",), "provider_error", True, id="a real provider error"),
        pytest.param(RATE_LIMITED, ("fault",), "provider_error", False, id="the planted fault, labelled otherwise"),
        pytest.param(RATE_LIMITED, (), "rate_limited", False, id="the label the mutant expects"),
        pytest.param(NO_MUTANT, ("upstream",), "app_error", False, id="the app's own failure"),
        pytest.param(NO_MUTANT, ("upstream",), "unusable_reply", False, id="the model's failure"),
    ],
)
def test_an_ask_is_left_out_only_when_the_stack_failed_for_a_reason_the_mutant_does_not_declare(
    mutant: Mutant, calls: tuple[Source, ...], label: FailureLabel, left_out: bool
) -> None:
    """Reads one failed ask of a run, on the run without a mutant or on a rate-limit mutant, with its label and the
    sources of its model calls in the ledger. Expects it left out only for a failure of the infrastructure that is
    neither the label the mutant expects nor a fault the proxy planted.

    What a pass tells: the matrix tells a slip of the stack from the mutant's effect and from the app's behaviour.
    A failure points to not_declared_asks or not_declared in matrix.py. Worth having because leaving out a planted
    fault would turn the provider mutants' kills into "nothing to compare", and leaving out an app error would hide a
    defect the mutant caused; counting a real outage would kill the mutant it met.
    """
    asks = [_ask("QA-01", 1, "failed", calls=calls, label=label), _ask("QA-02", 1, "cited_gold")]

    found = not_declared_asks(asks, mutant)

    assert found == (frozenset({"QA-01"}) if left_out else frozenset()), f"left out: {set(found)} for {label}"
