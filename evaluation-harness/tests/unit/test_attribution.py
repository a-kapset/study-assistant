"""Unit tests of the stage the request traces name for a mutant: asks described stage by stage by hand, never the
running stack."""

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from evalharness.evaluation.answer_record import AnswerCaseRecord, AnswerRunRecord
from evalharness.failures import FailureLabel
from evalharness.mutation.attribution import (
    AskOutcome,
    AttributionVerdict,
    MutantStage,
    accuracy,
    ask_outcomes,
    attribute,
    named_stage,
    verdict,
)
from evalharness.mutation.catalog import Component, Mutant
from evalharness.mutation.matrix import Matrix, Run, build_attribution
from evalharness.mutation.record import CheckResult, MutantRunRecord, Proof
from evalharness.oracles.answer import AnswerCaseScore
from evalharness.proxy.ledger import Source
from evalharness.stages import StageName

# An ask as the run without a mutant shows it: the model answered at the first attempt.
_HEALTHY: dict[StageName, str] = {
    "embedding": "200",
    "search": "ok hits=evalharness_c1,evalharness_c2",
    "rerank": "absent",
    "provider": "200",
    "generate": "unset",
}


def _ask(**changed: str) -> dict[StageName, str]:
    """Return the healthy ask with the given stages described otherwise."""
    return {**_HEALTHY, **changed}  # type: ignore[dict-item]


def _case(
    case_id: str,
    stages: Mapping[StageName, str] | None,
    calls: Sequence[Source] = ("cassette",),
    label: FailureLabel | None = None,
) -> AnswerCaseRecord:
    """Build the record of one ask of ``case_id`` with its stages, the sources of its model calls and the failure
    label the ask got."""
    return AnswerCaseRecord(
        case_id=case_id,
        repetition=1,
        gold_section="4.2",
        search_request_id=f"evalharness_s_{case_id}",
        ask_request_id=f"evalharness_a_{case_id}",
        ask_error=None,
        failure_label=label,
        abstain_reason=None,
        cited_sections=("4.2",),
        invalid_citations=0,
        search_sections=("4.2",),
        model_calls=tuple(calls),
        cost_usd=0.0,
        score=AnswerCaseScore(verdict="cited_gold", gold_retrieved=True),
        stages=dict(stages) if stages is not None else None,
    )


@pytest.mark.covers("US-019", "AC-9")
@pytest.mark.parametrize(
    ("ask", "reference", "expected"),
    [
        pytest.param(
            _ask(search="empty hits=", provider="absent", generate="absent"),
            [_HEALTHY],
            "search",
            id="first-difference-not-the-last",
        ),
        pytest.param(
            _ask(embedding="absent", search="absent"),
            [_HEALTHY],
            "embedding",
            id="a-missing-stage-is-a-difference",
        ),
        pytest.param(
            _ask(provider="500 200"),
            [_HEALTHY, _ask(provider="500 200")],
            None,
            id="the-references-own-variation-names-nothing",
        ),
    ],
)
def test_an_ask_names_the_first_stage_that_none_of_the_reference_asks_show(
    ask: dict[StageName, str], reference: list[dict[StageName, str]], expected: StageName | None
) -> None:
    """What it checks: an ask is compared stage by stage, in the order an ask passes them, with the reference asks of
    its question. A search that found nothing (so no model call followed) names the search, not the later stages
    it changed; a skipped search names the first stage it removed; a retry the reference itself showed in one of its
    repetitions names nothing.

    What it tells: a pass means each ask points at the stage where it first left the run without a mutant. A
    failure means the rule blames a later stage, ignores a skipped one, or calls the app's own variation a fault.

    Why it is worth having: an upstream fault changes every later stage too (Nassery's beta-test case: the symptom at
    the end, the cause in a middle stage), so only the first difference can point at the cause; and the model's
    provider fails now and then on its own, which must not be read as a planted fault.
    """
    named = named_stage(ask, reference)

    assert named == expected, f"the ask named {named!r}, expected {expected!r}"


@pytest.mark.covers("US-019", "AC-9")
@pytest.mark.parametrize(
    ("outcomes", "noise", "expected"),
    [
        pytest.param(["search"] * 6 + ["provider"] * 2 + ["none"] * 2, 0, "search", id="majority"),
        pytest.param(["search"] * 3 + ["provider"] * 3 + ["none"] * 4, 0, "several", id="no-majority"),
        pytest.param(["provider"] * 2 + ["none"] * 8, 2, "none", id="not-above-the-noise"),
        pytest.param(["not_measured"] * 9 + ["none"], 0, "not_measured", id="nothing-measured"),
    ],
)
def test_a_mutant_is_attributed_to_the_stage_most_of_its_differing_asks_name(
    outcomes: list[AskOutcome], noise: int, expected: MutantStage
) -> None:
    """What it checks: a mutant run gets the stage named by more than half of its asks that named one; "several"
    when no stage has that majority; "none" when no more asks named a stage than in the run without a mutant at the
    end of the experiment (``noise``); "not measured" when its asks could not be compared.

    What it tells: a pass means one or two odd asks never decide where a mutant is placed, and the app's own
    differences are not read as the mutant's. A failure names which of the four cases is decided wrongly.

    Why it is worth having: with two hundred asks per run, a few differ by chance (a provider error, a retry); a rule
    taking the most named stage without a majority, or ignoring the run without a mutant, would place a mutant that
    changes nothing at whatever stage happened to vary.
    """
    attributed = attribute(outcomes, noise, "search")

    assert attributed.stage == expected, f"outcomes {outcomes} with noise {noise} gave {attributed.stage!r}"


@pytest.mark.covers("US-019", "AC-9")
def test_a_part_without_a_span_is_not_attributable_and_the_accuracy_counts_mutants() -> None:
    """What it checks: a mutant of how the course is cut into passages, which the trace places at the search, is
    "not attributable from the trace", not "wrong"; the other parts are right or wrong by the stage of their span;
    the accuracy of a part is right mutants among its counted mutants, with an interval over mutants.

    What it tells: a pass means the share reports how often the trace pointed at the faulty part where it could.
    A failure means a part the trace cannot show lowers the share, or the asks, not the mutants, are counted.

    Why it is worth having: cutting the course happens before any ask, so no request span can show it; counting it
    as a wrong attribution would blame the rule for a limit of tracing. Counting asks instead of mutants would give
    a narrow interval from two hundred asks that share one fault and are not independent evidence.
    """
    verdicts: list[tuple[Component, AttributionVerdict]] = [
        ("ingestion", verdict("ingestion", "search")),
        ("search", verdict("search", "search")),
        ("search", verdict("search", "search")),
        ("provider", verdict("provider", "generate")),
    ]

    shares = accuracy(verdicts)

    seen = (verdicts[0][1], verdicts[3][1], shares["search"].counted, shares["all"].right, shares["all"].counted)
    assert seen == ("not_attributable", "wrong", 2, 2, 3), f"verdicts {verdicts}, shares {shares}"


@pytest.mark.covers("US-019", "AC-9")
def test_a_difference_after_a_missed_recording_is_not_measured_and_an_earlier_one_counts() -> None:
    """What it checks: in a run served from the proxy's recordings, an ask whose first difference is at the model
    provider while its call found no recording is "not measured"; so is one whose reference ask missed; an ask whose
    first difference is the search counts even though its model call missed.

    What it tells: a pass means the proxy's error for a missing recording is never read as the mutant's fault, while
    the stages before the model call, which do not depend on the recordings, still count. A failure means a mutant
    that only changes the model is placed at the provider by the replay, or every mutant of the search is lost.

    Why it is worth having: the attribution run is made without paid model calls, so any request the recordings do
    not cover gets the proxy's 404; a model swap would otherwise show up as "provider, wrong".
    """
    reference = [_case("QA-1", _HEALTHY), _case("QA-2", _HEALTHY, ("miss",)), _case("QA-3", _HEALTHY)]
    cases = [
        _case("QA-1", _ask(provider="404"), ("miss",)),
        _case("QA-2", _ask(provider="429 429 429"), ("fault",)),
        _case("QA-3", _ask(search="empty hits=", provider="404"), ("miss",)),
    ]

    outcomes = ask_outcomes(cases, reference)

    assert outcomes == ["not_measured", "not_measured", "search"], f"the asks gave {outcomes}"


def _stored_run(folder: Path, mutant_id: str, mutant: Mutant, cases: Sequence[AnswerCaseRecord]) -> Run:
    """Write one run of an attribution experiment to ``folder``: its answer record and its mutant run record."""
    folder.mkdir()
    answers = AnswerRunRecord(
        run_id="0" * 32,
        started_at=datetime(2026, 11, 20, tzinfo=UTC),
        harness_commit="0" * 40,
        harness_dirty=False,
        qa_file="evalharness_qa.toml",
        qa_sha256="a" * 64,
        course_id="evalharness_course",
        corpus_snapshot="b" * 64,
        app_settings={"git_commit": "app"},
        models_returned=(),
        repetitions=1,
        summaries=(),
        cases=tuple(cases),
    )
    (folder / "answers.json").write_text(answers.model_dump_json(), encoding="utf-8")
    record = MutantRunRecord.model_validate(
        {
            "run_id": f"evalharness_{mutant_id}",
            "experiment_id": "evalharness_experiment",
            "started_at": datetime(2026, 11, 20, tzinfo=UTC),
            "harness_commit": "commit",
            "harness_dirty": False,
            "catalog_file": "catalog.toml",
            "catalog_sha256": "evalharness_sha",
            "course_id": "evalharness_course",
            "mutant_id": mutant_id,
            "mutant": mutant,
            "app_settings": {"git_commit": "app"},
            "applied": Proof(changed={}, problem=None),
            "checks": {"answer_run": CheckResult(status="ran", result_file="answers.json")},
            "restored": Proof(changed={}, problem=None),
        }
    )
    return Run(record, folder)


@pytest.mark.covers("US-019", "AC-9")
def test_asks_the_stack_failed_on_are_compared_not_left_out(tmp_path: Path) -> None:
    """What it checks: an attribution experiment written to a folder: the run without a mutant answered both
    questions; under a mutant with an unreachable embedding server both asks were declined and labelled as a failure
    of the stack ("degraded_retrieval", which the mutant does not name). The mutant must be attributed to the
    embedding stage from both asks, with no ask left out.

    What it tells: a pass means the attribution reads how a request went wrong, failures included. A failure means a
    label of the stack removes the asks that show the mutant's effect, and the mutant reads "none".

    Why it is worth having: the first two attribution runs (2026-11-20) left such asks out by the rule of the judged
    cells (a failure of the stack the mutant does not declare says nothing about quality): first 194 of 224 asks of
    the mutant with one hit (missed recordings), then all 224 asks of the unreachable embedding server; each read
    "none".
    """
    unreachable = Mutant(
        kind="configuration",
        component="embedding",
        settings={"APP_EMBEDDING_SERVICE__URL": "http://embedder:1/"},
        expected_symptom="evalharness",
        origin="invented",
    )
    declined = _ask(embedding="error:ConnectError", search="degraded embedder_unavailable hits=", provider="absent")
    reference = _stored_run(
        tmp_path / "a",
        "baseline",
        Mutant(kind="baseline", expected_symptom="evalharness"),
        [_case("QA-1", _HEALTHY), _case("QA-2", _HEALTHY)],
    )
    mutant = _stored_run(
        tmp_path / "b",
        "embedder_unreachable",
        unreachable,
        [_case(case_id, declined, (), "degraded_retrieval") for case_id in ("QA-1", "QA-2")],
    )
    matrix = Matrix("evalharness_experiment", reference, (), (), {})

    table = build_attribution([reference, mutant], matrix)

    attribution = table.rows[0].attribution
    assert attribution is not None, f"the mutant run was not read: {table.rows[0].why_not}"
    assert (attribution.stage, dict(attribution.counts)) == ("embedding", {"embedding": 2}), (
        f"the mutant was attributed to {attribution.stage!r} from {dict(attribution.counts)}"
    )
