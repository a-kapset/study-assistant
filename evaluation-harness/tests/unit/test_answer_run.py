"""Unit tests of an answer run: the scores of one answer, the summary, the target verdict and a whole run against a
fake app, a fake corpus and a prepared ledger, never the running stack."""

import json
from collections.abc import Sequence
from datetime import UTC, datetime

import httpx2
import pytest
from pydantic import HttpUrl, JsonValue

from evalharness.datasets.qa_set import QaCase, QaSet, QaTargets
from evalharness.evaluation.answer_record import AnswerRunRecord
from evalharness.evaluation.answer_run import run, target_verdict
from evalharness.evaluation.retrieval import RunRefused
from evalharness.oracles.answer import AbstainReason, AnswerCaseScore, AnswerVerdict, AskReply, score_case, summarise
from evalharness.proxy.ledger import LedgerLine
from evalharness.settings import HarnessSettings
from evalharness.transport import REQUEST_ID_HEADER, SutTransport

COURSE = "evalharness_course"
# The gold section of the examples; "4.2.2" is one of its subsections, "4.20" only looks like one.
GOLD = "4.2"
TARGETS = QaTargets(answered_min=0.85, gold_cited_min=0.75, unsupported_answers_max=0)


def _reply(reason: AbstainReason | None = None, *sections: str | None) -> AskReply:
    """Build an answer that declines for ``reason`` or, without one, answers citing passages of ``sections``."""
    return AskReply(abstain_reason=reason, cited_sections=sections)


def _score(verdict: AnswerVerdict) -> AnswerCaseScore:
    """Build the score of one question with ``verdict``; whether search found its section does not matter here."""
    return AnswerCaseScore(verdict=verdict, gold_retrieved=None)


@pytest.mark.covers("US-011", "AC-3")
@pytest.mark.parametrize(
    ("gold_section", "reply", "verdict"),
    [
        pytest.param(GOLD, _reply(None, "4.2"), "cited_gold", id="answered-citing-the-gold-section"),
        pytest.param(GOLD, _reply(None, "5.1"), "cited_other", id="answered-citing-another-section"),
        pytest.param(None, _reply(None, "5.1"), "answered_unsupported", id="answered-what-the-course-does-not"),
        pytest.param(GOLD, _reply("no_passages"), "declined_no_passages", id="declined-by-search"),
        pytest.param(None, _reply("model_declined"), "declined_by_model", id="declined-by-the-model"),
        pytest.param(GOLD, None, "failed", id="no-answer-at-all"),
    ],
)
def test_each_answer_gets_the_verdict_of_what_the_app_did(
    gold_section: str | None, reply: AskReply | None, verdict: AnswerVerdict
) -> None:
    """Scores one answer of each kind and checks its verdict.

    A pass means every case of a run is put in the right class: answered with or without the gold section, answered
    a question the course does not answer, declined by search or by the model, or not answered at all. A failure
    means a run counts the wrong thing, worst of all an answer from the model's memory counted as an ordinary
    answer, so the one blocker of the set would never fire. Only this test checks the classes one by one.
    """
    score = score_case(gold_section, reply, None)

    assert score.verdict == verdict, f"gold section {gold_section}, reply {reply}: got {score.verdict}"


@pytest.mark.covers("US-011", "AC-3")
@pytest.mark.parametrize(
    ("section", "counts"),
    [
        pytest.param("4.2", True, id="the-section"),
        pytest.param("4.2.2", True, id="a-subsection"),
        pytest.param("4.20", False, id="an-id-with-the-same-start"),
        pytest.param(None, False, id="text-before-any-heading"),
    ],
)
def test_a_citation_counts_only_for_the_gold_section_and_its_subsections(section: str | None, counts: bool) -> None:
    """Scores an answer citing one passage of the given section and checks whether it counts as citing the gold one.

    A pass means a citation is credited by the same rule a search hit is. A failure means answers are credited for a
    section that only looks like the gold one ("4.20" for "4.2"), or not credited for its subsections, so the share
    of answers citing the gold section would be wrong in every run.
    """
    score = score_case(GOLD, _reply(None, section), None)

    assert (score.verdict == "cited_gold") is counts, f"a citation of {section!r} for gold section {GOLD}: {score}"


@pytest.mark.covers("US-011", "AC-7")
@pytest.mark.parametrize(
    ("search_sections", "found"),
    [
        pytest.param(["5.1", "4.2.2"], True, id="gold-subsection-found-second"),
        pytest.param(["5.1", "4.20"], False, id="only-other-sections"),
        pytest.param([], False, id="search-found-nothing"),
        pytest.param(None, None, id="search-failed"),
    ],
)
def test_the_stage_says_whether_search_found_the_gold_section(
    search_sections: list[str | None] | None, found: bool | None
) -> None:
    """Scores an answer that cites another section, with the given search result; checks whether the gold section
    counts as found by search.

    A pass means a wrong answer is placed at the right stage: search did not find the section (fix retrieval), or it
    did and the answer step did not use it (fix the prompt). A failure means the report sends the maintainer to the
    wrong stage, and a failed search would look like a search that found nothing.
    """
    score = score_case(GOLD, _reply(None, "5.1"), search_sections)

    assert score.gold_retrieved is found, f"search found {search_sections}: gold_retrieved is {score.gold_retrieved}"


@pytest.mark.covers("US-011", "AC-4", "AC-6")
def test_a_failed_ask_stays_out_of_every_share() -> None:
    """Sums up three answerable questions: one answered, one declined, one failed; expects one answered of two.

    A pass means a question that got no answer at all (for example a recording that no longer matches) does not
    count as a decline. A failure means a broken recording lowers the share of answered questions, so a fault of
    the run looks like an app that declines too often.
    """
    scores = [_score("cited_gold"), _score("declined_by_model"), _score("failed")]

    summary = summarise([GOLD, GOLD, GOLD], scores)

    assert (summary.answered, summary.failed) == (0.5, 1), f"expected 1 of 2 answered and 1 failed: {summary}"


@pytest.mark.covers("US-011", "AC-4")
def test_gold_cited_is_counted_over_the_answered_questions_only() -> None:
    """Sums up four answerable questions: two cite the gold section, one cites another, one is declined; expects
    two thirds.

    A pass means the share of answers citing the gold section says how good the answers are that the app gave. A
    failure means declines pull it down too, so the two numbers would measure the same thing and a prompt that
    declines more would look like one that cites worse.
    """
    scores = [_score("cited_gold"), _score("cited_gold"), _score("cited_other"), _score("declined_no_passages")]

    summary = summarise([GOLD] * 4, scores)

    assert summary.gold_cited == pytest.approx(2 / 3), f"expected 2 of 3 answers to cite the gold section: {summary}"


@pytest.mark.covers("US-011", "AC-5")
def test_one_unsupported_answer_fails_the_targets_whatever_the_shares() -> None:
    """Sums up a run where every answerable question cites the gold section and one unanswerable one is answered;
    expects the unsupported-answers target to be missed while both shares are met.

    A pass means one answer without support in the course is enough to miss the targets. A failure means such an
    answer hides behind good averages: the app would pass while it answers from the model's memory.
    """
    summary = summarise(
        [GOLD, GOLD, None], [_score("cited_gold"), _score("cited_gold"), _score("answered_unsupported")]
    )

    verdicts = target_verdict(summary, TARGETS)

    expected = {"answered": "met", "gold_cited": "met", "unsupported_answers": "below"}
    assert verdicts == expected, f"one unsupported answer among perfect shares gave {verdicts}"


@pytest.mark.covers("US-011", "AC-6")
def test_a_run_with_a_failed_question_has_no_target_verdict() -> None:
    """Sums up a run with one answer citing the gold section and one failed question; expects no verdict at all.

    A pass means a run that did not measure every question is reported as invalid instead of being judged. A
    failure means a run with broken recordings is judged on the questions that happened to work, and its verdict
    is read as the quality of the app.
    """
    summary = summarise([GOLD, GOLD], [_score("cited_gold"), _score("failed")])

    assert target_verdict(summary, TARGETS) is None, "a run with a failed question was judged against its targets"


def _qa(*cases: tuple[str, str | None]) -> QaSet:
    """Build a Q&A set of the fake course from (id, gold section) pairs."""
    return QaSet(
        course_id=COURSE,
        targets=TARGETS,
        cases=tuple(QaCase(id=case_id, question=f"Question {case_id}?", gold_section=gold) for case_id, gold in cases),
    )


class _Corpus:
    """A stored corpus with a fixed list of sections, in place of the database reader."""

    def __init__(self, sections: set[str]) -> None:
        """Keep the sections the corpus reports."""
        self._sections = sections

    def section_ids(self, course_id: str) -> set[str]:
        """Return the fixed sections."""
        return self._sections

    def snapshot(self, course_id: str) -> str:
        """Return a fixed corpus name."""
        return "evalharness_snapshot"

    def chunk_texts(self, course_id: str) -> list[str]:
        """Return no text: an answer run does not read it."""
        return []


def _line(request_id: str, model: str | None, cost: float | None, source: str = "cassette") -> LedgerLine:
    """Build one ledger line of a model call made for ``request_id``."""
    return LedgerLine.model_validate(
        {
            "at": datetime.now(UTC),
            "request_id": request_id,
            "key": "evalharness_key",
            "source": source,
            "status": 200 if source != "miss" else 404,
            "model_requested": "openai/evalharness_model",
            "model_returned": model,
            "input_tokens": None,
            "cached_tokens": None,
            "output_tokens": None,
            "reasoning_tokens": None,
            "latency_ms": 1.0,
            "cost_usd": cost,
            "error": None if source != "miss" else "replay_miss",
        }
    )


class _App:
    """A fake app: search finds one passage of section 4.2, ask answers citing it; every ask is written to a ledger
    as one model call answered from a recording, the way the proxy writes it."""

    def __init__(self, ledger: list[LedgerLine]) -> None:
        """Keep the ledger the asks are written to."""
        self._ledger = ledger

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        """Answer a search or an ask of the fake course."""
        body: dict[str, JsonValue] = {"hits": [{"section_id": GOLD}]}
        if request.url.path == "/v1/ask":
            body = {"abstain_reason": None, "citations": [{"section_id": GOLD}], "invalid_citations": 0}
            request_id = request.headers[REQUEST_ID_HEADER]
            self._ledger.append(_line(request_id, "evalharness_model-2026", 0.001))
        return httpx2.Response(200, content=json.dumps(body))


async def _run(qa: QaSet, ledger: list[LedgerLine], sections: set[str] | None = None) -> AnswerRunRecord:
    """Run ``qa`` against the fake app; the corpus holds the gold sections unless ``sections`` says otherwise."""
    gold = {case.gold_section for case in qa.cases if case.gold_section is not None}
    settings = HarnessSettings(sut_base_url=HttpUrl("http://sut.test"))

    def read() -> Sequence[LedgerLine]:
        """Return the ledger as it is after the last question."""
        return ledger

    async with SutTransport(settings, transport=httpx2.MockTransport(_App(ledger))) as sut:
        return await run(
            qa,
            sut,
            _Corpus(sections if sections is not None else gold),
            read,
            qa_file="evalharness_qa.toml",
            qa_bytes=b"evalharness",
            harness_commit="evalharness_commit",
            harness_dirty=False,
        )


@pytest.mark.covers("US-011", "AC-8")
async def test_only_the_runs_own_model_calls_are_recorded() -> None:
    """Runs one question against a fake app while the ledger already holds a call of another request, made by
    another model; expects the record to name only the call and the model of this run's question.

    A pass means the record says which model answered this run and what its answers cost, even though the ledger
    holds every call ever made on the machine. A failure means a run reports another run's model or cost as its
    own: the first dry run of this runner named a model that never answered it, because the names were read from
    the whole ledger.
    """
    ledger = [_line("evalharness_other_request", "evalharness_other_model", 0.5)]

    record = await _run(_qa(("QA-1", GOLD)), ledger)

    case = record.cases[0]
    seen = (record.models_returned, case.model_calls, case.cost_usd)
    assert seen == (("evalharness_model-2026",), ("cassette",), 0.001), f"the record mixed in another call: {seen}"


@pytest.mark.covers("US-011", "AC-2")
async def test_a_gold_section_missing_from_the_corpus_refuses_the_run() -> None:
    """Runs a set whose gold section 5.6 is not a section of the stored course; expects "gold_section_missing"
    before any question is asked.

    A pass means a mistyped gold section stops the run. A failure means its question can never cite that section,
    so it scores "cited_other" in every run and the share of answers citing the gold section is too low for no
    reason in the app.
    """
    ledger: list[LedgerLine] = []

    with pytest.raises(RunRefused) as refused:
        await _run(_qa(("QA-1", GOLD), ("QA-2", "5.6")), ledger, sections={GOLD})

    assert (refused.value.code, ledger) == ("gold_section_missing", []), f"got {refused.value.code}, calls {ledger}"
