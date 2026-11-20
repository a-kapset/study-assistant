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
from evalharness.evaluation.answer_run import ask_timeout_s, run, target_verdict
from evalharness.evaluation.retrieval import RunRefused
from evalharness.failures import FailureLabel
from evalharness.oracles.answer import (
    AbstainReason,
    AnswerCaseScore,
    AnswerVerdict,
    AskReply,
    flipped_cases,
    score_case,
    summarise,
)
from evalharness.proxy.keys import repetition_of
from evalharness.proxy.ledger import LedgerLine
from evalharness.settings import HarnessSettings
from evalharness.traces import TraceReader
from evalharness.transport import REQUEST_ID_HEADER, SutTransport

COURSE = "evalharness_course"
# The part of GET /v1/config a run reads: the app's code and how long it waits for the servers it calls, as the
# stack's .env sets them today.
APP_SETTINGS: dict[str, JsonValue] = {
    "git_commit": "evalharness_app",
    "search": {"rerank": False, "max_k": 50},
    "embedding_service": {"timeout_seconds": 30.0},
    "rerank": {"timeout_seconds": 30.0},
    "openai": {"timeout_seconds": 60.0},
}
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
    """Sums up a run where all 25 answerable questions cite the gold section and one unanswerable one is answered;
    expects the unsupported-answers target to be missed while both shares are met. 25 questions are needed for
    "met": with fewer, even a perfect share has an interval whose lower end is under the 0.85 target.

    A pass means one answer without support in the course is enough to miss the targets, even when both shares are
    confirmed by their intervals. A failure means such an answer hides behind good shares: the app would pass while
    it answers from the model's memory.
    """
    summary = summarise([GOLD] * 25 + [None], [_score("cited_gold")] * 25 + [_score("answered_unsupported")])

    verdicts = target_verdict(summary, TARGETS)

    expected = {"answered": "met", "gold_cited": "met", "unsupported_answers": "below"}
    assert verdicts == expected, f"one unsupported answer among confirmed shares gave {verdicts}"


@pytest.mark.covers("US-011", "AC-4")
def test_gold_cited_is_judged_over_the_answered_questions() -> None:
    """Sums up 30 answerable questions: 25 answered citing the gold section, 5 declined; expects the gold-cited
    target to be met (25 of 25, interval from 0.87).

    A pass means the verdict on citations counts only the answers the app gave, as the printed share does. A
    failure means the verdict counts the declined questions too (25 of 30, which is not confirmed against 0.75):
    declines would then pull the citation verdict down, and the two targets would judge the same thing.
    """
    summary = summarise([GOLD] * 30, [_score("cited_gold")] * 25 + [_score("declined_by_model")] * 5)

    verdicts = target_verdict(summary, TARGETS)

    assert verdicts is not None, "a run without failed questions got no verdict"
    assert verdicts["gold_cited"] == "met", f"25 of 25 answers citing the gold section gave {verdicts['gold_cited']!r}"


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
    as one model call answered from a recording, the way the proxy writes it. Asks of the repetition named in
    ``declines_in`` are declined by the model instead. Its readiness says ``ready``. Every search and ask is
    remembered as (path, repetition)."""

    def __init__(self, ledger: list[LedgerLine], declines_in: int | None = None, ready: str = "ok") -> None:
        """Keep the ledger the asks are written to, the repetition whose asks are declined and the readiness state."""
        self._ledger = ledger
        self._declines_in = declines_in
        self._ready = ready
        self.requests: list[tuple[str, int]] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        """Answer a search or an ask of the fake course; the settings and readiness requests are answered but not
        remembered."""
        if request.url.path == "/v1/config":
            return httpx2.Response(200, json={"app_version": "0.1.0", "settings": APP_SETTINGS})
        if request.url.path == "/ready":
            return httpx2.Response(200, json={"status": self._ready, "checks": {"db": "ok", "embedder": "ok"}})
        request_id = request.headers[REQUEST_ID_HEADER]
        repetition = repetition_of(request_id)
        self.requests.append((request.url.path, repetition))
        body: dict[str, JsonValue] = {"hits": [{"section_id": GOLD}]}
        if request.url.path == "/v1/ask":
            body = {"abstain_reason": None, "citations": [{"section_id": GOLD}], "invalid_citations": 0}
            if repetition == self._declines_in:
                body = {"abstain_reason": "model_declined", "citations": [], "invalid_citations": 0}
            self._ledger.append(_line(request_id, "evalharness_model-2026", 0.001))
        return httpx2.Response(200, content=json.dumps(body))


async def _run(
    qa: QaSet,
    ledger: list[LedgerLine],
    sections: set[str] | None = None,
    *,
    app: _App | None = None,
    repetitions: int = 1,
    allow_degraded: bool = False,
) -> AnswerRunRecord:
    """Run ``qa`` against the fake app (a new one unless ``app`` is given) ``repetitions`` times; the corpus holds
    the gold sections unless ``sections`` says otherwise."""
    gold = {case.gold_section for case in qa.cases if case.gold_section is not None}
    settings = HarnessSettings(sut_base_url=HttpUrl("http://sut.test"))

    def read() -> Sequence[LedgerLine]:
        """Return the ledger as it is after the last question."""
        return ledger

    async with SutTransport(settings, transport=httpx2.MockTransport(app or _App(ledger))) as sut:
        return await run(
            qa,
            sut,
            _Corpus(sections if sections is not None else gold),
            read,
            qa_file="evalharness_qa.toml",
            qa_bytes=b"evalharness",
            harness_commit="evalharness_commit",
            harness_dirty=False,
            repetitions=repetitions,
            allow_degraded=allow_degraded,
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


@pytest.mark.covers("US-014", "AC-1")
async def test_every_question_is_asked_once_per_repetition_and_searched_once() -> None:
    """Runs two questions with three repetitions against a fake app that reads the repetition from the request id
    the way the proxy does; expects each question searched once and asked once in each of repetitions 1, 2 and 3.

    A pass means every repetition reaches the proxy with its own mark, so the model answers each one anew, and the
    search, which gives the same hits every time, is not asked again. A failure means either the repetitions carry
    no mark (the proxy serves the first answer again and every question looks stable) or the run asks too often or
    too seldom.
    """
    app = _App([])

    await _run(_qa(("QA-1", GOLD), ("QA-2", None)), [], app=app, repetitions=3)

    expected = [("/v1/search", 1)] * 2 + [("/v1/ask", n) for n in (1, 1, 2, 2, 3, 3)]
    assert app.requests == expected, f"requests (path, repetition) sent by the run: {app.requests}"


@pytest.mark.covers("US-014", "AC-2")
async def test_each_repetition_is_summed_up_on_its_own() -> None:
    """Runs one answerable question with three repetitions against a fake app whose model declines it in the second
    repetition only; expects the answered shares 1, 0 and 1, one per repetition in order.

    A pass means every repetition gets its own results and target verdict, so one repetition's luck is not hidden in
    a total. A failure means the summaries mix repetitions, are out of order, or count every repetition as the
    first.
    """
    ledger: list[LedgerLine] = []

    record = await _run(_qa(("QA-1", GOLD)), ledger, app=_App(ledger, declines_in=2), repetitions=3)

    answered = [summary.answered for summary in record.summaries]
    assert answered == [1.0, 0.0, 1.0], f"answered share per repetition: {answered}"


@pytest.mark.covers("US-014", "AC-3")
def test_a_question_flips_when_its_verdict_is_not_the_same_in_every_repetition() -> None:
    """Counts the flips of four questions over three repetitions: one always cites the gold section, one cites
    another section once, one is always declined, and one failed in a repetition.

    Expects three questions counted and one flipped: a change between citing the gold section and citing another
    one is a flip, though the question was answered every time; a question with a failed repetition is left out,
    as that repetition measured nothing. A failure means the flip rate counts the wrong questions: too low when only
    answered / declined changes count, too high when a failed request counts as a changed answer.
    """
    verdicts: dict[str, list[AnswerVerdict]] = {
        "QA-1": ["cited_gold", "cited_gold", "cited_gold"],
        "QA-2": ["cited_gold", "cited_other", "cited_gold"],
        "QA-3": ["declined_by_model", "declined_by_model", "declined_by_model"],
        "QA-4": ["cited_gold", "failed", "cited_gold"],
    }

    counted, flipped = flipped_cases(verdicts)

    assert (counted, flipped) == (3, ["QA-2"]), f"counted {counted}, flipped {flipped}"


@pytest.mark.covers("US-016", "AC-6", "AC-7")
@pytest.mark.covers("US-011", "AC-6")
@pytest.mark.parametrize(
    ("reply", "failure", "verdict"),
    [
        pytest.param(None, "unusable_reply", "unusable_reply", id="the-model-spoiled-its-reply"),
        pytest.param(None, "rate_limited", "failed", id="the-provider-limited-the-calls"),
        pytest.param(_reply(None, "4.2"), "error_as_answer", "failed", id="the-app-showed-an-error-as-an-answer"),
    ],
)
def test_the_failure_label_decides_whether_a_case_counts_for_quality(
    reply: AskReply | None, failure: FailureLabel, verdict: AnswerVerdict
) -> None:
    """Scores an answerable question that failed for a reason of each category: the model, the provider, the app.

    A pass means only the model's own failure is scored as an answer, and a failure of the stack or the app is
    counted apart, even when the app's response looked like an answer citing the gold section. A failure means a
    provider outage lowers the quality numbers, or an error the app passed off as an answer raises them.
    """
    score = score_case(GOLD, reply, None, failure)

    assert score.verdict == verdict, f"a failure labelled {failure!r} was scored {score.verdict!r}"


@pytest.mark.covers("US-016", "AC-7")
@pytest.mark.covers("US-011", "AC-6")
def test_an_unusable_reply_counts_as_not_answered_and_keeps_the_target_verdict() -> None:
    """Sums up two answerable questions: one answer citing the gold section, one reply the model spoiled.

    Expects one answered of two and a target verdict. A pass means a model that sometimes returns nothing usable
    shows up in the answered share and in the verdict. A failure means such replies are dropped as if the stack had
    failed, and the run looks better than the model is.
    """
    summary = summarise([GOLD, GOLD], [_score("cited_gold"), _score("unusable_reply")])

    assert (summary.answered, summary.failed) == (0.5, 0), f"expected 1 of 2 answered and none failed: {summary}"
    assert target_verdict(summary, TARGETS) is not None, "a model failure took the run's target verdict away"


@pytest.mark.covers("US-016", "AC-5")
async def test_a_failed_ask_is_recorded_with_the_label_its_model_calls_give() -> None:
    """Runs one question against a fake app whose ask fails with "llm_failed" after three model calls that the
    provider answered with a rate limit, as the ledger shows.

    A pass means the record names the cause the app's single code hides. A failure means the label is not read
    from this request's ledger lines, so every provider failure looks alike in the record.
    """
    ledger: list[LedgerLine] = []

    def app(request: httpx2.Request) -> httpx2.Response:
        """Serve the settings, the readiness and a search; fail the ask the way the app does after a rate limit."""
        if request.url.path == "/v1/config":
            return httpx2.Response(200, json={"settings": APP_SETTINGS})
        if request.url.path == "/ready":
            return httpx2.Response(200, json={"status": "ok", "checks": {"db": "ok", "embedder": "ok"}})
        if request.url.path == "/v1/search":
            return httpx2.Response(200, json={"hits": [{"section_id": GOLD}]})
        line = _line(request.headers[REQUEST_ID_HEADER], None, None, source="upstream")
        ledger.extend([line.model_copy(update={"status": 429, "error": "rate_limit_exceeded"})] * 3)
        envelope = {"status": "error", "error": {"code": "llm_failed", "message": "429"}, "request_id": "x"}
        return httpx2.Response(502, json=envelope)

    gold = {GOLD}
    settings = HarnessSettings(sut_base_url=HttpUrl("http://sut.test"))
    async with SutTransport(settings, transport=httpx2.MockTransport(app)) as sut:
        record = await run(
            _qa(("QA-1", GOLD)),
            sut,
            _Corpus(gold),
            lambda: ledger,
            qa_file="evalharness_qa.toml",
            qa_bytes=b"evalharness",
            harness_commit="evalharness_commit",
            harness_dirty=False,
        )

    case = record.cases[0]
    assert (case.failure_label, case.score.verdict) == ("rate_limited", "failed"), f"recorded as {case}"


@pytest.mark.covers("US-016", "AC-9")
@pytest.mark.covers("US-011", "AC-6")
async def test_a_question_declined_on_a_search_without_the_embedder_is_a_failure_of_the_stack() -> None:
    """Runs one answerable and one unanswerable question against a fake app that answers every ask as the real one
    does while its embedding server cannot be reached: 200, declined with the reason "search_degraded", no model
    call. Expects both questions labelled "degraded_retrieval" and scored "failed", and no target verdict.

    A pass means an answer run that meets a stopped embedding server ends with a record, in which those questions
    are an outage and not a result. A failure means either the run cannot read the app's third reason for
    declining and stops with an error after its last question, losing every answer it collected, or the decline
    is scored as an answer: an unanswerable question would then count as answered without support, the one
    blocker of the set. The fake stands for the real response, which the api test with the stopped server reads.
    """

    def app(request: httpx2.Request) -> httpx2.Response:
        """Serve the settings, the readiness and a search; decline every ask for a search by words only."""
        if request.url.path == "/v1/config":
            return httpx2.Response(200, json={"settings": APP_SETTINGS})
        if request.url.path == "/ready":
            return httpx2.Response(200, json={"status": "ok", "checks": {"db": "ok", "embedder": "ok"}})
        if request.url.path == "/v1/search":
            return httpx2.Response(200, json={"hits": [{"section_id": GOLD}]})
        return httpx2.Response(200, json={"abstain_reason": "search_degraded", "citations": [], "invalid_citations": 0})

    settings = HarnessSettings(sut_base_url=HttpUrl("http://sut.test"))
    async with SutTransport(settings, transport=httpx2.MockTransport(app)) as sut:
        record = await run(
            _qa(("QA-1", GOLD), ("QA-2", None)),
            sut,
            _Corpus({GOLD}),
            list,
            qa_file="evalharness_qa.toml",
            qa_bytes=b"evalharness",
            harness_commit="evalharness_commit",
            harness_dirty=False,
        )

    recorded = [(case.failure_label, case.score.verdict) for case in record.cases]
    assert recorded == [("degraded_retrieval", "failed")] * 2, f"(label, verdict) of the two questions: {recorded}"
    assert target_verdict(record.summaries[0], TARGETS) is None, (
        "a run whose questions were declined for a stopped embedding server was judged against its targets"
    )


@pytest.mark.covers("US-011", "AC-11")
async def test_an_answer_run_on_a_degraded_stack_is_refused_before_the_first_question() -> None:
    """Runs one question against a fake app whose readiness says "degraded" with 200, while its searches and asks
    would answer. Expects the refusal "stack_not_ready" and no search or ask sent.

    A pass means an answer run never starts while a model server of the search is down. A failure means the run
    does not ask the app first: every question then waits for the missing server and is declined or answered from
    a weaker search, and the run ends as invalid, or as a result of another search, only after all that time.
    """
    app = _App([], ready="degraded")

    with pytest.raises(RunRefused) as refused:
        await _run(_qa(("QA-1", GOLD)), [], app=app)

    assert (refused.value.code, app.requests) == ("stack_not_ready", []), (
        f"a degraded stack gave {refused.value.code} after the requests {app.requests}"
    )


@pytest.mark.covers("US-019", "AC-9")
async def test_a_run_read_only_for_its_traces_starts_on_a_degraded_stack() -> None:
    """What it checks: one question is run against a fake app whose readiness says "degraded", with the run allowed
    to start on a degraded stack; the question is searched and asked.

    What it tells: a pass means the run made for its traces reaches the asks of a mutant that makes a model server of
    the search unreachable. A failure means such a mutant has no answer record and no stage named by its traces.

    Why it is worth having: the first attribution run (2026-11-20) recorded "no answer record" for the mutant whose
    embedding server is unreachable, because the answer run refused the degraded stack the mutant is meant to cause.
    """
    app = _App([], ready="degraded")

    await _run(_qa(("QA-1", GOLD)), [], app=app, allow_degraded=True)

    assert app.requests == [("/v1/search", 1), ("/v1/ask", 1)], f"requests sent on a degraded stack: {app.requests}"


@pytest.mark.covers("US-019", "AC-5")
@pytest.mark.parametrize(
    ("rerank", "expected_s"),
    [
        # 3 attempts of 30 s (embedding) and 60 s (model), 4 pauses of up to 120 s, 30 s of the app's own work.
        pytest.param(False, 3 * (30 + 60) + 4 * 120 + 30, id="search without rerank"),
        # The same, and two requests of 30 s to the rerank server for at most 50 passages in batches of 32.
        pytest.param(True, 3 * (30 + 60) + 4 * 120 + 30 + 2 * 30, id="search with rerank"),
    ],
)
def test_an_ask_waits_longer_than_the_app_may_take_with_every_attempt(rerank: bool, expected_s: float) -> None:
    """Computes the wait for one ask from the app's settings as GET /v1/config reports them, with the reranker off
    and on. Expects every attempt of the embedding call and of the model call with its timeout, the longest pause
    between attempts, the rerank requests when rerank is on, and the margin for the app's own work.

    What a pass tells: the harness waits for the app's own answer, a reply or its typed error after the last
    attempt, so a slow but normal answer is not recorded as "no response". A failure points to ask_timeout_s in
    answer_run.py. Worth having because a wait of 10 s against 60 s per model call turned two slow answers of a
    live experiment into failures of the stack and left a cell of the kill matrix unreadable.
    """
    settings: dict[str, JsonValue] = {**APP_SETTINGS, "search": {"rerank": rerank, "max_k": 50}}

    waited = ask_timeout_s(settings)

    assert waited == expected_s, f"an ask waits {waited} s; the app may take up to {expected_s} s before it answers"


@pytest.mark.covers("US-019", "AC-5")
async def test_every_ask_is_sent_with_the_wait_computed_from_the_apps_settings() -> None:
    """Runs one question against the fake app and reads the read timeout each request was sent with. Expects the
    ask to wait as long as ask_timeout_s gives for the app's settings, and the search to keep the usual wait.

    What a pass tells: the computed wait reaches the ask on the wire. A failure points to run in answer_run.py or
    to request in transport.py: the run would still give up on the app after the usual 10 s. Worth having because
    the computed number alone protects nothing if no request carries it.
    """
    waits: dict[str, float | None] = {}
    fake = _App([])

    def app(request: httpx2.Request) -> httpx2.Response:
        """Remember the read timeout of the search and the ask, then answer as the fake app does."""
        waits[request.url.path] = request.extensions["timeout"]["read"]
        return fake(request)

    settings = HarnessSettings(sut_base_url=HttpUrl("http://sut.test"))
    async with SutTransport(settings, transport=httpx2.MockTransport(app)) as sut:
        await run(
            _qa(("QA-1", GOLD)),
            sut,
            _Corpus({GOLD}),
            list,
            qa_file="evalharness_qa.toml",
            qa_bytes=b"evalharness",
            harness_commit="evalharness_commit",
            harness_dirty=False,
        )

    expected = {"/v1/search": settings.request_timeout_s, "/v1/ask": ask_timeout_s(APP_SETTINGS)}
    sent = {path: waits.get(path) for path in expected}
    assert sent == expected, f"requests were sent with the waits {sent}, expected {expected}"


# Trace ids the fake app names: the first trace is in the fake store, the second never arrives.
_STORED_TRACE = "8e472d7435ca66fac8ade84c55cc0b3b"
_LOST_TRACE = "0f472d7435ca66fac8ade84c55cc0b3c"


@pytest.mark.covers("US-019", "AC-9")
async def test_every_ask_keeps_the_stages_of_its_trace_or_why_they_could_not_be_read() -> None:
    """What it checks: three questions are asked of a fake app whose responses name a trace the fake store holds, a
    trace the store never gets, and no trace at all. The record keeps the stages of the first ask, and for the other
    two no stages and the reason: "trace_not_found" and "no_trace_id".

    What it tells: a pass means the answer records hold what the attribution of the mutants is built from, and an
    ask whose trace was lost says so. A failure means the stages are not written, or a lost trace looks like an ask
    whose every stage is absent.

    Why it is worth having: the trace store keeps traces in memory only, so the record is the only place the stages
    survive; a lost trace read as "every stage absent" would differ from the reference at the first stage and blame
    the embedding server for a fault of the store.
    """
    ledger: list[LedgerLine] = []
    app = _App(ledger)
    trace_of = {"Question QA-1?": _STORED_TRACE, "Question QA-2?": _LOST_TRACE}

    def answer(request: httpx2.Request) -> httpx2.Response:
        """Answer as the fake app does and name the trace of each ask in the standard response header."""
        response = app(request)
        question = json.loads(request.content).get("question") if request.url.path == "/v1/ask" else None
        if question in trace_of:
            response.headers["traceresponse"] = f"00-{trace_of[question]}-b1a1658393577b26-01"
        return response

    def store(request: httpx2.Request) -> httpx2.Response:
        """Hold one trace of an ask (the request span and a search that found one passage) and nothing else."""
        if not request.url.path.endswith(_STORED_TRACE):
            return httpx2.Response(404, json={"error": {"httpCode": 404}})
        spans = [
            {"spanId": "r1", "name": "POST /v1/ask", "startTimeUnixNano": "1", "endTimeUnixNano": "2"},
            {
                "spanId": "s1",
                "parentSpanId": "r1",
                "name": "search",
                "startTimeUnixNano": "1",
                "endTimeUnixNano": "2",
                "attributes": [
                    {"key": "search.status", "value": {"stringValue": "ok"}},
                    {"key": "search.hit_ids", "value": {"arrayValue": {"values": [{"stringValue": "evalharness_c1"}]}}},
                ],
            },
        ]
        return httpx2.Response(200, json={"result": {"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]}})

    settings = HarnessSettings(sut_base_url=HttpUrl("http://sut.test"))
    async with (
        SutTransport(settings, transport=httpx2.MockTransport(answer)) as sut,
        TraceReader("http://jaeger.invalid", wait_s=0.0, transport=httpx2.MockTransport(store)) as traces,
    ):
        record = await run(
            _qa(("QA-1", GOLD), ("QA-2", GOLD), ("QA-3", GOLD)),
            sut,
            _Corpus({GOLD}),
            lambda: ledger,
            qa_file="evalharness_qa.toml",
            qa_bytes=b"evalharness",
            harness_commit="evalharness_commit",
            harness_dirty=False,
            traces=traces,
        )

    seen = [(case.case_id, case.stages is not None, case.trace_error) for case in record.cases]
    assert seen == [
        ("QA-1", True, None),
        ("QA-2", False, "trace_not_found"),
        ("QA-3", False, "no_trace_id"),
    ], f"(question, stages kept, trace error) per ask: {seen}"
    assert record.cases[0].stages == {
        "embedding": "absent",
        "search": "ok hits=evalharness_c1",
        "rerank": "absent",
        "provider": "absent",
        "generate": "absent",
    }, f"the stages kept for the first ask: {record.cases[0].stages}"
