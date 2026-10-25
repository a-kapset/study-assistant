"""Unit tests of a retrieval run: a small invented gold set, a fake corpus and a fake app, never the running stack."""

import json
from collections.abc import Callable
from pathlib import Path

import httpx2
import pytest
from pydantic import HttpUrl, JsonValue

from evalharness.datasets.retrieval_gold import GoldCase, GoldDeclaration, GoldSet
from evalharness.evaluation.__main__ import _print_summary
from evalharness.evaluation.record import RunRecord, recorded_held_out_uses
from evalharness.evaluation.retrieval import RunRefused, run, select_cases, summaries
from evalharness.oracles.retrieval import score_case
from evalharness.settings import HarnessSettings
from evalharness.transport import SutTransport

COURSE = "evalharness_course"

type Handler = Callable[[httpx2.Request], httpx2.Response]


def _case(case_id: str, gold_section: str, knowledge_level: str = "K2") -> GoldCase:
    """Build one gold case; the exam set is the part of the id before the dash, the chapter the section's first part."""
    exam_set = case_id.split("-", 1)[0]
    return GoldCase(
        case_id=case_id,
        query=f"Question {case_id}?",
        gold_section=gold_section,
        exam_set=exam_set,
        chapter=gold_section.split(".", 1)[0],
        knowledge_level=knowledge_level,
        origin="sample_exam",
        held_out=exam_set == "D",
    )


def _gold(*cases: GoldCase) -> GoldSet:
    """Build a gold set of ``cases`` under a declaration with set D held out and recall at 5 on set A as main result."""
    declaration = GoldDeclaration.model_validate(
        {
            "course_id": COURSE,
            "bank_file": "bank.jsonl",
            "bank_sha256": "0" * 64,
            "objective_section": r"FL-(?P<section>(?P<chapter>\d+)\.\d+)\.\d+",
            "held_out_sets": ["D"],
            "primary": {"metric": "recall", "k": 5, "sets": ["A"], "target": 0.8},
        }
    )
    return GoldSet(declaration=declaration, cases=cases, sha256="1" * 64)


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


def _found(request: httpx2.Request, k: int = 5) -> httpx2.Response:
    """Answer a search as the app does when it finds one passage of section 1.1; reports the k asked for, else ``k``."""
    body = json.loads(request.content)
    hit = {
        "chunk_id": "c1",
        "section_id": "1.1",
        "source_file": "f",
        "page_start": 1,
        "page_end": 1,
        "text": "t",
        "score": 0.9,
        "rank": 1,
    }
    return httpx2.Response(
        200,
        json={
            "status": "ok",
            "course_id": COURSE,
            "embedding_space": "evalharness_space",
            "k": body.get("k", k),
            "min_score": body.get("min_score", 0.83),
            "hits": [hit],
        },
    )


async def _run(gold: GoldSet, handler: Handler, sections: set[str] | None = None) -> RunRecord:
    """Run every case of ``gold`` against a fake app answering with ``handler``; the corpus holds the gold sections."""
    corpus = _Corpus(sections if sections is not None else {case.gold_section for case in gold.cases})
    settings = HarnessSettings(sut_base_url=HttpUrl("http://sut.test"))
    async with SutTransport(settings, transport=httpx2.MockTransport(handler)) as sut:
        return await run(
            gold,
            gold.cases,
            sut,
            corpus,
            declaration_file="evalharness_gold.toml",
            held_out_use=None,
            harness_commit="evalharness_commit",
            harness_dirty=False,
        )


@pytest.mark.covers("US-007", "AC-5")
def test_a_held_out_set_is_refused_without_permission() -> None:
    """Asks for the held-out set D without permission and expects the refusal "held_out_not_allowed".

    A pass means held-out questions cannot enter a run by a slip in the list of sets. A failure means set D can be
    run, and looked at, without anyone deciding to, so its number slowly stops being a fair estimate for new
    questions. Nothing else stops this: the run itself works the same on any set.
    """
    gold = _gold(_case("A-1", "1.1"), _case("D-1", "1.2"))

    with pytest.raises(RunRefused) as refused:
        select_cases(gold, ["A", "D"], include_held_out=False)

    assert refused.value.code == "held_out_not_allowed", f"set D without permission gave {refused.value.code}"


def test_an_unknown_set_is_refused() -> None:
    """Asks for sets A and X, where X is not in the gold set, and expects the refusal "unknown_set".

    A pass means a mistyped set name stops the run. A failure means the run quietly measures only the known sets,
    and the printed number looks like a result on all the sets that were asked for.
    """
    gold = _gold(_case("A-1", "1.1"), _case("B-1", "1.2"))

    with pytest.raises(RunRefused) as refused:
        select_cases(gold, ["A", "X"], include_held_out=False)

    assert refused.value.code == "unknown_set", f"an unknown set gave {refused.value.code}"


@pytest.mark.covers("US-007", "AC-7")
async def test_an_answer_section_missing_from_the_corpus_refuses_the_run() -> None:
    """Runs a gold set whose answer section 2.1 is not a section of the stored course; expects "gold_section_missing".

    A pass means a run never scores questions that no search could answer. A failure means such questions become
    misses in every run: after a change of the course's section numbering, retrieval would look worse while the
    real cause is the gold set no longer matching the corpus.
    """
    gold = _gold(_case("A-1", "1.1"), _case("A-2", "2.1"))

    with pytest.raises(RunRefused) as refused:
        await _run(gold, _found, sections={"1.1"})

    assert refused.value.code == "gold_section_missing", f"a missing answer section gave {refused.value.code}"


@pytest.mark.covers("US-007", "AC-9")
async def test_the_primary_pass_sends_the_question_as_a_user_does() -> None:
    """Runs two questions and reads the requests of the primary pass: only the course and the unchanged question.

    A pass means the main result is what a user gets with the app's own number of hits and lowest score. A failure
    means the harness sends its own values or changes the question, so the number measures a search no user makes,
    and a changed setting of the app would no longer show in it.
    """
    gold = _gold(_case("A-1", "1.1"), _case("A-2", "1.2"))
    bodies: list[JsonValue] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        """Remember the request body and answer with one found passage."""
        bodies.append(json.loads(request.content))
        return _found(request)

    await _run(gold, handler)

    expected = [{"course_id": COURSE, "query": case.query} for case in gold.cases]
    assert bodies[: len(gold.cases)] == expected, f"the primary pass sent {bodies[: len(gold.cases)]}"


@pytest.mark.covers("US-007", "AC-9")
async def test_the_diagnostic_pass_asks_for_the_apps_k_with_no_lowest_score() -> None:
    """The app uses 7 hits; the first primary search fails. Expects every diagnostic request with k 7, min_score -1.

    A pass means both passes are scored over the same places, so the difference between them is caused only by the
    lowest score. A failure means the diagnostic pass asks for another number of hits (for example the declared 5,
    or the default after a failed first question) or keeps the lowest score, and the two passes cannot be compared.
    """
    gold = _gold(_case("A-1", "1.1"), _case("A-2", "1.2"), _case("A-3", "1.3"))
    bodies: list[dict[str, JsonValue]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        """Remember the body; fail the first search, answer the others as an app configured for 7 hits."""
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            return httpx2.Response(503, json={"status": "error", "error": {"code": "embedding_unavailable"}})
        return _found(request, k=7)

    await _run(gold, handler)

    diagnostic = [(body.get("k"), body.get("min_score")) for body in bodies[len(gold.cases) :]]
    assert diagnostic == [(7, -1)] * len(gold.cases), f"the diagnostic pass asked for (k, min_score) {diagnostic}"


def _connect_error(request: httpx2.Request) -> httpx2.Response:
    """Fail the request as an app that cannot be reached."""
    raise httpx2.ConnectError("connection refused", request=request)


@pytest.mark.covers("US-007", "AC-4")
@pytest.mark.parametrize(
    ("handler", "reason"),
    [
        (
            lambda request: httpx2.Response(503, json={"status": "error", "error": {"code": "embedding_unavailable"}}),
            "embedding_unavailable",
        ),
        (lambda request: httpx2.Response(502, text="Bad Gateway"), "http_502"),
        (_connect_error, "connect"),
    ],
    ids=["app error", "other error", "no response"],
)
async def test_a_search_without_a_result_counts_as_failed_with_its_reason(handler: Handler, reason: str) -> None:
    """Answers the search with an error of the app, an error in another format, or no response at all.

    Expects the verdict "failed" with the reason named. A pass means a broken stack is reported as broken and says
    where to look. A failure means it is counted as "empty", which points to the lowest score instead of the stack,
    or the reason is lost.
    """
    gold = _gold(_case("A-1", "1.1"))

    record = await _run(gold, handler)

    result = record.cases[0].passes["primary"]
    assert (result.score.verdict, result.error) == ("failed", reason), (
        f"a search without a result was recorded as {result.score.verdict} with reason {result.error}"
    )


@pytest.mark.covers("US-007", "AC-8")
def test_every_question_is_counted_in_all_its_slices() -> None:
    """Summarises three questions from two sets, two chapters and two knowledge levels; checks n of every slice.

    A pass means each question is counted in "all", in its exam set, its chapter and its knowledge level, and
    nowhere else. A failure means the numbers per slice, which decide where retrieval is improved next, describe
    other questions than their names say.
    """
    cases = [_case("A-1", "1.1", "K1"), _case("A-2", "2.1"), _case("B-1", "1.2")]
    scores = [score_case(case.gold_section, [case.gold_section], 5) for case in cases]

    counts = {key: summary.n for key, summary in summaries(cases, scores).items()}

    expected = {"all": 3, "set A": 2, "set B": 1, "chapter 1": 2, "chapter 2": 1, "level K1": 1, "level K2": 2}
    assert counts == expected, f"questions per slice were {counts}"


@pytest.mark.covers("US-007", "AC-5")
def test_held_out_uses_count_only_the_courses_runs_with_held_out_sets(tmp_path: Path) -> None:
    """Writes three records: this course with set D, this course without it, another course with it. Expects 1.

    A pass means the printed use number of the held-out sets grows only with runs that looked at them. A failure
    means runs of another course (the synthetic course in the same folder) or runs without held-out sets make the
    held-out sets look more used than they are, or less.
    """
    for name, course_id, held_out_use in [("a", COURSE, 2), ("b", COURSE, None), ("c", "evalharness_other", 2)]:
        (tmp_path / f"{name}.json").write_text(json.dumps({"course_id": course_id, "held_out_use": held_out_use}))

    uses = recorded_held_out_uses(tmp_path, COURSE)

    assert uses == 1, f"counted {uses} uses of the held-out sets, expected 1"


@pytest.mark.covers("US-007", "AC-10")
@pytest.mark.parametrize(("app_k", "verdict_printed"), [(5, True), (10, False)])
async def test_the_target_verdict_is_printed_only_for_the_declared_k(
    app_k: int, verdict_printed: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    """Runs the declared set A against an app using 5 hits, then 10 hits; the target is declared for 5.

    Expects the line "recall@k ... target ...: met/below" only with 5. A pass means a result over another number of
    hits is shown without being compared to the target. A failure means recall at 10, which is easier to reach, is
    reported as meeting the target set for recall at 5.
    """
    gold = _gold(_case("A-1", "1.1"))
    record = await _run(gold, lambda request: _found(request, k=app_k))

    _print_summary(record, gold, Path("evalharness_run.json"))

    printed = any(line.startswith("recall@") for line in capsys.readouterr().out.splitlines())
    assert printed == verdict_printed, f"with k={app_k} the target verdict printed: {printed}"
