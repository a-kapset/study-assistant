"""Unit tests of a paraphrase relation run: a small invented pair set, a fake corpus and a fake app, never the
running stack."""

import json
from collections.abc import Callable

import httpx2
import pytest
from pydantic import HttpUrl

from evalharness.datasets.paraphrase_set import ParaphrasePair, ParaphraseSet, RelationRules
from evalharness.evaluation.paraphrase_record import ParaphraseRunRecord
from evalharness.evaluation.paraphrase_run import run
from evalharness.evaluation.retrieval import RunRefused
from evalharness.settings import HarnessSettings
from evalharness.transport import SutTransport

COURSE = "evalharness_course"

type Handler = Callable[[httpx2.Request], httpx2.Response]
RULES = RelationRules(min_meaningful_change=0.05, min_overlap=0.6, stable_share_min=0.7)


class _Corpus:
    """A stored corpus with the sections the pairs need, in place of the database reader."""

    def section_ids(self, course_id: str) -> set[str]:
        """Return the sections of the fake course."""
        return {"1.1", "2.1"}

    def snapshot(self, course_id: str) -> str:
        """Return a fixed corpus name."""
        return "evalharness_snapshot"

    def chunk_texts(self, course_id: str) -> list[str]:
        """Return no chunk texts; a paraphrase run does not read them."""
        return []


def _pair(case_id: str, term_removed: bool) -> ParaphrasePair:
    """Build a pair whose question and rewording name the case, answered by section 1.1."""
    return ParaphrasePair(
        case_id=case_id,
        question=f"question {case_id}",
        paraphrase=f"rewording {case_id}",
        gold_section="1.1",
        term_removed=term_removed,
    )


def _app(request: httpx2.Request) -> httpx2.Response:
    """Answer a search as an app whose lowest score cuts off every passage of a rewording.

    A question finds five passages, the first from the gold section 1.1. A rewording finds nothing with the app's
    own lowest score; asked with no lowest score it finds three of the question's passages and two others. The
    settings request gets a minimal settings object, the readiness request the status "ok".
    """
    if request.url.path == "/v1/config":
        return httpx2.Response(200, json={"app_version": "0.1.0", "settings": {"git_commit": "evalharness_app"}})
    if request.url.path == "/ready":
        return httpx2.Response(200, json={"status": "ok", "checks": {"db": "ok", "embedder": "ok"}})
    body = json.loads(request.content)
    if body["query"].startswith("question"):
        chunks = [("c1", "1.1"), ("c2", "2.1"), ("c3", "2.1"), ("c4", "2.1"), ("c5", "2.1")]
    elif "min_score" in body:
        chunks = [("c1", "1.1"), ("c2", "2.1"), ("c3", "2.1"), ("x4", "2.1"), ("x5", "2.1")]
    else:
        chunks = []
    hits = [
        {"chunk_id": c, "section_id": s, "source_file": "f", "page_start": 1, "page_end": 1, "text": "t"}
        | {"score": 0.9, "rank": i}
        for i, (c, s) in enumerate(chunks, start=1)
    ]
    return httpx2.Response(
        200,
        json={
            "status": "ok" if hits else "empty",
            "course_id": COURSE,
            "strategy": "vector",
            "reranked": False,
            "degraded_reason": None,
            "embedding_space": "evalharness_space",
            "k": 5,
            "min_score": body.get("min_score", 0.83),
            "hits": hits,
        },
    )


async def _run(pairs: tuple[ParaphrasePair, ...], app: Handler = _app) -> ParaphraseRunRecord:
    """Run the pairs against the fake app ``app`` and the fake corpus."""
    settings = HarnessSettings(sut_base_url=HttpUrl("http://sut.test"))
    pair_set = ParaphraseSet(course_id=COURSE, rules=RULES, pairs=pairs, sha256="0" * 64)
    async with SutTransport(settings, transport=httpx2.MockTransport(app)) as sut:
        return await run(
            pair_set,
            sut,
            _Corpus(),
            pair_file="evalharness_pairs.toml",
            harness_commit="evalharness_commit",
            harness_dirty=False,
        )


@pytest.mark.covers("US-015", "AC-3", "AC-4")
async def test_the_diagnostic_pass_searches_without_a_lowest_score_and_tells_its_losses_apart() -> None:
    """Runs one pair against a fake app whose lowest score cuts off every passage of the rewording.

    A pass means the primary pass counts the pair as "lost" (what a learner gets) while the diagnostic pass, sent
    with no lowest score, counts it as "kept" and stable (3 of 5 passages shared). A failure means the diagnostic
    pass is sent with the app's lowest score, so the run can no longer say whether a loss comes from our threshold
    or from the embedding model, and the top-k stable relation is judged on lists the threshold cut short.
    """
    record = await _run((_pair("QA-1", term_removed=True),))

    primary = record.summaries["primary"]["all"]
    diagnostic = record.summaries["diagnostic"]["all"]

    assert (primary.lost, diagnostic.kept, diagnostic.stable) == (1, 1, 1), (
        f"primary {primary}, diagnostic {diagnostic}"
    )


@pytest.mark.covers("US-015", "AC-8")
async def test_a_paraphrase_run_on_a_degraded_stack_is_refused_before_the_first_search() -> None:
    """Answers GET /ready with 200 and the status "degraded" (the rerank server is down); the searches themselves
    would succeed. Expects the refusal "stack_not_ready" and no search sent.

    A pass means the relation is never judged on a stack that answers on a weaker path than the configured one. A
    failure means the paraphrase run starts without asking the app: its rewordings are then measured without the
    reranker and the result is filed under the rerank variant. The retrieval run has the same check; this test
    shows that the paraphrase run, a separate command, makes it too.
    """
    searches: list[str] = []

    def app(request: httpx2.Request) -> httpx2.Response:
        """Report a degraded stack at /ready; remember every search and answer it as the usual fake app."""
        if request.url.path == "/ready":
            checks = {"db": "ok", "embedder": "ok", "reranker": "unavailable"}
            return httpx2.Response(200, json={"status": "degraded", "checks": checks})
        if request.url.path == "/v1/search":
            searches.append(request.url.path)
        return _app(request)

    with pytest.raises(RunRefused) as refused:
        await _run((_pair("QA-1", term_removed=True),), app)

    assert refused.value.code == "stack_not_ready", f"a degraded stack gave {refused.value.code}"
    assert searches == [], f"the run sent {len(searches)} searches before it was refused"


@pytest.mark.covers("US-015", "AC-5")
async def test_pairs_are_also_reported_by_whether_their_question_lost_a_key_term() -> None:
    """Runs two pairs whose question held a key term and one whose question did not.

    A pass means the record has a slice of two pairs for "key term removed" and one of one pair for "no key term in
    question", next to "all". A failure means the slices are missing or mixed, so the run cannot show whether
    losses come from removing the course term or from rewording in general.
    """
    record = await _run((_pair("QA-1", True), _pair("QA-2", True), _pair("QA-3", False)))

    sizes = {key: summary.pairs for key, summary in record.summaries["primary"].items()}

    assert sizes == {"all": 3, "key term removed": 2, "no key term in question": 1}, f"slices {sizes}"
