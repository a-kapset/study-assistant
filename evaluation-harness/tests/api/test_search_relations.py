"""API tests of two relations on course search, on the running stack: the same question is searched again with
another number of hits or another lowest score, and the two results must relate as stated.

The questions are not written by hand. Hypothesis draws them: a run of up to twelve words taken from a stored
passage of the stack's local course, so a question is close to some passages by meaning and shares words with
them, whatever course the stack serves. On a failure Hypothesis prints the smallest question, number of hits and
lowest scores it found that still break the relation.
"""

from typing import cast

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from pydantic import JsonValue

from evalharness.corpus_db import CorpusReader
from evalharness.evaluation.retrieval import SearchResult, search
from evalharness.relations.search_limits import higher_score_violation, more_hits_violation
from evalharness.settings import HarnessSettings, StackSettings, load_stack_settings
from evalharness.transport import SutTransport

# The app's own limit on the length of a question.
_LONGEST_QUESTION = 2000
# Every example sends real searches, so there is no time limit per example, and the number of examples keeps the
# two tests at a few seconds on the default search. The clients are shared by all examples of a test on purpose.
RELATION_SETTINGS = settings(
    max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)


class Passages(list[str]):
    """The stored passages the questions are drawn from.

    Hypothesis prints every argument of a failing test. This list prints its size only, so a failure report stays
    readable and holds no course text beyond the drawn question.
    """

    def __repr__(self) -> str:
        """Describe the list by its size."""
        return f"<{len(self)} stored passages>"


@pytest.fixture(scope="module")
def passages() -> Passages:
    """The stored passages of the stack's local course that hold at least one word, read once for the module."""
    harness = HarnessSettings()
    stack = load_stack_settings(harness.stack_env_file)
    with CorpusReader(harness, stack) as corpus:
        texts = [text for text in corpus.chunk_texts(stack.local_course_id) if text.split()]
    assert texts, f"the local course {stack.local_course_id!r} has no stored passages; ingest it first"
    return Passages(texts)


@pytest.fixture
async def most_hits(sut: SutTransport) -> int:
    """The largest number of hits the app lets a request ask for, from its settings."""
    config = await sut.request("GET", "/v1/config")
    app_settings = cast(dict[str, dict[str, JsonValue]], config.body_json)["settings"]
    return cast(int, cast(dict[str, JsonValue], app_settings["search"])["max_k"])


@st.composite
def questions(draw: st.DrawFn, passages: Passages) -> str:
    """Draw a question: one to twelve words in a row from one stored passage."""
    words = draw(st.sampled_from(passages)).split()
    start = draw(st.integers(0, len(words) - 1))
    length = draw(st.integers(1, 12))
    question = " ".join(words[start : start + length])
    assume(len(question) <= _LONGEST_QUESTION)
    return question


async def _searched(sut: SutTransport, course_id: str, question: str, k: int, min_score: float | None) -> SearchResult:
    """Search once and return the result; the test stops when the search failed or was answered on a weaker path,
    because the relations are stated for the configured search only."""
    extra: dict[str, JsonValue] = {"k": k} if min_score is None else {"k": k, "min_score": min_score}
    one = await search(sut, course_id, question, extra)
    assert one.result is not None, f"the search of {question!r} failed: {one.error}"
    assert one.result.degraded_reason is None, (
        f"the search was answered on a weaker path ({one.result.degraded_reason}); the stack is not healthy"
    )
    return one.result


def _cut_points(result: SearchResult) -> list[float]:
    """Return lowest scores that cut the result at every possible place, lowest first: -1 (keeps all), the middle
    between each two neighbouring scores, and the middle between the best score and 1 (keeps none).

    Middles are used, not the scores themselves, so that a score differing in its last digit between two searches
    cannot move a passage across the lowest score.
    """
    scores = sorted({hit.score for hit in result.hits if hit.score is not None})
    middles = [(low + high) / 2 for low, high in zip(scores, scores[1:], strict=False)]
    return [-1.0, *middles, (scores[-1] + 1) / 2]


@pytest.mark.covers("US-018", "AC-5")
@RELATION_SETTINGS
@given(data=st.data())
async def test_fewer_hits_asked_are_the_beginning_of_more_hits_asked(
    data: st.DataObject, sut: SutTransport, stack: StackSettings, passages: Passages, most_hits: int
) -> None:
    """What it checks: a question drawn from the course's passages is searched twice, asking for a smaller and
    a larger number of hits (up to the app's limit) with the same lowest score: the app's own, or none. The hits
    of the smaller request must be the first hits of the larger one, on whatever search the stack is configured for.

    What it tells: a pass means the number of hits only cuts one fixed order, so a learner who asks for more
    passages keeps the ones already seen, and recall can never fall when more hits are asked. A failure prints the
    question, the two numbers and both lists of passage ids: a passage that moved, appeared or was lost.

    Why it is worth having: with a hybrid search or a reranker the order depends on how many passages each ranking
    keeps before they are merged; an app that ties that number to the hits asked gives another order for every
    number, each of them looking normal on its own. The app's own tests check this on a handful of invented
    passages; only this test asks the real database, the real model servers and real course text, and it needs no
    labelled answers.
    """
    question = data.draw(questions(passages), label="question")
    fewer_k = data.draw(st.integers(1, most_hits - 1), label="fewer hits")
    more_k = data.draw(st.integers(fewer_k + 1, most_hits), label="more hits")
    min_score = data.draw(st.sampled_from([None, -1.0]), label="lowest score (None: the app's own)")

    fewer = await _searched(sut, stack.local_course_id, question, fewer_k, min_score)
    more = await _searched(sut, stack.local_course_id, question, more_k, min_score)

    violation = more_hits_violation(fewer, more)
    assert violation is None, f"asking for more hits changed the first ones for {question!r}: {violation}"


@pytest.mark.covers("US-018", "AC-4", "AC-13")
@RELATION_SETTINGS
@given(data=st.data())
async def test_a_higher_lowest_score_returns_no_passage_under_it_and_never_more_passages(
    data: st.DataObject, sut: SutTransport, stack: StackSettings, passages: Passages, most_hits: int
) -> None:
    """What it checks: a question drawn from the course's passages is searched without a lowest score, and two
    lowest scores are drawn that cut that result at different places. The question is then searched with each.
    Under the higher score every hit must reach it and there must not be more hits than under the lower one; when
    the stack searches by meaning alone, the hits must be exactly the hits of the lower score that reach the
    higher one, in the same order.

    What it tells: a pass means the lowest score works as a filter: raising it only takes passages away. A failure
    prints the question, the number of hits, both scores and the passage that is under the score, new, lost or
    moved. With a hybrid search or a reranker a new passage is allowed (the lowest score chooses what is ranked
    there), so for them the test judges only the scores and the number of hits.

    Why it is worth having: the lowest score is what keeps an off-topic question from getting passages, and each
    search path applies it in its own place: after the nearest passages are read, inside the hybrid statement,
    before the reranker. A path that skips it returns well-formed results with a few passages too many, which no
    status shows and a gold set only sees as a small change of precision.
    """
    question = data.draw(questions(passages), label="question")
    k = data.draw(st.integers(1, most_hits), label="hits")
    unfiltered = await _searched(sut, stack.local_course_id, question, k, -1.0)
    assume(unfiltered.hits)
    cuts = _cut_points(unfiltered)
    lower_at = data.draw(st.integers(0, len(cuts) - 2), label="place of the lower score")
    higher_at = data.draw(st.integers(lower_at + 1, len(cuts) - 1), label="place of the higher score")

    lower = await _searched(sut, stack.local_course_id, question, k, cuts[lower_at])
    higher = await _searched(sut, stack.local_course_id, question, k, cuts[higher_at])

    violation = higher_score_violation(lower, higher)
    assert violation is None, f"raising the lowest score did more than remove passages for {question!r}: {violation}"
