"""Unit tests of the two relations on a search asked twice with a changed limit, on results written by hand."""

import pytest

from evalharness.evaluation.retrieval import SearchResult
from evalharness.relations.search_limits import higher_score_violation, more_hits_violation


def _result(
    hits: list[tuple[str, float | None]], *, k: int, min_score: float, strategy: str = "vector", reranked: bool = False
) -> SearchResult:
    """Build a search result as the app returns it: ``hits`` are (passage id, score) in the app's order."""
    return SearchResult.model_validate(
        {
            "embedding_space": "evalharness_space",
            "k": k,
            "min_score": min_score,
            "strategy": strategy,
            "reranked": reranked,
            "hits": [{"chunk_id": chunk_id, "section_id": "1.1", "score": score} for chunk_id, score in hits],
        }
    )


@pytest.mark.parametrize(
    ("fewer", "more", "holds"),
    [
        ((2, ["a", "b"]), (4, ["a", "b", "c", "d"]), True),
        # Only two passages reach the lowest score, so both requests return them.
        ((3, ["a", "b"]), (5, ["a", "b"]), True),
        ((2, ["b", "a"]), (4, ["a", "b", "c", "d"]), False),
        ((2, ["a", "x"]), (4, ["a", "b", "c", "d"]), False),
        ((2, ["a", "b", "c"]), (4, ["a", "b", "c", "d"]), False),
        ((3, ["a", "b"]), (5, ["a", "b", "c"]), False),
    ],
    ids=["beginning", "both-short", "other-order", "passage-not-in-larger", "more-than-asked", "passage-lost"],
)
def test_fewer_hits_asked_must_be_the_beginning_of_more_hits_asked(
    fewer: tuple[int, list[str]], more: tuple[int, list[str]], holds: bool
) -> None:
    """Judges a smaller and a larger request of one question, each given as (hits asked, passage ids returned),
    and expects no violation exactly when the smaller result is the first part of the larger one.

    Catches a relation that compares the passages as sets (a changed order would pass), that does not cut the
    larger result to the number the smaller request asked for (a request answered with more hits than asked would
    pass), or that accepts any shorter list (a passage lost by the smaller request would pass).
    """
    small = _result([(chunk_id, 0.9) for chunk_id in fewer[1]], k=fewer[0], min_score=0.83)
    large = _result([(chunk_id, 0.9) for chunk_id in more[1]], k=more[0], min_score=0.83)

    violation = more_hits_violation(small, large)

    assert (violation is None) == holds, f"expected the relation to {'hold' if holds else 'fail'}, got: {violation}"


@pytest.mark.parametrize(
    ("strategy", "reranked", "score"),
    [("vector", False, 0.84), ("hybrid", False, 0.84), ("vector", True, 0.84), ("hybrid", False, None)],
    ids=["by-meaning", "with-words", "reranked", "no-score"],
)
def test_a_hit_under_the_higher_lowest_score_is_a_violation_on_every_search_path(
    strategy: str, reranked: bool, score: float | None
) -> None:
    """Judges a result asked with the lowest score 0.85 that holds a passage scored 0.84, or one without a score,
    under each search path, and expects a violation that names the passage.

    Catches a relation that checks the lowest score only for the search by meaning: a hybrid search that lets a
    passage in through its ranking by words although it is under the lowest score is the defect the weaker form of
    the relation is kept for.
    """
    lower = _result([("a", 0.9), ("b", 0.84)], k=5, min_score=0.8, strategy=strategy, reranked=reranked)
    higher = _result([("a", 0.9), ("b", score)], k=5, min_score=0.85, strategy=strategy, reranked=reranked)

    violation = higher_score_violation(lower, higher)

    assert violation is not None and "'b'" in violation, f"the hit under the lowest score was not named: {violation}"


def test_more_hits_under_a_higher_lowest_score_is_a_violation_on_a_search_with_words() -> None:
    """Judges a hybrid search that returned two passages under the lowest score 0.8 and three under 0.85, all
    reaching it, and expects a violation.

    Catches a relation that judges nothing but the scores for a search with words: fewer passages are ranked under
    a higher lowest score, so a longer result means the lower request lost passages it should have returned.
    """
    lower = _result([("a", 0.9), ("b", 0.88)], k=5, min_score=0.8, strategy="hybrid")
    higher = _result([("a", 0.9), ("b", 0.88), ("c", 0.86)], k=5, min_score=0.85, strategy="hybrid")

    violation = higher_score_violation(lower, higher)

    assert violation is not None and "more than" in violation, f"the longer result was not reported: {violation}"


@pytest.mark.parametrize(
    ("higher_hits", "holds"),
    [
        ([("a", 0.9), ("b", 0.87)], True),
        ([("a", 0.9), ("x", 0.86)], False),
        ([("a", 0.9)], False),
        ([("b", 0.87), ("a", 0.9)], False),
    ],
    ids=["exactly-those-that-reach-it", "new-passage", "passage-lost", "order-changed"],
)
def test_by_meaning_alone_a_higher_lowest_score_keeps_exactly_the_hits_that_reach_it(
    higher_hits: list[tuple[str, float | None]], holds: bool
) -> None:
    """Judges a search by meaning that returned passages scored 0.9, 0.87 and 0.84 under the lowest score 0.8,
    against its result under 0.85, and expects no violation only when that result is the first two passages in
    the same order.

    Catches a relation that, for the search by meaning, checks only the scores and the number of hits: a new
    passage, a lost one or a changed order all keep every score above the lowest score and the result short enough.
    """
    lower = _result([("a", 0.9), ("b", 0.87), ("c", 0.84)], k=5, min_score=0.8)
    higher = _result(higher_hits, k=5, min_score=0.85)

    violation = higher_score_violation(lower, higher)

    assert (violation is None) == holds, f"expected the relation to {'hold' if holds else 'fail'}, got: {violation}"


@pytest.mark.parametrize(
    ("strategy", "reranked"), [("hybrid", False), ("vector", True)], ids=["with-words", "reranked"]
)
def test_a_new_passage_under_a_higher_lowest_score_is_allowed_with_words_or_a_reranker(
    strategy: str, reranked: bool
) -> None:
    """Judges a search with words, and one with a reranker, whose result under the higher lowest score holds a
    passage that was not in the result under the lower one, and expects no violation.

    Catches a relation that applies the strong form to every search path: in these searches the lowest score
    chooses which passages are ranked, so a passage from outside the first hits can enter them when a passage
    before it is left out, and a correct app would fail the check.
    """
    lower = _result([("a", 0.9), ("low", 0.82)], k=2, min_score=0.8, strategy=strategy, reranked=reranked)
    higher = _result([("a", 0.9), ("b", 0.86)], k=2, min_score=0.85, strategy=strategy, reranked=reranked)

    violation = higher_score_violation(lower, higher)

    assert violation is None, f"a new passage is how this search works, yet it was reported: {violation}"
