"""Two relations on one question searched twice with a changed limit: more hits asked, and a higher lowest score.

Neither needs to know the right answer. They state how the two results of the same question must relate when only
the number of hits or the lowest score changes, so they hold for any course, model and question:

- more hits asked: the hits of the smaller request are the beginning of the hits of the larger one;
- higher lowest score: every hit reaches it and there are never more hits than before; when the search ranks by
  meaning alone, the hits are exactly the earlier hits that reach it, in the same order.

The second relation has a weaker form for a search that ranks by meaning and by words together, or lets a reranker
order the passages. There the lowest score chooses the passages that are ranked at all, so a higher score changes
the places of the passages that stay, and one that was outside the first hits can enter them. That is how those
searches are built, not a fault, so only what must hold on every path is judged for them.

Each function returns None when the relation holds and a sentence naming the first thing that breaks it otherwise.
"""

from evalharness.evaluation.retrieval import SearchResult

# The strategy name the app reports for a search that ranks by meaning alone.
_BY_MEANING = "vector"


def more_hits_violation(fewer: SearchResult, more: SearchResult) -> str | None:
    """Judge two results of one question and lowest score, where ``fewer`` asked for fewer hits than ``more``.

    The hits of ``fewer`` must be the first hits of ``more``, as many as ``fewer`` asked for. This also means that
    a request never gets more hits than it asked for and that asking for more never loses a passage.
    """
    got = _ids(fewer)
    expected = _ids(more)[: fewer.k]
    if got == expected:
        return None
    return (
        f"the {fewer.k} hit(s) asked first are not the beginning of the {more.k} asked next: "
        f"got {got}, the larger request begins with {expected}"
    )


def higher_score_violation(lower: SearchResult, higher: SearchResult) -> str | None:
    """Judge two results of one question and number of hits, where ``higher`` asked for a higher lowest score.

    On every search path each hit of ``higher`` must reach its lowest score, and ``higher`` must not have more hits
    than ``lower``. When both results were ranked by meaning alone, ``higher`` must hold exactly the hits of
    ``lower`` that reach the higher score, in the same order: nothing new, nothing lost, nothing moved.
    """
    under = [hit.chunk_id for hit in higher.hits if hit.score is None or hit.score < higher.min_score]
    if under:
        return f"{len(under)} hit(s) do not reach the lowest score {higher.min_score}: {under}"
    if len(higher.hits) > len(lower.hits):
        return (
            f"the lowest score {higher.min_score} gave {len(higher.hits)} hit(s), "
            f"more than the {len(lower.hits)} under {lower.min_score}"
        )
    if ranked_by_meaning_alone(lower) and ranked_by_meaning_alone(higher):
        expected = [hit.chunk_id for hit in lower.hits if hit.score is not None and hit.score >= higher.min_score]
        if _ids(higher) != expected:
            return (
                f"the hits under the lowest score {higher.min_score} are not the hits under {lower.min_score} "
                f"that reach it: got {_ids(higher)}, expected {expected}"
            )
    return None


def ranked_by_meaning_alone(result: SearchResult) -> bool:
    """Say whether the app reports that it ranked this result by meaning only, without words and without a
    reranker: the search path on which the strong form of the lowest-score relation holds."""
    return result.strategy == _BY_MEANING and not result.reranked


def _ids(result: SearchResult) -> list[str]:
    """Return the passage ids of a result in the order the app returned them."""
    return [hit.chunk_id for hit in result.hits]
