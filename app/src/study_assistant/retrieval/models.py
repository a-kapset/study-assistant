"""What a search returns: the hits, and a status that says whether anything was found and how.

An empty list alone cannot tell a client whether nothing was relevant or whether the search did not run, so every
result names its status, and a search that cannot run is an error, never an empty result. A search that ran on a
weaker path because a server it uses was down says so, so its hits are never taken for the usual ones.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

# "ok": at least one passage reached the lowest score. "empty": the search ran over the whole course and no passage
# reached it, so a caller should treat the question as not covered by the course. "degraded": a server the search
# uses could not be reached and the search ran without it; the result names the reason.
type SearchStatus = Literal["ok", "empty", "degraded"]

# Why a search is degraded. "embedder_unavailable": the question could not be turned into a vector, so the passages
# are ranked by the question's words only, with no lowest score. "reranker_unavailable": the passages are those of
# the strategy, in the strategy's order.
type DegradedReason = Literal["embedder_unavailable", "reranker_unavailable"]

# How passages are ranked: "vector" by meaning only; "hybrid" by meaning and by the question's words together.
type SearchStrategy = Literal["vector", "hybrid"]


class SearchHit(BaseModel):
    """One passage of the course, how close it is to the question, and where it comes from."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    chunk_id: str
    # The section the passage belongs to; None for text before the first heading of a file.
    section_id: str | None
    source_file: str
    page_start: int
    page_end: int
    text: str
    # Cosine similarity between the question and the passage, from -1 to 1; higher is closer. Under the hybrid
    # strategy and with the reranker too, so every hit can be checked against the lowest score; the order of the
    # hits may then differ from it. None when the question has no vector (a search degraded by the embedder).
    score: float | None
    # Position in the result, starting at 1 for the best passage.
    rank: int


class SearchResult(BaseModel):
    """The hits of one search, best first, with the values the search really used.

    The strategy, whether the reranker ran, the number of hits asked for, the lowest score and the embedding space
    are returned with the hits, so a client can record exactly what produced them and never compares results made
    under different settings.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    status: SearchStatus
    # Set exactly when the status is "degraded": which server was down, so a caller can tell how far to trust the
    # hits (an answer needs the vector; a missing reranker leaves the strategy's own order).
    degraded_reason: DegradedReason | None
    course_id: str
    # The ranking that produced the hits, so results of the two strategies are never mistaken for each other. It
    # names the configured strategy also when the embedder was down and the hits were ranked by words only.
    strategy: SearchStrategy
    # Whether the reranker put the hits in their order. The hits are then the strategy's passages in the reranker's
    # order, and results with and without it are never mistaken for each other.
    reranked: bool
    embedding_space: str
    k: int
    # Not applied when the embedder was down: passages ranked by words have no similarity to compare with it.
    min_score: float
    hits: list[SearchHit]
