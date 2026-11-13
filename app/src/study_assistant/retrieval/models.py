"""What a search returns: the hits, and a status that says whether anything was found.

An empty list alone cannot tell a client whether nothing was relevant or whether the search did not run, so every
result names its status, and a search that cannot run is an error, never an empty result.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

# "ok": at least one passage reached the lowest score. "empty": the search ran over the whole course and no passage
# reached it, so a caller should treat the question as not covered by the course.
type SearchStatus = Literal["ok", "empty"]

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
    # hits may then differ from it.
    score: float
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
    course_id: str
    # The ranking that produced the hits, so results of the two strategies are never mistaken for each other.
    strategy: SearchStrategy
    # Whether the reranker put the hits in their order. The hits are then the strategy's passages in the reranker's
    # order, and results with and without it are never mistaken for each other.
    reranked: bool
    embedding_space: str
    k: int
    min_score: float
    hits: list[SearchHit]
