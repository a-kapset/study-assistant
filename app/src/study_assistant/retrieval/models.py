"""What a search returns: the hits, and a status that says whether anything was found.

An empty list alone cannot tell a client whether nothing was relevant or whether the search did not run, so every
result names its status, and a search that cannot run is an error, never an empty result.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

# "ok": at least one passage reached the lowest score. "empty": the search ran over the whole course and no passage
# reached it, so a caller should treat the question as not covered by the course.
type SearchStatus = Literal["ok", "empty"]


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
    # Cosine similarity between the question and the passage, from -1 to 1; higher is closer.
    score: float
    # Position in the result, starting at 1 for the closest passage.
    rank: int


class SearchResult(BaseModel):
    """The hits of one search, closest first, with the values the search really used.

    The number of hits asked for, the lowest score and the embedding space are returned with the hits, so a
    client can record exactly what produced them and never compares results made under different settings.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    status: SearchStatus
    course_id: str
    embedding_space: str
    k: int
    min_score: float
    hits: list[SearchHit]
