"""Vector search within one course: the question's vector, the closest passages, and the lowest score they need."""

from study_assistant.embed.base import Embedder
from study_assistant.retrieval.models import SearchHit, SearchResult
from study_assistant.retrieval.repository import SearchStore


class CourseNotSearchableError(Exception):
    """The course has no chunks, or some of its chunks have no vector in the configured embedding space.

    Searching such a course would quietly look at part of it, or at nothing, so the search is refused instead.
    Ingesting the course again computes the missing vectors.
    """

    def __init__(self, course_id: str, chunks: int, vectors: int) -> None:
        """Keep the counts; the message says what is missing and how to fix it."""
        if chunks == 0:
            reason = "it has no stored chunks; ingest it first"
        else:
            reason = f"only {vectors} of its {chunks} chunks have a vector in the configured space; ingest it again"
        super().__init__(f"Course '{course_id}' cannot be searched: {reason}")
        self.course_id = course_id
        self.chunks = chunks
        self.vectors = vectors


class VectorSearch:
    """Find the passages of a course that are closest to a question, in the embedder's space."""

    def __init__(self, store: SearchStore, embedder: Embedder) -> None:
        """Keep the store and the embedder; the space searched is always the one the embedder computes."""
        self._store = store
        self._embedder = embedder

    async def search(self, course_id: str, query: str, k: int, min_score: float) -> SearchResult:
        """Return up to ``k`` passages of the course scoring at least ``min_score``, closest first.

        The course is checked before the question is embedded: a course that is not fully in the embedder's
        space raises ``CourseNotSearchableError``, so a result always covers the whole course. The result is
        "empty" when no passage reaches ``min_score``. Errors of the embedding server pass through unchanged.
        """
        space = self._embedder.space
        coverage = await self._store.coverage(course_id, space)
        if coverage.chunks == 0 or coverage.vectors < coverage.chunks:
            raise CourseNotSearchableError(course_id, coverage.chunks, coverage.vectors)
        vector = await self._embedder.aembed_query(query)
        passages = await self._store.nearest(course_id, space, vector, k)
        # The passages come closest first, so keeping those that reach the score keeps the order and the ranks.
        hits = [
            SearchHit(
                chunk_id=p.chunk_id,
                section_id=p.section_id,
                source_file=p.source_file,
                page_start=p.page_start,
                page_end=p.page_end,
                text=p.text,
                score=p.score,
                rank=rank,
            )
            for rank, p in enumerate((p for p in passages if p.score >= min_score), start=1)
        ]
        return SearchResult(
            status="ok" if hits else "empty",
            course_id=course_id,
            embedding_space=space.id,
            k=k,
            min_score=min_score,
            hits=hits,
        )
