"""Search within one course: the question's vector, the best passages by the configured strategy, and the lowest
score every hit needs."""

from opentelemetry.trace import Span, Tracer

from study_assistant.embed.base import Embedder
from study_assistant.retrieval.models import SearchHit, SearchResult, SearchStrategy
from study_assistant.retrieval.repository import HybridQuery, Passage, SearchStore
from study_assistant.tracing import stage_span

# The constant of reciprocal rank fusion: a passage scores 1 / (60 + place) in each ranking. 60 is the value of the
# method's original paper and the default of common search engines; it keeps the first places close, so a passage
# placed well in both rankings comes before one placed first in only one.
RRF_CONSTANT = 60


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


class CourseSearch:
    """Find the best passages of a course for a question, by meaning only or by meaning and words together."""

    def __init__(
        self, store: SearchStore, embedder: Embedder, tracer: Tracer, *, strategy: SearchStrategy, window: int
    ) -> None:
        """Keep the store, the embedder, the tracer and the strategy; the space searched is always the one the
        embedder computes. ``window`` is how many passages each ranking of a hybrid search keeps before fusion: the
        most hits a request may ask for, so the hits of a smaller request are the first hits of a larger one.
        """
        self._store = store
        self._embedder = embedder
        self._tracer = tracer
        self._strategy: SearchStrategy = strategy
        self._window = window

    async def search(self, course_id: str, query: str, k: int, min_score: float) -> SearchResult:
        """Return up to ``k`` passages of the course scoring at least ``min_score``, best first.

        The course is checked before the question is embedded: a course that is not fully in the embedder's
        space raises ``CourseNotSearchableError``, so a result always covers the whole course. The result is
        "empty" when no passage reaches ``min_score``. Errors of the embedding server pass through unchanged.
        The whole search is one span of the request's trace, marked as failed when the search raises; it names the
        strategy and the hits, never the question.
        """
        with stage_span(self._tracer, "search") as span:
            span.set_attribute("search.strategy", self._strategy)
            return await self._search(course_id, query, k, min_score, span)

    async def _search(self, course_id: str, query: str, k: int, min_score: float, span: Span) -> SearchResult:
        """Run the search that ``search`` describes and record its outcome on ``span``."""
        space = self._embedder.space
        coverage = await self._store.coverage(course_id, space)
        if coverage.chunks == 0 or coverage.vectors < coverage.chunks:
            raise CourseNotSearchableError(course_id, coverage.chunks, coverage.vectors)
        vector = await self._embedder.aembed_query(query)
        if self._strategy == "hybrid":
            hybrid = HybridQuery(
                vector=vector, text=query, min_score=min_score, k=k, window=self._window, rrf_constant=RRF_CONSTANT
            )
            fused = await self._store.hybrid(course_id, space, hybrid)
            span.set_attribute("search.hybrid.by_meaning", fused.by_meaning)
            span.set_attribute("search.hybrid.by_words", fused.by_words)
            passages = fused.passages
        else:
            # The passages come closest first, so keeping those that reach the score keeps the order.
            passages = [p for p in await self._store.nearest(course_id, space, vector, k) if p.score >= min_score]
        hits = [_hit(passage, rank) for rank, passage in enumerate(passages, start=1)]
        span.set_attribute("search.hit_ids", [hit.chunk_id for hit in hits])
        return SearchResult(
            status="ok" if hits else "empty",
            course_id=course_id,
            strategy=self._strategy,
            embedding_space=space.id,
            k=k,
            min_score=min_score,
            hits=hits,
        )


def _hit(passage: Passage, rank: int) -> SearchHit:
    """Turn a stored passage into the hit at ``rank`` of a result."""
    return SearchHit(
        chunk_id=passage.chunk_id,
        section_id=passage.section_id,
        source_file=passage.source_file,
        page_start=passage.page_start,
        page_end=passage.page_end,
        text=passage.text,
        score=passage.score,
        rank=rank,
    )
