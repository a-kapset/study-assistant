"""Search within one course: the question's vector, the best passages by the configured strategy, the lowest score
every hit needs, and, when it is on, a new order of those passages by the reranker.

When the embedding server or the rerank server cannot be reached, the search still answers on a weaker path and
says so: without the vector it ranks the passages by the question's words only, without the reranker it keeps the
strategy's order. A server that answers with an error is not such a case: the error passes through, because a
broken request or reply is a fault to fix, not an outage to wait out.
"""

import structlog
from opentelemetry.trace import Span, Tracer

from study_assistant.embed.base import Embedder
from study_assistant.embed.remote import EmbeddingServiceUnavailableError
from study_assistant.retrieval.models import DegradedReason, SearchHit, SearchResult, SearchStatus, SearchStrategy
from study_assistant.retrieval.repository import HybridQuery, Passage, SearchStore
from study_assistant.retrieval.rerank import Reranker, RerankServiceUnavailableError
from study_assistant.tracing import stage_span

# The constant of reciprocal rank fusion: a passage scores 1 / (60 + place) in each ranking. 60 is the value of the
# method's original paper and the default of common search engines; it keeps the first places close, so a passage
# placed well in both rankings comes before one placed first in only one.
RRF_CONSTANT = 60
# Name of the span of the reranking stage, inside the search span.
RERANK_SPAN = "rerank"


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
    """Find the best passages of a course for a question, by meaning only or by meaning and words together, and
    put them in the reranker's order when one is given."""

    def __init__(
        self,
        store: SearchStore,
        embedder: Embedder,
        tracer: Tracer,
        *,
        strategy: SearchStrategy,
        window: int,
        reranker: Reranker | None = None,
    ) -> None:
        """Keep the store, the embedder, the tracer, the strategy and the reranker; the space searched is always the
        one the embedder computes. ``window`` is the most hits a request may ask for: each ranking of a hybrid search
        keeps that many passages before fusion, and the reranker gets that many passages of the strategy, so the hits
        of a smaller request are the first hits of a larger one. Without ``reranker`` the strategy's order is kept.
        """
        self._store = store
        self._embedder = embedder
        self._tracer = tracer
        self._strategy: SearchStrategy = strategy
        self._window = window
        self._reranker = reranker
        self._log: structlog.typing.FilteringBoundLogger = structlog.get_logger(__name__)

    async def search(self, course_id: str, query: str, k: int, min_score: float) -> SearchResult:
        """Return up to ``k`` passages of the course scoring at least ``min_score``, best first.

        The course is checked before the question is embedded: a course that is not fully in the embedder's
        space raises ``CourseNotSearchableError``, so a result always covers the whole course. The result is
        "empty" when no passage reaches ``min_score``. An embedding server that cannot be reached gives a "degraded"
        result ranked by the question's words only, without the lowest score and without the reranker; a rerank
        server that cannot be reached gives a "degraded" result in the strategy's order. Other errors of both
        servers pass through unchanged. The whole search is one span of the request's trace, marked as failed when
        the search raises; it names the strategy, whether the reranker was to run, the status, the reason of a
        degraded result and the hits, never the question.
        """
        with stage_span(self._tracer, "search") as span:
            span.set_attribute("search.strategy", self._strategy)
            span.set_attribute("search.rerank", self._reranker is not None)
            result = await self._search(course_id, query, k, min_score, span)
            span.set_attribute("search.status", result.status)
            if result.degraded_reason is not None:
                span.set_attribute("search.degraded_reason", result.degraded_reason)
                self._log.warning("search_degraded", course_id=course_id, reason=result.degraded_reason)
            span.set_attribute("search.hit_ids", [hit.chunk_id for hit in result.hits])
            return result

    async def _search(self, course_id: str, query: str, k: int, min_score: float, span: Span) -> SearchResult:
        """Run the search that ``search`` describes; ``span`` gets the counts of a hybrid search and the candidates
        given to the reranker."""
        space = self._embedder.space
        coverage = await self._store.coverage(course_id, space)
        if coverage.chunks == 0 or coverage.vectors < coverage.chunks:
            raise CourseNotSearchableError(course_id, coverage.chunks, coverage.vectors)
        try:
            vector = await self._embedder.aembed_query(query)
        except EmbeddingServiceUnavailableError:
            # Ranked by words only. The reranker is left out too: this path is measured as one thing, and an answer
            # is not written from it anyway.
            passages = await self._store.lexical(course_id, query, k)
            return self._result(course_id, k, min_score, passages, reranked=False, reason="embedder_unavailable")
        # The reranker chooses the hits from the whole window, so the strategy gives it that many passages.
        found = self._window if self._reranker is not None else k
        if self._strategy == "hybrid":
            hybrid = HybridQuery(
                vector=vector, text=query, min_score=min_score, k=found, window=self._window, rrf_constant=RRF_CONSTANT
            )
            fused = await self._store.hybrid(course_id, space, hybrid)
            span.set_attribute("search.hybrid.by_meaning", fused.by_meaning)
            span.set_attribute("search.hybrid.by_words", fused.by_words)
            passages = fused.passages
        else:
            # The passages come closest first, so keeping those that reach the score keeps the order.
            nearest = await self._store.nearest(course_id, space, vector, found)
            passages = [p for p in nearest if p.score is not None and p.score >= min_score]
        if self._reranker is None:
            return self._result(course_id, k, min_score, passages, reranked=False, reason=None)
        try:
            reranked = await self._rerank(self._reranker, query, passages, span)
        except RerankServiceUnavailableError:
            return self._result(course_id, k, min_score, passages[:k], reranked=False, reason="reranker_unavailable")
        return self._result(course_id, k, min_score, reranked[:k], reranked=True, reason=None)

    def _result(
        self,
        course_id: str,
        k: int,
        min_score: float,
        passages: list[Passage],
        *,
        reranked: bool,
        reason: DegradedReason | None,
    ) -> SearchResult:
        """Build the result of ``passages`` in their order; with a ``reason`` it is "degraded", else "ok" or "empty"."""
        hits = [_hit(passage, rank) for rank, passage in enumerate(passages, start=1)]
        status: SearchStatus = "degraded" if reason is not None else "ok" if hits else "empty"
        return SearchResult(
            status=status,
            degraded_reason=reason,
            course_id=course_id,
            strategy=self._strategy,
            reranked=reranked,
            embedding_space=self._embedder.space.id,
            k=k,
            min_score=min_score,
            hits=hits,
        )

    async def _rerank(self, reranker: Reranker, query: str, passages: list[Passage], span: Span) -> list[Passage]:
        """Return ``passages`` best first by the reranker's scores; passages with equal scores keep their order.

        The search span keeps the passages' order before reranking, so a trace shows whether a passage was never
        found or was found and then placed too low. The call to the reranker is a span of its own.
        """
        span.set_attribute("search.rerank.candidate_ids", [passage.chunk_id for passage in passages])
        with stage_span(self._tracer, RERANK_SPAN):
            scores = await reranker.ascore(query, [passage.text for passage in passages])
        # sorted() keeps the order of equal keys, so a tie keeps the strategy's order.
        order = sorted(range(len(passages)), key=lambda place: -scores[place])
        return [passages[place] for place in order]


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
