"""Reading a course's passages and vectors for a search, from PostgreSQL."""

from dataclasses import dataclass
from typing import Any, Protocol

from psycopg import sql

from study_assistant.db import Database
from study_assistant.embed.base import EmbeddingSpace


@dataclass(frozen=True)
class Coverage:
    """How many chunks a course has, and how many of them have a vector in one embedding space."""

    chunks: int
    vectors: int


@dataclass(frozen=True)
class Passage:
    """A stored chunk with its cosine similarity to the question."""

    chunk_id: str
    section_id: str | None
    source_file: str
    page_start: int
    page_end: int
    text: str
    score: float


@dataclass(frozen=True)
class HybridQuery:
    """What a hybrid search ranks by, and its limits.

    Only passages whose cosine similarity reaches ``min_score`` are ranked, by meaning and by the words of ``text``.
    Each ranking keeps its first ``window`` passages; reciprocal rank fusion then scores a passage
    1 / (``rrf_constant`` + place) in each ranking it is in, adds the scores, and the first ``k`` passages are kept.
    """

    vector: list[float]
    text: str
    min_score: float
    k: int
    window: int
    rrf_constant: int


@dataclass(frozen=True)
class Fused:
    """The fused passages of a hybrid search, best first, and how many passages each ranking held before fusion.

    The two counts show what each ranking contributed: a ranking by words that holds no passage for most questions
    makes a hybrid search the same as a search by meaning, which the hits alone do not show.
    """

    passages: list[Passage]
    by_meaning: int
    by_words: int


class SearchStore(Protocol):
    """What a search needs from the database: whether a course can be searched, and its best passages.

    The search service depends on this protocol; unit tests pass a fake that holds passages in memory, so the
    rules of a search are tested without a database.
    """

    async def coverage(self, course_id: str, space: EmbeddingSpace) -> Coverage:
        """Count the course's chunks and those of them that have a vector in ``space``."""
        ...

    async def nearest(self, course_id: str, space: EmbeddingSpace, vector: list[float], k: int) -> list[Passage]:
        """Return up to ``k`` passages of the course, closest to ``vector`` first, whatever their score."""
        ...

    async def hybrid(self, course_id: str, space: EmbeddingSpace, query: HybridQuery) -> Fused:
        """Return the passages of the course that reach the lowest score, ranked by meaning and by words together."""
        ...


# Every chunk of the course, joined with its vector if it has one, so one statement gives both counts.
_COVERAGE = """
SELECT count(*), count(v.chunk_id)
FROM chunks c LEFT JOIN {table} v ON v.chunk_id = c.chunk_id
WHERE c.course_id = %s
"""

# "<=>" is pgvector's cosine distance, which the vector index is built for; similarity is 1 minus the distance.
# The vector table holds the course id itself, so the course filter is applied where the vectors are read.
_NEAREST = """
SELECT c.chunk_id, c.section_id, c.source_file, c.page_start, c.page_end, c.text,
       1 - (v.embedding <=> %(vector)s::vector) AS score
FROM {table} v JOIN chunks c ON c.chunk_id = v.chunk_id
WHERE v.course_id = %(course)s
ORDER BY v.embedding <=> %(vector)s::vector
LIMIT %(k)s
"""

# The hybrid search in one statement, after the hybrid search example of pgvector-python:
# - words: the question's stems joined by OR. plainto_tsquery joins them by AND, and a chunk rarely holds every
#   word of a question, so the ranking by words would be empty for most questions. The course's language picks
#   the configuration, the same one that built the chunks' text_search column.
# - passing: the course's passages that reach the lowest score; both rankings are made from them only, so every
#   hit reaches the lowest score and a question no passage reaches stays empty.
# - by_meaning / by_words: each ranking's first places; row_number with the chunk id as the second key gives every
#   passage its own place, so equal scores cannot swap places between two runs.
# - the fused score adds 1 / (constant + place) over the rankings a passage is in; equal fused scores are ordered
#   by chunk id for the same reason.
_HYBRID = """
WITH words AS (
    SELECT CAST(replace(plainto_tsquery(text_search_config(language), %(text)s)::text, ' & ', ' | ') AS tsquery)
           AS query
    FROM courses WHERE course_id = %(course)s
),
passing AS (
    SELECT v.chunk_id, 1 - (v.embedding <=> %(vector)s::vector) AS score
    FROM {table} v
    WHERE v.course_id = %(course)s AND 1 - (v.embedding <=> %(vector)s::vector) >= %(min_score)s
),
by_meaning AS (
    SELECT chunk_id, row_number() OVER (ORDER BY score DESC, chunk_id) AS place
    FROM passing ORDER BY place LIMIT %(window)s
),
by_words AS (
    SELECT p.chunk_id, row_number() OVER (ORDER BY ts_rank_cd(c.text_search, w.query) DESC, p.chunk_id) AS place
    FROM passing p JOIN chunks c ON c.chunk_id = p.chunk_id CROSS JOIN words w
    WHERE c.text_search @@ w.query
    ORDER BY place LIMIT %(window)s
),
fused AS (
    SELECT chunk_id,
           coalesce(1.0 / (%(rrf_constant)s + m.place), 0) + coalesce(1.0 / (%(rrf_constant)s + w.place), 0) AS fused
    FROM by_meaning m FULL JOIN by_words w USING (chunk_id)
)
SELECT c.chunk_id, c.section_id, c.source_file, c.page_start, c.page_end, c.text, p.score,
       (SELECT count(*) FROM by_meaning), (SELECT count(*) FROM by_words)
FROM fused f JOIN passing p USING (chunk_id) JOIN chunks c USING (chunk_id)
ORDER BY f.fused DESC, f.chunk_id
LIMIT %(k)s
"""


class PostgresSearchStore:
    """The real search store: read-only statements against the corpus and the space's vector table."""

    def __init__(self, database: Database) -> None:
        """Keep the database; nothing is read until a search."""
        self._database = database

    async def coverage(self, course_id: str, space: EmbeddingSpace) -> Coverage:
        """Count the course's chunks and those of them that have a vector in ``space``."""
        async with self._database.connection() as conn:
            cur = await conn.execute(_in_space(_COVERAGE, space), (course_id,))
            row = await cur.fetchone()
        assert row is not None  # an aggregate without GROUP BY always returns one row
        return Coverage(chunks=row[0], vectors=row[1])

    async def nearest(self, course_id: str, space: EmbeddingSpace, vector: list[float], k: int) -> list[Passage]:
        """Return up to ``k`` passages of the course, closest to ``vector`` first, whatever their score."""
        async with self._database.connection() as conn:
            cur = await conn.execute(_in_space(_NEAREST, space), {"vector": vector, "course": course_id, "k": k})
            rows = await cur.fetchall()
        return [_passage(row) for row in rows]

    async def hybrid(self, course_id: str, space: EmbeddingSpace, query: HybridQuery) -> Fused:
        """Return the passages of the course that reach the lowest score, ranked by meaning and by words together.

        When no passage reaches the lowest score, both rankings are empty and so are their counts.
        """
        params = {
            "course": course_id,
            "vector": query.vector,
            "text": query.text,
            "min_score": query.min_score,
            "k": query.k,
            "window": query.window,
            "rrf_constant": query.rrf_constant,
        }
        async with self._database.connection() as conn:
            cur = await conn.execute(_in_space(_HYBRID, space), params)
            rows = await cur.fetchall()
        if not rows:
            return Fused(passages=[], by_meaning=0, by_words=0)
        return Fused(passages=[_passage(row[:7]) for row in rows], by_meaning=rows[0][7], by_words=rows[0][8])


def _passage(row: tuple[Any, ...]) -> Passage:
    """Turn a row of chunk id, section id, file, first and last page, text and score into a passage."""
    chunk_id, section_id, source_file, page_start, page_end, text, score = row
    return Passage(
        chunk_id=chunk_id,
        section_id=section_id,
        source_file=source_file,
        page_start=page_start,
        page_end=page_end,
        text=text,
        score=float(score),
    )


def _in_space(statement: str, space: EmbeddingSpace) -> sql.Composed:
    """Put the space's vector table into a statement, quoted as an identifier."""
    return sql.SQL(statement).format(table=sql.Identifier(space.vector_table))
