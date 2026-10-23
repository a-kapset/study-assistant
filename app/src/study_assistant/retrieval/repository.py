"""Reading a course's passages and vectors for a search, from PostgreSQL."""

from dataclasses import dataclass
from typing import Protocol

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


class SearchStore(Protocol):
    """What a search needs from the database: whether a course can be searched, and its closest passages.

    The search service depends on this protocol; unit tests pass a fake that holds passages in memory, so the
    rules of a search are tested without a database.
    """

    async def coverage(self, course_id: str, space: EmbeddingSpace) -> Coverage:
        """Count the course's chunks and those of them that have a vector in ``space``."""
        ...

    async def nearest(self, course_id: str, space: EmbeddingSpace, vector: list[float], k: int) -> list[Passage]:
        """Return up to ``k`` passages of the course, closest to ``vector`` first, whatever their score."""
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


class PostgresSearchStore:
    """The real search store: two read-only statements against the corpus and the space's vector table."""

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
        return [
            Passage(
                chunk_id=chunk_id,
                section_id=section_id,
                source_file=source_file,
                page_start=page_start,
                page_end=page_end,
                text=text,
                score=float(score),
            )
            for chunk_id, section_id, source_file, page_start, page_end, text, score in rows
        ]


def _in_space(statement: str, space: EmbeddingSpace) -> sql.Composed:
    """Put the space's vector table into a statement, quoted as an identifier."""
    return sql.SQL(statement).format(table=sql.Identifier(space.vector_table))
