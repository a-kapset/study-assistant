"""The course corpus in PostgreSQL: the SQL behind ingestion's writes and the counts read back.

Every statement names the course, so writing one course can never change another course's rows, even where
both courses have a file of the same name.
"""

from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager

from study_assistant.db import Connection, Database
from study_assistant.ingest.service import CorpusCounts, Edition, PreparedDocument

# Sections are counted only when they have a heading: the text before a file's first heading is not a section.
_COUNTS = """
SELECT
    (SELECT count(*) FROM documents WHERE course_id = %(course)s),
    (SELECT count(*) FROM sections WHERE course_id = %(course)s AND section_id IS NOT NULL),
    (SELECT count(*) FROM learning_objectives WHERE course_id = %(course)s),
    (SELECT count(*) FROM chunks WHERE course_id = %(course)s)
"""

_EDITIONS = """
SELECT source_file, sha256, chunker_version, parser, parser_version, engine_version, rules_sha256
FROM documents WHERE course_id = %s
"""

_SAVE_COURSE = """
INSERT INTO courses (course_id, title, language) VALUES (%s, %s, %s)
ON CONFLICT (course_id) DO UPDATE SET title = EXCLUDED.title, language = EXCLUDED.language
"""

_INSERT_DOCUMENT = """
INSERT INTO documents (course_id, source_file, sha256, chunker_version, rules_sha256, role, authority,
                       parser, parser_version, engine_version, page_count)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

_INSERT_SECTION = """
INSERT INTO sections (course_id, source_file, position, section_id, parent_id, title, page_start, page_end)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
"""

_INSERT_OBJECTIVE = """
INSERT INTO learning_objectives (course_id, source_file, objective_id, section_id, knowledge_level, page, line)
VALUES (%s, %s, %s, %s, %s, %s, %s)
"""

_INSERT_CHUNK = """
INSERT INTO chunks (chunk_id, course_id, source_file, position, section_id, objective_ids, page_start, page_end,
                    language, text)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


class PostgresCorpusStore:
    """The corpus store of the running app, on the app's connection pool."""

    def __init__(self, database: Database) -> None:
        """Keep the database; connections are taken from it per call."""
        self._database = database

    @asynccontextmanager
    async def course_writer(self, course_id: str) -> AsyncGenerator[PostgresCourseWriter]:
        """Open one transaction for a course; it commits when the block ends and rolls back on an error."""
        async with self._database.connection() as conn, conn.transaction():
            # Held until the transaction ends: a second ingestion of the course waits here instead of reading
            # the stored documents before the first one has written them.
            await conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (course_id,))
            yield PostgresCourseWriter(conn, course_id)

    async def counts(self, course_id: str) -> CorpusCounts:
        """Count a course's stored rows outside any ingestion."""
        async with self._database.connection() as conn:
            return await _count(conn, course_id)


class PostgresCourseWriter:
    """The writes of one course inside the transaction ``PostgresCorpusStore.course_writer`` opened."""

    def __init__(self, conn: Connection, course_id: str) -> None:
        """Keep the transaction's connection and the course every statement is limited to."""
        self._conn = conn
        self._course_id = course_id

    async def editions(self) -> dict[str, Edition]:
        """Return what each stored document of the course was made from, keyed by source file."""
        cursor = await self._conn.execute(_EDITIONS, (self._course_id,))
        return {row[0]: Edition(*row[1:]) for row in await cursor.fetchall()}

    async def save_course(self, title: str, language: str) -> None:
        """Create the course's row, or update its title and language from the manifest."""
        await self._conn.execute(_SAVE_COURSE, (self._course_id, title, language))

    async def delete_documents(self, source_files: Sequence[str]) -> None:
        """Delete these documents of this course; their sections, objectives and chunks go with them."""
        if source_files:
            await self._conn.execute(
                "DELETE FROM documents WHERE course_id = %s AND source_file = ANY(%s)",
                (self._course_id, list(source_files)),
            )

    async def insert_document(self, document: PreparedDocument) -> None:
        """Store a document, then its sections, objectives and chunks, all in reading order."""
        course, file, edition = self._course_id, document.source.file, document.edition
        await self._conn.execute(
            _INSERT_DOCUMENT,
            (
                course,
                file,
                edition.sha256,
                edition.chunker_version,
                edition.rules_sha256,
                document.source.role,
                document.source.authority,
                edition.parser,
                edition.parser_version,
                edition.engine_version,
                document.page_count,
            ),
        )
        structure = document.structure
        async with self._conn.cursor() as cursor:
            await cursor.executemany(
                _INSERT_SECTION,
                [
                    (course, file, position, s.id, s.parent_id, s.title, s.lines[0].page, s.lines[-1].page)
                    for position, s in enumerate(structure.sections)
                ],
            )
            await cursor.executemany(
                _INSERT_OBJECTIVE,
                [
                    (course, file, o.id, o.section_id, o.knowledge_level, o.line.page, o.line.text)
                    for o in structure.objectives
                ],
            )
            await cursor.executemany(
                _INSERT_CHUNK,
                [
                    (
                        c.id,
                        course,
                        file,
                        position,
                        c.section_id,
                        list(c.lo_ids),
                        c.page_start,
                        c.page_end,
                        c.language,
                        c.text,
                    )
                    for position, c in enumerate(document.chunks)
                ],
            )

    async def counts(self) -> CorpusCounts:
        """Count the course's rows as this transaction sees them, including its own writes."""
        return await _count(self._conn, self._course_id)


async def _count(conn: Connection, course_id: str) -> CorpusCounts:
    """Read the four counts of a course in one query."""
    cursor = await conn.execute(_COUNTS, {"course": course_id})
    row = await cursor.fetchone()
    if row is None:
        raise RuntimeError("a query of four counts returned no row")
    documents, sections, objectives, chunks = row
    return CorpusCounts(documents=documents, sections=sections, learning_objectives=objectives, chunks=chunks)
