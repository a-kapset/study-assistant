"""Direct reads of the app's stored course corpus, as a second source of truth next to what its API reports.

An API that reports counts can be wrong in ways its own responses never show: counts taken from what it meant
to write, rows left over from an older edition, rows written under another course. Reading the tables
themselves catches those. Every read goes through the stack's read-only role.
"""

import hashlib
from typing import Self

import psycopg
from psycopg.conninfo import make_conninfo
from pydantic import SecretStr

from evalharness.settings import HarnessSettings, StackSettings

# Sections are counted only when they have a heading, as the app counts them.
_COUNTS = """
SELECT
    (SELECT count(*) FROM documents WHERE course_id = %(course)s),
    (SELECT count(*) FROM sections WHERE course_id = %(course)s AND section_id IS NOT NULL),
    (SELECT count(*) FROM learning_objectives WHERE course_id = %(course)s),
    (SELECT count(*) FROM chunks WHERE course_id = %(course)s)
"""


def conninfo(settings: HarnessSettings, stack: StackSettings, user: str, password: SecretStr) -> str:
    """Build a connection string to the stack's published database port for ``user``."""
    return make_conninfo(
        host=settings.db_host,
        port=stack.db_published_port,
        dbname=stack.db_name,
        user=user,
        password=password.get_secret_value(),
        connect_timeout=5,
    )


class CorpusReader:
    """Read-only access to the corpus tables; use it as a context manager so the connection is closed."""

    def __init__(self, settings: HarnessSettings, stack: StackSettings) -> None:
        """Connect as the read-only role; autocommit, so each read sees everything committed before it."""
        self._conn = psycopg.connect(
            conninfo(settings, stack, stack.db_reader_user, stack.db_reader_password), autocommit=True
        )

    def __enter__(self) -> Self:
        """Return the open reader."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the connection."""
        self._conn.close()

    def counts(self, course_id: str) -> dict[str, int]:
        """Count a course's documents, headed sections, learning objectives and chunks in the tables."""
        row = self._conn.execute(_COUNTS, {"course": course_id}).fetchone()
        if row is None:
            raise RuntimeError("a query of four counts returned no row")
        return dict(zip(("documents", "sections", "learning_objectives", "chunks"), row, strict=True))

    def chunk_ids(self, course_id: str) -> list[str]:
        """List a course's chunk ids in reading order: by source file, then position."""
        rows = self._conn.execute(
            "SELECT chunk_id FROM chunks WHERE course_id = %s ORDER BY source_file, position", (course_id,)
        ).fetchall()
        return [row[0] for row in rows]

    def chunk_texts(self, course_id: str) -> list[str]:
        """Return the text of every stored chunk of a course, in reading order."""
        rows = self._conn.execute(
            "SELECT text FROM chunks WHERE course_id = %s ORDER BY source_file, position", (course_id,)
        ).fetchall()
        return [row[0] for row in rows]

    def section_ids(self, course_id: str) -> set[str]:
        """Return the ids of a course's headed sections, in the form search hits name them (e.g. "4.2")."""
        rows = self._conn.execute(
            "SELECT DISTINCT section_id FROM sections WHERE course_id = %s AND section_id IS NOT NULL", (course_id,)
        ).fetchall()
        return {row[0] for row in rows}

    def snapshot(self, course_id: str) -> str:
        """Name the course's current corpus: the SHA-256 of its chunk ids, sorted.

        The app makes a chunk id from the source file's hash, the section, the position and the text, so the name
        changes when the text or the section of any chunk changes, and stays the same when an unchanged course is
        ingested again.
        """
        ids = sorted(self.chunk_ids(course_id))
        return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()

    def documents(self, course_id: str) -> dict[str, str]:
        """Map each stored document of a course to the SHA-256 of the file it was made from."""
        rows = self._conn.execute(
            "SELECT source_file, sha256 FROM documents WHERE course_id = %s", (course_id,)
        ).fetchall()
        return {file: sha256 for file, sha256 in rows}
