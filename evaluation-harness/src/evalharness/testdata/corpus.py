"""Test data in the app's course corpus tables: copies of real courses, extra documents, documents made to look old.

Rows are written as the app's own database role, the only role that may write them. Every row this module
creates carries the test data prefix in its course id, its file name, its chunk id or its hash, and
``remove_all`` deletes by that prefix. A document's sections, objectives and chunks are deleted with it by the
database, and a course's documents with the course.
"""

import psycopg

from evalharness.corpus_db import conninfo
from evalharness.settings import HarnessSettings, StackSettings
from evalharness.testdata import PREFIX, name

# The hash written into a stored document to make it look made from an older version of its file.
AGED_SHA256 = name("aged")

# A course copied row by row: the copy has the same documents, sections, objectives and chunks as the real one,
# so checks see data of the shape the app really stores. Chunk ids are unique across courses, so the copy's
# chunk ids get the prefix.
_COPY_COURSE = [
    """INSERT INTO courses (course_id, title, language)
       SELECT %(target)s, title, language FROM courses WHERE course_id = %(source)s""",
    """INSERT INTO documents (course_id, source_file, sha256, chunker_version, rules_sha256, role, authority,
                              parser, parser_version, engine_version, page_count)
       SELECT %(target)s, source_file, sha256, chunker_version, rules_sha256, role, authority,
              parser, parser_version, engine_version, page_count
       FROM documents WHERE course_id = %(source)s""",
    """INSERT INTO sections (course_id, source_file, position, section_id, parent_id, title, page_start, page_end)
       SELECT %(target)s, source_file, position, section_id, parent_id, title, page_start, page_end
       FROM sections WHERE course_id = %(source)s""",
    """INSERT INTO learning_objectives (course_id, source_file, objective_id, section_id, knowledge_level, page, line)
       SELECT %(target)s, source_file, objective_id, section_id, knowledge_level, page, line
       FROM learning_objectives WHERE course_id = %(source)s""",
    """INSERT INTO chunks (chunk_id, course_id, source_file, position, section_id, objective_ids, page_start,
                           page_end, language, text)
       SELECT %(prefix)s || chunk_id, %(target)s, source_file, position, section_id, objective_ids, page_start,
              page_end, language, text
       FROM chunks WHERE course_id = %(source)s""",
]

_INSERT_DOCUMENT = """
INSERT INTO documents (course_id, source_file, sha256, chunker_version, rules_sha256, role, authority,
                       parser, parser_version, engine_version, page_count)
VALUES (%s, %s, %s, %s, %s, 'material', 'primary', %s, '0', '0', 1)
"""

# Documents with a prefixed file name or hash, in any course; then prefixed courses with all their rows.
_REMOVE_DOCUMENTS = "DELETE FROM documents WHERE starts_with(source_file, %(p)s) OR starts_with(sha256, %(p)s)"
_REMOVE_COURSES = "DELETE FROM courses WHERE starts_with(course_id, %(p)s)"

_LEFTOVERS = """
SELECT 'course ' || course_id FROM courses WHERE starts_with(course_id, %(p)s)
UNION ALL
SELECT 'document ' || course_id || '/' || source_file FROM documents
WHERE starts_with(course_id, %(p)s) OR starts_with(source_file, %(p)s) OR starts_with(sha256, %(p)s)
UNION ALL
SELECT 'chunk ' || chunk_id FROM chunks WHERE starts_with(chunk_id, %(p)s) OR starts_with(course_id, %(p)s)
"""


def connect_writer(settings: HarnessSettings, stack: StackSettings) -> psycopg.Connection:
    """Open a connection as the app's role, with every statement saved at once (autocommit).

    The data must be saved before the test calls the app, because the app reads it through its own connection.
    """
    return psycopg.connect(conninfo(settings, stack, stack.db_app_user, stack.db_app_password), autocommit=True)


class CorpusTestData:
    """Creates test data in the corpus tables through one open connection of the app's role."""

    def __init__(self, conn: psycopg.Connection) -> None:
        """Keep the connection used for writing."""
        self._conn = conn

    def copy_course(self, source: str, target: str) -> None:
        """Store a copy of an ingested course under the id ``target``, which must start with the prefix.

        Every row is copied: documents, sections, objectives and chunks. Use it to put another course next to the
        one under test that looks like a real ingested course, with the same file names. All rows are written
        in one transaction, so a failed copy leaves nothing behind.
        """
        _require_prefix(target)
        params = {"source": source, "target": target, "prefix": PREFIX}
        with self._conn.transaction():
            if self._conn.execute(_COPY_COURSE[0], params).rowcount != 1:
                raise LookupError(f"no stored course {source!r} to copy; ingest it first")
            for statement in _COPY_COURSE[1:]:
                self._conn.execute(statement, params)

    def add_document(self, course_id: str, file: str) -> None:
        """Store one more document row in an existing course; the file name must start with the prefix.

        The row has fake version values and no sections or chunks. Used to give a course a document that its
        manifest does not list, as if the file had been removed from the course.
        """
        _require_prefix(file)
        marker = name("fake")
        self._conn.execute(_INSERT_DOCUMENT, (course_id, file, marker, marker, marker, marker))

    def age_document(self, course_id: str, file: str) -> None:
        """Replace the stored file hash of a document with ``AGED_SHA256``.

        The app then sees a document made from another version of its file and must rebuild it on the next
        ingest, which also writes the real hash back. If that does not happen, ``remove_all`` deletes the
        document, and the next ingest stores it again.
        """
        cursor = self._conn.execute(
            "UPDATE documents SET sha256 = %s WHERE course_id = %s AND source_file = %s", (AGED_SHA256, course_id, file)
        )
        if cursor.rowcount != 1:
            raise LookupError(f"no stored document {course_id}/{file} to age; ingest the course first")


def remove_all(conn: psycopg.Connection) -> int:
    """Delete every corpus row that carries the prefix and return how many document and course rows went."""
    removed = conn.execute(_REMOVE_DOCUMENTS, {"p": PREFIX}).rowcount
    removed += conn.execute(_REMOVE_COURSES, {"p": PREFIX}).rowcount
    return removed


def leftovers(conn: psycopg.Connection) -> list[str]:
    """List the corpus rows that still carry the prefix, such as "course evalharness_foreign"."""
    return [row[0] for row in conn.execute(_LEFTOVERS, {"p": PREFIX}).fetchall()]


def _require_prefix(value: str) -> None:
    """Refuse a name without the prefix: the cleanup could not find such data and it would stay in the database."""
    if not value.startswith(PREFIX):
        raise ValueError(f"test data names must start with {PREFIX!r}, got {value!r}")
