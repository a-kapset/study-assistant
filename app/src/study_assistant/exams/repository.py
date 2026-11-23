"""Question banks, practice sessions and answers in PostgreSQL.

Every read of a session names its learner, so one learner can never read or answer another learner's session.
"""

from collections.abc import Sequence
from uuid import UUID

from psycopg.rows import TupleRow
from psycopg.types.json import Jsonb

from study_assistant.db import Connection, Database
from study_assistant.exams.bank import BankItem
from study_assistant.exams.service import ObjectiveCount, SessionState, StoredAnswer, StoredItem, StoredOption

# A question of the file that is already stored is updated in place and offered again.
_UPSERT_ITEM = """
INSERT INTO bank_items (course_id, item_id, position, stem, options, keys, objective_id, retired)
VALUES (%s, %s, %s, %s, %s, %s, %s, false)
ON CONFLICT (course_id, item_id) DO UPDATE SET
    position = EXCLUDED.position, stem = EXCLUDED.stem, options = EXCLUDED.options, keys = EXCLUDED.keys,
    objective_id = EXCLUDED.objective_id, retired = false
"""

_RETIRE_MISSING = "UPDATE bank_items SET retired = true WHERE course_id = %s AND NOT (item_id = ANY(%s))"

_COUNT_ACTIVE = "SELECT count(*) FROM bank_items WHERE course_id = %s AND NOT retired"

_ACTIVE_ITEMS = """
SELECT item_id, stem, options, keys, objective_id FROM bank_items
WHERE course_id = %s AND NOT retired ORDER BY position
"""

_INSERT_SESSION = "INSERT INTO exam_sessions (learner_id, course_id) VALUES (%s, %s) RETURNING session_id"

_INSERT_SESSION_ITEM = """
INSERT INTO exam_session_items (session_id, position, course_id, item_id) VALUES (%s, %s, %s, %s)
"""

# The learner is part of the condition: a session of another learner is not found.
_SESSION_COURSE = "SELECT course_id FROM exam_sessions WHERE session_id = %s AND learner_id = %s"

_SESSION_ITEMS = """
SELECT b.item_id, b.stem, b.options, b.keys, b.objective_id
FROM exam_session_items i JOIN bank_items b ON b.course_id = i.course_id AND b.item_id = i.item_id
WHERE i.session_id = %s ORDER BY i.position
"""

_SESSION_ANSWERS = "SELECT item_id, selected, correct FROM exam_answers WHERE session_id = %s"

# The primary key allows one answer per question of a session; a second one is not stored, whatever it says.
_INSERT_ANSWER = """
INSERT INTO exam_answers (session_id, item_id, selected, correct, objective_id) VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (session_id, item_id) DO NOTHING
"""

_STORED_ANSWER = "SELECT selected, correct FROM exam_answers WHERE session_id = %s AND item_id = %s"

# Both the learner and the course limit the sum: answers of other learners and of other courses stay out.
_PROGRESS = """
SELECT a.objective_id, count(*), count(*) FILTER (WHERE a.correct)
FROM exam_answers a JOIN exam_sessions s ON s.session_id = a.session_id
WHERE s.learner_id = %s AND s.course_id = %s
GROUP BY a.objective_id ORDER BY a.objective_id
"""


class PostgresExamStore:
    """The exam store of the running app, on the app's connection pool."""

    def __init__(self, database: Database) -> None:
        """Keep the database; connections are taken from it per call."""
        self._database = database

    async def replace_bank(self, course_id: str, items: Sequence[BankItem]) -> int:
        """Store the questions and retire the stored ones that are not among them, in one transaction."""
        async with self._database.connection() as conn, conn.transaction():
            await conn.execute(_RETIRE_MISSING, (course_id, [item.id for item in items]))
            for position, item in enumerate(items):
                options = Jsonb([{"id": option.id, "text": option.text} for option in item.options])
                await conn.execute(
                    _UPSERT_ITEM, (course_id, item.id, position, item.stem, options, list(item.keys), item.objective_id)
                )
            cursor = await conn.execute(_COUNT_ACTIVE, (course_id,))
            row = await cursor.fetchone()
        return int(row[0]) if row is not None else 0

    async def active_items(self, course_id: str) -> list[StoredItem]:
        """Read the questions of a course that are not retired, in bank order."""
        async with self._database.connection() as conn:
            cursor = await conn.execute(_ACTIVE_ITEMS, (course_id,))
            return [_item(row) for row in await cursor.fetchall()]

    async def create_session(self, learner_id: UUID, course_id: str, item_ids: Sequence[str]) -> UUID:
        """Insert the session and its questions in one transaction, so a session never exists without them."""
        async with self._database.connection() as conn, conn.transaction():
            cursor = await conn.execute(_INSERT_SESSION, (learner_id, course_id))
            row = await cursor.fetchone()
            if row is None:
                raise RuntimeError("the database returned no id for the new session")
            session_id: UUID = row[0]
            for position, item_id in enumerate(item_ids, start=1):
                await conn.execute(_INSERT_SESSION_ITEM, (session_id, position, course_id, item_id))
        return session_id

    async def session(self, learner_id: UUID, session_id: UUID) -> SessionState | None:
        """Read the learner's session with its questions and answers; ``None`` when the learner has no such one."""
        async with self._database.connection() as conn:
            cursor = await conn.execute(_SESSION_COURSE, (session_id, learner_id))
            row = await cursor.fetchone()
            if row is None:
                return None
            items = await conn.execute(_SESSION_ITEMS, (session_id,))
            item_rows = await items.fetchall()
            answers = await conn.execute(_SESSION_ANSWERS, (session_id,))
            answer_rows = await answers.fetchall()
        return SessionState(
            course_id=row[0],
            items=tuple(_item(item_row) for item_row in item_rows),
            answers={item_id: StoredAnswer(frozenset(selected), correct) for item_id, selected, correct in answer_rows},
        )

    async def save_answer(
        self, session_id: UUID, item_id: str, answer: StoredAnswer, objective_id: str
    ) -> StoredAnswer:
        """Insert the answer unless the question has one, then read back the answer the session holds."""
        async with self._database.connection() as conn, conn.transaction():
            stored = await _save_answer(conn, session_id, item_id, answer, objective_id)
        return stored

    async def progress(self, learner_id: UUID, course_id: str) -> list[ObjectiveCount]:
        """Sum the learner's answers in the course per objective."""
        async with self._database.connection() as conn:
            cursor = await conn.execute(_PROGRESS, (learner_id, course_id))
            return [
                ObjectiveCount(objective_id, answered, correct)
                for objective_id, answered, correct in await cursor.fetchall()
            ]


async def _save_answer(
    conn: Connection, session_id: UUID, item_id: str, answer: StoredAnswer, objective_id: str
) -> StoredAnswer:
    """Run the insert and the read-back on one connection; two requests for one question end with one row."""
    await conn.execute(_INSERT_ANSWER, (session_id, item_id, sorted(answer.selected), answer.correct, objective_id))
    cursor = await conn.execute(_STORED_ANSWER, (session_id, item_id))
    row = await cursor.fetchone()
    if row is None:
        raise RuntimeError(f"no answer is stored for question {item_id} right after storing one")
    return StoredAnswer(selected=frozenset(row[0]), correct=row[1])


def _item(row: TupleRow) -> StoredItem:
    """Build a question from its row: id, stem, options (a JSON list), keys (a text array), objective."""
    item_id, stem, options, keys, objective_id = row
    return StoredItem(
        item_id=item_id,
        stem=stem,
        options=tuple(StoredOption(id=str(option["id"]), text=str(option["text"])) for option in options),
        keys=frozenset(str(key) for key in keys),
        objective_id=objective_id,
    )
