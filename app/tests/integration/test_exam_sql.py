"""Question banks, sessions, answers and learners against a real PostgreSQL, as the app's own role.

The unit tests run the exam rules on an in-memory store that assumes what the tables do: a question that left the
bank stays readable, a session is found only for its learner, a question holds one answer, and progress sums only
the learner's answers in the course. These tests check those assumptions on the real tables, through the same
service the endpoints call. Course rows are created the way ingestion creates them.
"""

import asyncio
from typing import TYPE_CHECKING
from uuid import UUID

import pytest

from study_assistant.courses.models import ExamSettings
from study_assistant.db import PostgresDatabase
from study_assistant.exams.bank import BankItem, BankOption, BankSource
from study_assistant.exams.repository import PostgresExamStore
from study_assistant.exams.service import AlreadyAnsweredError, AnswerResult, ExamService, SessionNotFoundError
from study_assistant.ingest.repository import PostgresCorpusStore
from study_assistant.learners.repository import PostgresLearnerStore
from study_assistant.learners.service import Learners

if TYPE_CHECKING:
    from .conftest import FixedLengthEmbedder, MigratedDatabase

pytestmark = [pytest.mark.database, pytest.mark.anyio]

COURSE = "river_board"
OTHER_COURSE = "hill_board"
SETTINGS = ExamSettings(pass_mark_percent=50)


def _item(item_id: str, objective_id: str = "OBJ-01", stem: str | None = None) -> BankItem:
    """One question of an invented course with the options a to c and the key b."""
    return BankItem(
        id=item_id,
        stem=stem or f"Question {item_id} about the board.",
        options=tuple(BankOption(id=option_id, text=f"Answer {option_id}") for option_id in "abc"),
        keys=("b",),
        objective_id=objective_id,
        source=BankSource(document="board_questions.pdf", version="1", question=item_id),
    )


BANK = (_item("Q1"), _item("Q2"), _item("Q3", objective_id="OBJ-02"))


async def _exams(database: PostgresDatabase, embedder: FixedLengthEmbedder, *course_ids: str) -> ExamService:
    """The exam service on the real store, with a row and the bank for each given course."""
    corpus = PostgresCorpusStore(database)
    service = ExamService(PostgresExamStore(database))
    for course_id in course_ids:
        async with corpus.course_writer(course_id, embedder.space) as writer:
            await writer.save_course(f"Course {course_id}", "en")
        await service.load_bank(course_id, BANK)
    return service


async def _learner(database: PostgresDatabase) -> UUID:
    """Register a learner the way the admin endpoint does and return its id."""
    learner_id, _ = await Learners(PostgresLearnerStore(database)).register()
    return learner_id


async def _session(service: ExamService, learner_id: UUID, course_id: str = COURSE) -> UUID:
    """Start a session of the learner over the whole bank of the course."""
    return (await service.create_session(learner_id, course_id, SETTINGS, None, None)).session_id


def _answer_rows(postgres: MigratedDatabase, session_id: UUID) -> list[tuple[str, list[str], bool]]:
    """Read the stored answers of a session as the database owner: question, selected options, correct."""
    with postgres.owner_connection() as conn:
        rows = conn.execute(
            "SELECT item_id, selected, correct FROM exam_answers WHERE session_id = %s ORDER BY item_id", (session_id,)
        )
        return [(item_id, selected, correct) for item_id, selected, correct in rows.fetchall()]


@pytest.mark.covers("US-020", "AC-10")
async def test_a_question_that_left_the_bank_is_not_offered_but_stays_in_its_sessions(
    app_database: PostgresDatabase, embedder: FixedLengthEmbedder
) -> None:
    """Loads a bank of three questions, starts a session on them, then loads a bank where Q2 is gone and Q1 has a
    new text. New sessions must get Q1 (new text) and Q3 only, while the old session still shows its three
    questions. Catches a reload that deletes questions under running sessions, or keeps offering removed ones."""
    service = await _exams(app_database, embedder, COURSE)
    learner = await _learner(app_database)
    old_session = await _session(service, learner)

    offered = await service.load_bank(COURSE, (_item("Q1", stem="Question Q1, second edition."), BANK[2]))
    new_session = await service.create_session(learner, COURSE, SETTINGS, None, None)

    old_state = await service.state(learner, old_session)
    new_state = await service.state(learner, new_session.session_id)
    assert offered == 2, f"the new bank offers 2 questions, the load reports {offered}"
    assert [item.item_id for item in new_state.items] == ["Q1", "Q3"], f"a new session got {new_state.items}"
    assert new_state.items[0].stem == "Question Q1, second edition.", "a changed question must be stored anew"
    assert [item.item_id for item in old_state.items] == ["Q1", "Q2", "Q3"], "the old session lost a question"


@pytest.mark.covers("US-020", "AC-9")
async def test_a_session_is_found_only_for_the_learner_it_belongs_to(
    app_database: PostgresDatabase, embedder: FixedLengthEmbedder
) -> None:
    """Two learners; the second asks for the first one's session. Catches a session query without the learner
    in its condition: the service's refusal rests on the store returning nothing here."""
    service = await _exams(app_database, embedder, COURSE)
    owner, stranger = await _learner(app_database), await _learner(app_database)
    session_id = await _session(service, owner)

    assert (await service.state(owner, session_id)).course_id == COURSE, "the owner must find the session"
    with pytest.raises(SessionNotFoundError):
        await service.state(stranger, session_id)


@pytest.mark.covers("US-020", "AC-6")
async def test_a_question_keeps_its_first_answer(
    app_database: PostgresDatabase, embedder: FixedLengthEmbedder, postgres: MigratedDatabase
) -> None:
    """Answers Q1 wrongly, sends the same answer again, then the right one. The table must hold the first answer
    alone. Catches an insert that fails on the repeat, or an update that lets the right answer replace it."""
    service = await _exams(app_database, embedder, COURSE)
    learner = await _learner(app_database)
    session_id = await _session(service, learner)

    first = await service.answer(learner, session_id, "Q1", ["a"])
    again = await service.answer(learner, session_id, "Q1", [" A "])
    with pytest.raises(AlreadyAnsweredError):
        await service.answer(learner, session_id, "Q1", ["b"])

    assert first == again, f"the repeated answer must get the first reply: {first} / {again}"
    assert _answer_rows(postgres, session_id) == [("Q1", ["a"], False)], "the table must hold the first answer only"


@pytest.mark.covers("US-020", "AC-6")
async def test_two_different_answers_sent_at_once_end_as_one_row(
    app_database: PostgresDatabase, embedder: FixedLengthEmbedder, postgres: MigratedDatabase
) -> None:
    """Sends a right and a wrong answer to one question at the same time, ten questions' worth of times over
    fresh sessions. Each time exactly one must be accepted and one refused, with one row stored that equals the
    accepted reply. Catches "read, then insert" without the table's key: both would be accepted."""
    service = await _exams(app_database, embedder, COURSE)
    learner = await _learner(app_database)

    for _ in range(10):
        session_id = await _session(service, learner)
        outcomes = await asyncio.gather(
            service.answer(learner, session_id, "Q1", ["b"]),
            service.answer(learner, session_id, "Q1", ["c"]),
            return_exceptions=True,
        )

        accepted = [outcome for outcome in outcomes if isinstance(outcome, AnswerResult)]
        refused = [outcome for outcome in outcomes if isinstance(outcome, AlreadyAnsweredError)]
        rows = _answer_rows(postgres, session_id)
        assert len(accepted) == 1 and len(refused) == 1, f"one answer must win and one be refused: {outcomes}"
        assert len(rows) == 1 and rows[0][2] is accepted[0].correct, f"the stored row {rows} is not the accepted one"


@pytest.mark.covers("US-020", "AC-8", "AC-9")
async def test_progress_sums_only_the_learner_answers_in_the_course(
    app_database: PostgresDatabase, embedder: FixedLengthEmbedder
) -> None:
    """The database already holds answers of another learner in the same course and of the same learner in
    another course, on questions with the same ids and objectives. Progress must count neither. Catches a sum
    that misses the learner or the course in its condition, which a database with one learner would never show."""
    service = await _exams(app_database, embedder, COURSE, OTHER_COURSE)
    learner, other = await _learner(app_database), await _learner(app_database)
    for who, course, selected in ((other, COURSE, "b"), (learner, OTHER_COURSE, "b")):
        noise = await _session(service, who, course)
        for item_id in ("Q1", "Q2", "Q3"):
            await service.answer(who, noise, item_id, [selected])
    first, second = await _session(service, learner), await _session(service, learner)
    await service.answer(learner, first, "Q1", ["b"])
    await service.answer(learner, first, "Q3", ["a"])
    await service.answer(learner, second, "Q1", ["c"])

    progress = await service.progress(learner, COURSE)

    counts = [(row.objective_id, row.answered, row.correct) for row in progress.objectives]
    assert counts == [("OBJ-01", 2, 1), ("OBJ-02", 1, 0)], (
        f"progress counts answers that are not the learner's: {counts}"
    )


@pytest.mark.covers("US-020", "AC-2")
async def test_a_learner_is_found_by_its_key_and_the_key_is_not_in_the_table(
    app_database: PostgresDatabase, postgres: MigratedDatabase
) -> None:
    """Registers two learners. Each key must lead to its own learner and an unknown key to none, and the table
    must not contain either key. Catches a lookup that ignores the key, and a key stored in readable form."""
    learners = Learners(PostgresLearnerStore(app_database))
    (first_id, first_key), (second_id, second_key) = await learners.register(), await learners.register()

    found = [await learners.authenticate(key) for key in (first_key, second_key, "no-such-key")]

    with postgres.owner_connection() as conn:
        stored = {row[0] for row in conn.execute("SELECT key_sha256 FROM learners").fetchall()}
    assert found == [first_id, second_id, None], f"keys lead to the wrong learners: {found}"
    assert not {first_key, second_key} & stored, "a learner's key is stored in readable form"


@pytest.mark.covers("US-020", "AC-12")
async def test_removing_a_learner_deletes_its_sessions_and_answers_only(
    app_database: PostgresDatabase, embedder: FixedLengthEmbedder, postgres: MigratedDatabase
) -> None:
    """Two learners, each with a session and an answer; the first is removed. Its learner row, session and answer
    must be gone, and the second learner's must all be there. Catches a removal that leaves a learner's answers
    behind (the unit tests' store only forgets the key) and one that deletes more than the learner's own rows."""
    service = await _exams(app_database, embedder, COURSE)
    learners = Learners(PostgresLearnerStore(app_database))
    removed, kept = await _learner(app_database), await _learner(app_database)
    sessions = {learner: await _session(service, learner) for learner in (removed, kept)}
    for learner, session_id in sessions.items():
        await service.answer(learner, session_id, "Q1", ["b"])

    existed = await learners.remove(removed)
    again = await learners.remove(removed)

    with postgres.owner_connection() as conn:
        left = {
            table: [row[0] for row in conn.execute(f"SELECT {column} FROM {table}").fetchall()]  # noqa: S608
            for table, column in (("learners", "learner_id"), ("exam_sessions", "session_id"))
        }
    assert (existed, again) == (True, False), (
        f"the first removal finds the learner, the second does not: {existed}, {again}"
    )
    assert removed not in left["learners"] and kept in left["learners"], f"wrong learners left: {left['learners']}"
    assert left["exam_sessions"] == [sessions[kept]], f"only the kept learner's session may remain: {left}"
    assert _answer_rows(postgres, sessions[removed]) == [], "the removed learner's answers are still stored"
    assert len(_answer_rows(postgres, sessions[kept])) == 1, "the kept learner's answer was deleted too"
