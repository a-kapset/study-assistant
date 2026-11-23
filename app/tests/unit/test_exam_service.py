"""Practice sessions on an in-memory store: which questions a session gets, how an answer is checked and counted,
when a session is passed, and what a learner's progress sums.

The store is a fake with the rules of the database tables; those rules themselves are checked against a real
PostgreSQL in ``tests/integration/test_exam_sql.py``.
"""

import random
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import pytest

from study_assistant.courses.models import ExamSettings
from study_assistant.exams.bank import BankItem, BankOption, BankSource
from study_assistant.exams.service import (
    AlreadyAnsweredError,
    ExamService,
    ItemNotInSessionError,
    QuestionBankEmptyError,
    SessionNotFoundError,
    UnknownObjectiveError,
    UnknownOptionError,
    normalise_selection,
    passed,
)

if TYPE_CHECKING:
    from .conftest import FakeExamStore

pytestmark = pytest.mark.anyio

COURSE = "river_board"
OTHER_COURSE = "hill_board"
SETTINGS = ExamSettings(pass_mark_percent=65)
LEARNER = uuid4()
OTHER_LEARNER = uuid4()


def _item(item_id: str, objective_id: str = "OBJ-01", keys: tuple[str, ...] = ("b",)) -> BankItem:
    """One question of an invented course with the options a to d."""
    return BankItem(
        id=item_id,
        stem=f"Question {item_id} about the river board.",
        options=tuple(BankOption(id=option_id, text=f"Answer {option_id}") for option_id in "abcd"),
        keys=keys,
        objective_id=objective_id,
        source=BankSource(document="rules_questions.pdf", version="1", question=item_id),
    )


# Four questions of objective 1 (the last asks for two options) and two of objective 2.
BANK = (
    _item("Q1"),
    _item("Q2"),
    _item("Q3"),
    _item("Q4", keys=("a", "c")),
    _item("Q5", objective_id="OBJ-02"),
    _item("Q6", objective_id="OBJ-02"),
)


async def _service(store: FakeExamStore, seed: int = 7) -> ExamService:
    """A service on ``store`` with the bank loaded for the course, picking questions with a seeded generator."""
    service = ExamService(store, random.Random(seed))
    await service.load_bank(COURSE, BANK)
    return service


async def _session(service: ExamService, learner: UUID = LEARNER, objective_id: str | None = None) -> UUID:
    """Start a session over every question in scope and return its id."""
    created = await service.create_session(learner, COURSE, SETTINGS, objective_id, None)
    return created.session_id


@pytest.mark.covers("US-020", "AC-5")
@pytest.mark.parametrize("sent", [["b"], [" b "], ["B"], ["\tB\n"], ["b", " B"]])
def test_spaces_and_letter_case_around_an_option_id_are_ignored(sent: list[str]) -> None:
    """Guards the repair of an answer's form: without it " B " would be refused or, worse, counted as wrong."""
    chosen = normalise_selection(sent, ["a", "b", "c"])

    assert chosen == frozenset({"b"}), f"{sent!r} must mean option b, got {sorted(chosen)}"


@pytest.mark.covers("US-020", "AC-5")
@pytest.mark.parametrize("sent", [["e"], [""], ["b or c"], ["b", "x"], ["bb"]])
def test_an_option_the_question_does_not_have_is_refused(sent: list[str]) -> None:
    """Guards against guessing: an id outside the question's options must stop the answer, not be dropped or
    matched to something near it."""
    with pytest.raises(UnknownOptionError):
        normalise_selection(sent, ["a", "b", "c"])


@pytest.mark.covers("US-020", "AC-5")
def test_an_id_that_fits_two_options_when_case_is_ignored_is_refused() -> None:
    """A question whose options "Ab" and "aB" differ by case only: an id written exactly is taken as written, but
    "ab" fits both and would have to be guessed. Catches a repair that picks the first match."""
    assert normalise_selection(["aB"], ["Ab", "aB"]) == frozenset({"aB"}), "an exact id must win over case folding"
    with pytest.raises(UnknownOptionError):
        normalise_selection(["ab"], ["Ab", "aB"])


@pytest.mark.covers("US-020", "AC-7")
@pytest.mark.parametrize(
    ("correct", "total", "mark", "expected"),
    [(26, 40, 65, True), (25, 40, 65, False), (13, 20, 65, True), (2, 3, 67, False), (3, 3, 100, True)],
)
def test_pass_mark_is_reached_exactly_at_the_boundary(correct: int, total: int, mark: int, expected: bool) -> None:
    """Guards the comparison with the pass mark: an off-by-one (">" for ">=") or a rounded division would flip
    the verdict of a learner who is exactly on the mark or just under it."""
    verdict = passed(correct, total, mark)

    assert verdict is expected, f"{correct} of {total} at {mark} percent must give passed={expected}"


@pytest.mark.covers("US-020", "AC-3")
async def test_session_without_a_size_takes_the_course_session_size(fake_exam_store: FakeExamStore) -> None:
    """A bank of six questions and a course session size of four: the session must hold four different questions
    of the bank. Catches a session that ignores the course's size or repeats a question."""
    service = await _service(fake_exam_store)

    created = await service.create_session(
        LEARNER, COURSE, ExamSettings(pass_mark_percent=65, session_size=4), None, None
    )

    _, _, item_ids = fake_exam_store.sessions[created.session_id]
    assert created.question_count == 4, f"the course asks for 4 questions, the session reports {created.question_count}"
    assert len(set(item_ids)) == 4 and set(item_ids) <= {item.id for item in BANK}, f"not 4 bank questions: {item_ids}"


@pytest.mark.covers("US-020", "AC-3")
async def test_requested_size_wins_over_the_course_size_and_never_exceeds_the_scope(
    fake_exam_store: FakeExamStore,
) -> None:
    """Asks for two questions, then for sixty, on a bank of six with a course size of four. Catches a request
    size that is ignored, and a session that fails or pads when more is asked than there is."""
    service = await _service(fake_exam_store)
    settings = ExamSettings(pass_mark_percent=65, session_size=4)

    small = await service.create_session(LEARNER, COURSE, settings, None, 2)
    large = await service.create_session(LEARNER, COURSE, settings, None, 60)

    assert small.question_count == 2, f"2 questions were asked for, got {small.question_count}"
    assert large.question_count == len(BANK), f"only {len(BANK)} questions exist, got {large.question_count}"


@pytest.mark.covers("US-020", "AC-3")
async def test_session_on_an_objective_holds_only_its_questions(fake_exam_store: FakeExamStore) -> None:
    """Catches an objective filter that is accepted but not applied: the session would hold the whole bank."""
    service = await _service(fake_exam_store)

    session_id = await _session(service, objective_id="OBJ-02")

    _, _, item_ids = fake_exam_store.sessions[session_id]
    assert set(item_ids) == {"Q5", "Q6"}, f"a session on OBJ-02 must hold Q5 and Q6, got {item_ids}"


@pytest.mark.covers("US-020", "AC-3")
async def test_session_on_an_objective_no_question_tests_is_refused(fake_exam_store: FakeExamStore) -> None:
    """Catches an empty session created as if it were fine: it would be "finished" with nothing asked."""
    service = await _service(fake_exam_store)

    with pytest.raises(UnknownObjectiveError):
        await _session(service, objective_id="OBJ-99")
    assert not fake_exam_store.sessions, "a refused session must not be stored"


@pytest.mark.covers("US-020", "AC-3")
async def test_session_on_a_course_without_a_loaded_bank_is_refused(fake_exam_store: FakeExamStore) -> None:
    """A course whose bank was never loaded: the refusal must say so instead of creating an empty session."""
    service = ExamService(fake_exam_store)

    with pytest.raises(QuestionBankEmptyError):
        await service.create_session(LEARNER, COURSE, SETTINGS, None, None)


@pytest.mark.covers("US-020", "AC-4")
async def test_next_is_the_first_question_without_an_answer(fake_exam_store: FakeExamStore) -> None:
    """Answers the first question of a session and asks for the next one. Catches a "next" that repeats an
    answered question or skips one."""
    service = await _service(fake_exam_store)
    session_id = await _session(service)
    _, _, item_ids = fake_exam_store.sessions[session_id]

    first = await service.next_question(LEARNER, session_id)
    await service.answer(LEARNER, session_id, item_ids[0], ["a"])
    second = await service.next_question(LEARNER, session_id)

    assert first.question is not None and second.question is not None, "an open session must give a question"
    assert (first.question.item_id, first.position) == (item_ids[0], 1), f"the first question must come first: {first}"
    assert (second.question.item_id, second.position) == (item_ids[1], 2), f"the second must follow: {second}"


@pytest.mark.covers("US-020", "AC-4")
async def test_next_is_finished_when_every_question_has_an_answer(fake_exam_store: FakeExamStore) -> None:
    """Answers both questions of a session. Catches a "next" that fails or starts over at the end."""
    service = await _service(fake_exam_store)
    session_id = await _session(service, objective_id="OBJ-02")
    for item_id in ("Q5", "Q6"):
        await service.answer(LEARNER, session_id, item_id, ["a"])

    reply = await service.next_question(LEARNER, session_id)

    assert reply.status == "finished" and reply.question is None, f"a fully answered session is finished: {reply}"


@pytest.mark.covers("US-020", "AC-5")
@pytest.mark.parametrize(
    ("selected", "expected"),
    [(["a", "c"], True), (["c", "a"], True), (["a"], False), (["a", "b", "c"], False), (["b", "d"], False)],
)
async def test_answer_is_correct_only_when_the_selected_options_equal_the_key(
    fake_exam_store: FakeExamStore, selected: list[str], expected: bool
) -> None:
    """A question whose key is the two options a and c. Catches a comparison by order (a list instead of a set),
    and one that accepts a part of the key or a selection that only contains it."""
    service = await _service(fake_exam_store)
    session_id = await _session(service)

    result = await service.answer(LEARNER, session_id, "Q4", selected)

    assert result.correct is expected, f"selecting {selected} for the key a, c must give correct={expected}"
    assert result.keys == ["a", "c"], f"the reply must name the key a, c, got {result.keys}"


@pytest.mark.covers("US-020", "AC-6")
async def test_the_same_answer_sent_again_is_stored_once(fake_exam_store: FakeExamStore) -> None:
    """Sends one answer three times, the repeats with other spacing and case. Catches a repeated request (a retry
    after a lost reply) that is counted again or refused."""
    service = await _service(fake_exam_store)
    session_id = await _session(service)

    replies = [await service.answer(LEARNER, session_id, "Q1", sent) for sent in (["b"], ["b"], [" B "])]
    result = await service.result(LEARNER, session_id, lambda _course: SETTINGS)

    assert replies[0] == replies[1] == replies[2], f"the same answer must get the same reply: {replies}"
    assert (result.answered, result.correct) == (1, 1), f"one answer must count once, got {result}"


@pytest.mark.covers("US-020", "AC-6")
async def test_a_different_answer_to_an_answered_question_changes_nothing(fake_exam_store: FakeExamStore) -> None:
    """Answers a question wrongly, then sends the right answer. Catches an answer that can be replaced: a learner
    could then correct every wrong answer after seeing the key in the reply."""
    service = await _service(fake_exam_store)
    session_id = await _session(service)
    await service.answer(LEARNER, session_id, "Q1", ["a"])

    with pytest.raises(AlreadyAnsweredError):
        await service.answer(LEARNER, session_id, "Q1", ["b"])

    result = await service.result(LEARNER, session_id, lambda _course: SETTINGS)
    assert (result.answered, result.correct) == (1, 0), f"the first, wrong answer must stand, got {result}"


@pytest.mark.covers("US-020", "AC-5")
async def test_answer_to_a_question_the_session_does_not_hold_is_refused(fake_exam_store: FakeExamStore) -> None:
    """A session on objective 2 gets an answer for a question of objective 1. Catches answers counted for
    questions that were never asked in the session."""
    service = await _service(fake_exam_store)
    session_id = await _session(service, objective_id="OBJ-02")

    with pytest.raises(ItemNotInSessionError):
        await service.answer(LEARNER, session_id, "Q1", ["b"])
    assert not fake_exam_store.answers, "an answer to a question outside the session must not be stored"


@pytest.mark.covers("US-020", "AC-7")
async def test_session_is_judged_only_when_every_question_is_answered(fake_exam_store: FakeExamStore) -> None:
    """Two questions, pass mark 50 percent: after one correct answer the mark is already reached, but the session
    is not over. Catches a verdict given early (and a failed one given before the learner could pass)."""
    service = await _service(fake_exam_store)
    session_id = await _session(service, objective_id="OBJ-02")
    half = ExamSettings(pass_mark_percent=50)

    await service.answer(LEARNER, session_id, "Q5", ["b"])
    open_result = await service.result(LEARNER, session_id, lambda _course: half)
    await service.answer(LEARNER, session_id, "Q6", ["a"])
    final_result = await service.result(LEARNER, session_id, lambda _course: half)

    assert open_result.passed is None, f"an open session is neither passed nor failed, got {open_result}"
    assert (final_result.answered, final_result.correct, final_result.passed) == (2, 1, True), (
        f"1 of 2 at 50 percent is passed once both are answered, got {final_result}"
    )


@pytest.mark.covers("US-020", "AC-7")
async def test_verdict_follows_the_pass_mark_of_the_session_course(fake_exam_store: FakeExamStore) -> None:
    """One finished session with 1 of 2 correct, judged under two courses' marks. Catches a pass mark that is
    fixed in the code, or taken from another course than the session's."""
    service = await _service(fake_exam_store)
    session_id = await _session(service, objective_id="OBJ-02")
    await service.answer(LEARNER, session_id, "Q5", ["b"])
    await service.answer(LEARNER, session_id, "Q6", ["a"])
    marks = {COURSE: ExamSettings(pass_mark_percent=51), OTHER_COURSE: ExamSettings(pass_mark_percent=50)}

    strict = await service.result(LEARNER, session_id, marks.get)
    marks[COURSE] = ExamSettings(pass_mark_percent=50)
    lenient = await service.result(LEARNER, session_id, marks.get)

    assert (strict.pass_mark_percent, strict.passed) == (51, False), f"1 of 2 fails at 51 percent, got {strict}"
    assert (lenient.pass_mark_percent, lenient.passed) == (50, True), f"1 of 2 passes at 50 percent, got {lenient}"


@pytest.mark.covers("US-020", "AC-9")
async def test_another_learner_cannot_read_or_answer_a_session(fake_exam_store: FakeExamStore) -> None:
    """A second learner uses the first learner's session id for every operation. Catches a session looked up by
    its id alone: anyone holding an id could read the questions or answer in another's name."""
    service = await _service(fake_exam_store)
    session_id = await _session(service)

    with pytest.raises(SessionNotFoundError):
        await service.next_question(OTHER_LEARNER, session_id)
    with pytest.raises(SessionNotFoundError):
        await service.answer(OTHER_LEARNER, session_id, "Q1", ["b"])
    with pytest.raises(SessionNotFoundError):
        await service.result(OTHER_LEARNER, session_id, lambda _course: SETTINGS)
    assert not fake_exam_store.answers, "the other learner's answer must not be stored"


@pytest.mark.covers("US-020", "AC-8")
async def test_progress_sums_the_answers_of_all_sessions_per_objective(fake_exam_store: FakeExamStore) -> None:
    """Two sessions of one learner: Q1 right and Q5 wrong in the first, Q1 wrong in the second. Progress must give
    objective 1 two answers with one correct and objective 2 one answer with none correct. Catches progress
    taken from the last session only, and correct answers counted as all answers."""
    service = await _service(fake_exam_store)
    first, second = await _session(service), await _session(service)
    await service.answer(LEARNER, first, "Q1", ["b"])
    await service.answer(LEARNER, first, "Q5", ["a"])
    await service.answer(LEARNER, second, "Q1", ["d"])

    progress = await service.progress(LEARNER, COURSE)

    counts = [(row.objective_id, row.answered, row.correct) for row in progress.objectives]
    assert counts == [("OBJ-01", 2, 1), ("OBJ-02", 1, 0)], f"progress does not equal the sum of the answers: {counts}"


@pytest.mark.covers("US-020", "AC-8")
async def test_an_objective_without_answers_is_absent_from_progress(fake_exam_store: FakeExamStore) -> None:
    """Only objective 1 is answered. Catches an objective never attempted that is reported with zero answers,
    which a reader takes for zero percent."""
    service = await _service(fake_exam_store)
    session_id = await _session(service)
    await service.answer(LEARNER, session_id, "Q1", ["b"])

    progress = await service.progress(LEARNER, COURSE)

    assert [row.objective_id for row in progress.objectives] == ["OBJ-01"], f"only OBJ-01 has data: {progress}"
