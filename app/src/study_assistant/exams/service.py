"""Practice exams: a learner starts a session on a course's question bank, answers its questions and reads the result.

Everything here is decided by plain rules, with no language model: which questions a session gets, whether an
answer is correct (the selected options equal the question's key), whether the session is passed (the course's
pass mark) and the learner's progress (a sum over the answer log).

An answer is checked after a repair of its form only: spaces around an option id and its letter case are ignored,
so " B " means "b". An id the question does not have is refused, never guessed. A question is answered once per
session: the same answer sent again gives the same reply and stores nothing new, a different one is refused.
"""

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from study_assistant.courses.models import ExamSettings
from study_assistant.exams.bank import BankItem


class QuestionBankEmptyError(Exception):
    """The course has no question a session could ask: its bank is not loaded, or it has none."""


class UnknownObjectiveError(Exception):
    """No question of the course's bank tests the learning objective a session was asked for."""


class SessionNotFoundError(Exception):
    """The learner has no session with this id; a session of another learner looks the same."""


class ItemNotInSessionError(Exception):
    """The session does not hold the question an answer was sent for."""


class UnknownOptionError(Exception):
    """An answer names an option the question does not have."""


class AlreadyAnsweredError(Exception):
    """The question already has a different answer in this session."""


class NotAnsweredError(Exception):
    """The question has no answer in this session yet, so there is nothing to explain."""


class AnsweredCorrectlyError(Exception):
    """The question was answered correctly in this session, so there is no mistake to explain."""


@dataclass(frozen=True)
class StoredOption:
    """One answer option of a stored question."""

    id: str
    text: str


@dataclass(frozen=True)
class StoredItem:
    """A question as a session needs it: what to show, the key to check against and the objective it tests."""

    item_id: str
    stem: str
    options: tuple[StoredOption, ...]
    keys: frozenset[str]
    objective_id: str


@dataclass(frozen=True)
class StoredAnswer:
    """The answer a session holds for one question."""

    selected: frozenset[str]
    correct: bool


@dataclass(frozen=True)
class SessionState:
    """A session as stored: its course, its questions in the order they are asked, and the answers given so far."""

    course_id: str
    items: tuple[StoredItem, ...]
    answers: dict[str, StoredAnswer]


@dataclass(frozen=True)
class WrongAnswer:
    """A question of a session with the wrong answer the session holds for it, as an explanation needs them."""

    course_id: str
    item: StoredItem
    selected: frozenset[str]


@dataclass(frozen=True)
class ObjectiveCount:
    """How many answers a learner gave to questions of one objective, and how many of them were correct."""

    objective_id: str
    answered: int
    correct: int


class ExamStore(Protocol):
    """Where question banks, sessions and answers are stored."""

    async def replace_bank(self, course_id: str, items: Sequence[BankItem]) -> int:
        """Make the stored bank of a course equal to ``items`` and return how many questions it now offers.

        A stored question missing from ``items`` is retired, not deleted, so sessions that hold it stay readable.
        """
        ...

    async def active_items(self, course_id: str) -> list[StoredItem]:
        """Return the questions of a course a new session may ask, in bank order."""
        ...

    async def create_session(self, learner_id: UUID, course_id: str, item_ids: Sequence[str]) -> UUID:
        """Store a session of the learner with these questions in this order and return its id."""
        ...

    async def session(self, learner_id: UUID, session_id: UUID) -> SessionState | None:
        """Return the learner's session, or ``None`` when the learner has no session with this id."""
        ...

    async def save_answer(
        self, session_id: UUID, item_id: str, answer: StoredAnswer, objective_id: str
    ) -> StoredAnswer:
        """Store the answer unless the question already has one, and return the answer the session now holds.

        The returned answer differs from ``answer`` only when another one was stored first.
        """
        ...

    async def progress(self, learner_id: UUID, course_id: str) -> list[ObjectiveCount]:
        """Count the learner's answers in a course per objective, over all sessions, ordered by objective id."""
        ...


class ResponseModel(BaseModel):
    """Base for the replies of the exam endpoints."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SessionCreated(ResponseModel):
    """A new session: its id and how many questions it holds."""

    session_id: UUID
    question_count: int


class QuestionOption(ResponseModel):
    """One answer option as shown to the learner."""

    id: str
    text: str


class Question(ResponseModel):
    """A question as shown to the learner: never with its key."""

    item_id: str
    stem: str
    options: list[QuestionOption]
    # How many options the learner is asked to select, as an exam paper prints it ("select two").
    select: int


class NextQuestion(ResponseModel):
    """The first question of the session without an answer, or "finished" when every question has one."""

    status: Literal["question", "finished"]
    question_count: int
    # The question's place in the session, from 1; absent when the session is finished.
    position: int | None = None
    question: Question | None = None


class AnswerResult(ResponseModel):
    """What the session holds for a question after an answer: whether it was correct, and the key to learn from."""

    item_id: str
    correct: bool
    keys: list[str]


class SessionResult(ResponseModel):
    """Where a session stands. ``passed`` is set only when every question is answered: before that a session is
    neither passed nor failed. Both pass fields are absent for a course that no longer sets a pass mark."""

    question_count: int
    answered: int
    correct: int
    pass_mark_percent: int | None
    passed: bool | None


class ObjectiveProgress(ResponseModel):
    """A learner's answers to questions of one objective."""

    objective_id: str
    answered: int
    correct: int


class Progress(ResponseModel):
    """A learner's progress in a course. An objective without any answer is absent: no data is not zero percent."""

    course_id: str
    objectives: list[ObjectiveProgress]


def normalise_selection(selected: Sequence[str], option_ids: Sequence[str]) -> frozenset[str]:
    """Turn the option ids a learner sent into ids of the question, ignoring spaces around them and letter case.

    The same option named twice counts once. An id the question does not have raises ``UnknownOptionError``, and
    so does one that fits two options of the question when case is ignored: the answer is never guessed.
    """
    exact = set(option_ids)
    by_folded: dict[str, list[str]] = {}
    for option_id in option_ids:
        by_folded.setdefault(option_id.strip().casefold(), []).append(option_id)
    chosen: set[str] = set()
    for raw in selected:
        text = raw.strip()
        if text in exact:
            chosen.add(text)
            continue
        candidates = by_folded.get(text.casefold(), [])
        if len(candidates) != 1:
            raise UnknownOptionError(
                f"the question has the options {list(option_ids)}; one selected id is not among them"
            )
        chosen.add(candidates[0])
    return frozenset(chosen)


def passed(correct: int, question_count: int, pass_mark_percent: int) -> bool:
    """Tell whether ``correct`` of ``question_count`` reaches the pass mark, compared in whole numbers.

    26 of 40 at 65 percent passes and 25 of 40 does not, with no division that could round either way.
    """
    return correct * 100 >= pass_mark_percent * question_count


class ExamService:
    """Runs practice sessions on a store; every session, answer and progress row belongs to one learner."""

    def __init__(self, store: ExamStore, rng: random.Random | None = None) -> None:
        """Keep the store and the source of randomness that picks a session's questions; tests pass a seeded one."""
        self._store = store
        self._rng = rng if rng is not None else random.Random()

    async def load_bank(self, course_id: str, items: Sequence[BankItem]) -> int:
        """Make the stored bank of a course equal to the questions read from its files; return how many it offers."""
        return await self._store.replace_bank(course_id, items)

    async def create_session(
        self,
        learner_id: UUID,
        course_id: str,
        settings: ExamSettings | None,
        objective_id: str | None,
        size: int | None,
    ) -> SessionCreated:
        """Start a session over the whole bank of the course, or over the questions of one learning objective.

        The session gets ``size`` questions, or the course's session size, or every question in scope; never more
        than there are. When fewer are taken than there are, they are picked at random.
        """
        items = await self._store.active_items(course_id)
        if not items:
            raise QuestionBankEmptyError(f"the course {course_id} has no question bank loaded")
        if objective_id is not None:
            items = [item for item in items if item.objective_id == objective_id]
            if not items:
                raise UnknownObjectiveError(f"no question of the course {course_id} tests the objective {objective_id}")
        wanted = size if size is not None else (settings.session_size if settings is not None else None)
        if wanted is not None and wanted < len(items):
            items = self._rng.sample(items, wanted)
        session_id = await self._store.create_session(learner_id, course_id, [item.item_id for item in items])
        return SessionCreated(session_id=session_id, question_count=len(items))

    async def state(self, learner_id: UUID, session_id: UUID) -> SessionState:
        """Return the learner's session; raise ``SessionNotFoundError`` when the learner has none with this id."""
        state = await self._store.session(learner_id, session_id)
        if state is None:
            raise SessionNotFoundError(f"no session {session_id}")
        return state

    async def next_question(self, learner_id: UUID, session_id: UUID) -> NextQuestion:
        """Return the first question of the session that has no answer yet, without its key."""
        state = await self.state(learner_id, session_id)
        for position, item in enumerate(state.items, start=1):
            if item.item_id not in state.answers:
                question = Question(
                    item_id=item.item_id,
                    stem=item.stem,
                    options=[QuestionOption(id=option.id, text=option.text) for option in item.options],
                    select=len(item.keys),
                )
                return NextQuestion(
                    status="question", question_count=len(state.items), position=position, question=question
                )
        return NextQuestion(status="finished", question_count=len(state.items))

    async def answer(self, learner_id: UUID, session_id: UUID, item_id: str, selected: Sequence[str]) -> AnswerResult:
        """Check an answer to one question of the session and store it, once.

        The same answer sent again gets the same reply and stores nothing; a different answer to a question that
        already has one raises ``AlreadyAnsweredError`` and changes nothing.
        """
        state = await self.state(learner_id, session_id)
        item = next((candidate for candidate in state.items if candidate.item_id == item_id), None)
        if item is None:
            raise ItemNotInSessionError(f"the session {session_id} has no question {item_id}")
        chosen = normalise_selection(selected, [option.id for option in item.options])
        # Compared as sets: the order in which the options were named means nothing.
        given = StoredAnswer(selected=chosen, correct=chosen == item.keys)
        stored = await self._store.save_answer(session_id, item_id, given, item.objective_id)
        if stored.selected != chosen:
            raise AlreadyAnsweredError(f"the question {item_id} already has a different answer in this session")
        return AnswerResult(item_id=item_id, correct=stored.correct, keys=sorted(item.keys))

    async def wrong_answer(self, learner_id: UUID, session_id: UUID, item_id: str) -> WrongAnswer:
        """Return a question of the learner's session together with the wrong answer stored for it.

        The answer is the one the session holds, never one the caller names. A question without an answer raises
        ``NotAnsweredError`` and a correctly answered one ``AnsweredCorrectlyError``: neither has a mistake.
        """
        state = await self.state(learner_id, session_id)
        item = next((candidate for candidate in state.items if candidate.item_id == item_id), None)
        if item is None:
            raise ItemNotInSessionError(f"the session {session_id} has no question {item_id}")
        stored = state.answers.get(item_id)
        if stored is None:
            raise NotAnsweredError(f"the question {item_id} has no answer in this session yet")
        if stored.correct:
            raise AnsweredCorrectlyError(f"the question {item_id} was answered correctly in this session")
        return WrongAnswer(course_id=state.course_id, item=item, selected=stored.selected)

    async def result(
        self, learner_id: UUID, session_id: UUID, settings_of: Callable[[str], ExamSettings | None]
    ) -> SessionResult:
        """Count the session's answers and, once every question is answered, judge it by the pass mark of its
        course, which ``settings_of`` gives for a course id."""
        state = await self.state(learner_id, session_id)
        settings = settings_of(state.course_id)
        total = len(state.items)
        answered = len(state.answers)
        correct = sum(1 for answer in state.answers.values() if answer.correct)
        mark = settings.pass_mark_percent if settings is not None else None
        verdict = passed(correct, total, mark) if mark is not None and answered == total else None
        return SessionResult(
            question_count=total, answered=answered, correct=correct, pass_mark_percent=mark, passed=verdict
        )

    async def progress(self, learner_id: UUID, course_id: str) -> Progress:
        """Return the learner's answers in a course per objective, over all sessions."""
        counts = await self._store.progress(learner_id, course_id)
        return Progress(
            course_id=course_id,
            objectives=[
                ObjectiveProgress(objective_id=count.objective_id, answered=count.answered, correct=count.correct)
                for count in counts
            ],
        )
