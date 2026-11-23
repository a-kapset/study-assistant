"""A state machine test of practice sessions on the running stack: two learners start sessions, answer, repeat
and try to change answers in any order, and after every step the app must report what the test's own model says.

Hypothesis draws the sequences of steps. The model is kept in the test: which questions each session got and
whether the first accepted answer to each was correct by the bank file the harness reads itself. The pass mark
comes from the course's manifest. So a result or a progress number that differs from the model is the app's
counting, not a second copy of it. Every response is also checked against the app's own API description.

The app picks the questions of a session at random when it takes fewer than there are, so the test cannot know
in advance which question comes next. What Hypothesis draws is therefore never a question or an option, only
numbers (which session, a pattern of options, a way of writing them); the step turns them into an answer to
whatever question the app shows. The same drawn numbers always mean the same kind of step.

The course is the stack's local course, whose bank and manifest are on this machine. On a failure Hypothesis
prints the shortest sequence of steps it found that still breaks a rule.
"""

from functools import cache
from typing import NamedTuple, cast

import httpx2
import pytest
import schemathesis
import yaml
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule
from pydantic import JsonValue
from schemathesis.specs.openapi.schemas import OpenApiSchema

from evalharness.datasets.question_bank import BankQuestion, read_question_bank
from evalharness.exam_client import SESSIONS_PATH, ExamClient, Learner
from evalharness.oracles.exam_session import SessionModel, expected_progress, expected_result, is_correct
from evalharness.settings import HarnessSettings, load_stack_settings

# Steps per sequence: enough to finish small sessions and to mix two learners' answers.
_STEPS = 15
_NEXT = SESSIONS_PATH + "/{session_id}/next"
_ANSWERS = SESSIONS_PATH + "/{session_id}/answers"
_RESULT = SESSIONS_PATH + "/{session_id}/result"

# The ways one option id is written: as it is, in upper case, with spaces, with both.
_FORMS = (lambda text: text, str.upper, lambda text: f" {text} ", lambda text: f"\t{text.upper()}\n")

# How an answer is written: where to start in the order of its ids, whether one id comes twice, and the form of
# each id by its place.
type Writing = tuple[int, bool, tuple[int, int, int, int]]

# What the steps draw. None of it depends on a question, so a sequence means the same whichever questions the
# app puts into a session.
_which = st.integers(0, 1000)
_pattern = st.integers(1, 255)
_form = st.integers(0, len(_FORMS) - 1)
_writing = st.tuples(st.integers(0, 7), st.booleans(), st.tuples(_form, _form, _form, _form))


class Reference(NamedTuple):
    """What the test knows without asking the app's exam endpoints: the course, its bank, its exam settings and
    the app's API description."""

    settings: HarnessSettings
    admin_key: str
    course_id: str
    bank: dict[str, BankQuestion]
    pass_mark_percent: int
    session_size: int | None
    schema: OpenApiSchema


@cache
def reference() -> Reference:
    """Read the local course's manifest and bank from this machine and the API description from the app, once.

    The manifest and the bank are the files the stack itself mounts, found through the stack's .env file.
    """
    harness = HarnessSettings()
    stack = load_stack_settings(harness.stack_env_file)
    root = harness.stack_env_file.resolve().parent
    manifest_path = root / "courses" / stack.local_course_id / "course.yaml"
    manifest = cast(dict[str, JsonValue], yaml.safe_load(manifest_path.read_text(encoding="utf-8")))
    exam = cast(dict[str, int] | None, manifest.get("exam"))
    assert exam is not None, f"the local course {stack.local_course_id!r} sets no exam block, so it has no sessions"
    sources = cast(list[dict[str, str]], manifest["sources"])
    bank: dict[str, BankQuestion] = {}
    for source in sources:
        if source["role"] == "question_bank":
            bank.update(read_question_bank(root / stack.local_course_sources / source["file"]))
    assert bank, f"the local course {stack.local_course_id!r} lists no question bank"
    folded = {(question.item_id, option.casefold()) for question in bank.values() for option in question.option_ids}
    assert len(folded) == sum(len(question.option_ids) for question in bank.values()), (
        "two options of a question differ by letter case only; the test writes ids in upper case to check the repair"
    )
    schema = schemathesis.openapi.from_url(f"{str(harness.sut_base_url).rstrip('/')}/openapi.json")
    return Reference(
        settings=harness,
        admin_key=stack.admin_api_key.get_secret_value(),
        course_id=stack.local_course_id,
        bank=bank,
        pass_mark_percent=exam["pass_mark_percent"],
        session_size=exam.get("session_size"),
        schema=schema,
    )


def _objectives() -> list[str]:
    """The learning objectives the bank's questions test; read only when a test runs, never at import."""
    return sorted({question.objective_id for question in reference().bank.values()})


def selection(question: BankQuestion, pattern: int) -> frozenset[str]:
    """Turn a drawn number into a set of the question's options: option ``i`` is in when bit ``i`` of the number
    is set; a number that selects none takes one option instead. Every non-empty set of up to eight options can
    come out, the key among them."""
    ids = question.option_ids
    chosen = {option_id for place, option_id in enumerate(ids) if pattern >> place & 1}
    return frozenset(chosen or {ids[pattern % len(ids)]})


def other_selection(question: BankQuestion, pattern: int, taken: frozenset[str]) -> frozenset[str]:
    """Like ``selection``, but never the set ``taken``: when the number gives exactly that set, the first option is
    added or removed, or the second option is taken when nothing would be left."""
    chosen = selection(question, pattern)
    if chosen != taken:
        return chosen
    flipped = chosen ^ {question.option_ids[0]}
    return frozenset(flipped or {question.option_ids[1]})


def as_written(chosen: frozenset[str], writing: Writing) -> list[str]:
    """Write the chosen option ids the way a careless client might: starting anywhere in their order, one of them
    possibly twice, each in one of the four forms. The meaning stays the same."""
    start, twice, forms = writing
    ids = sorted(chosen)
    ids = ids[start % len(ids) :] + ids[: start % len(ids)]
    if twice:
        ids.append(ids[0])
    return [_FORMS[forms[place % len(forms)]](option_id) for place, option_id in enumerate(ids)]


class ExamSessions(RuleBasedStateMachine):
    """Two learners practise on the local course in any order; the app must agree with the test's model.

    What it checks: after every step, each touched session reports exactly the answered and correct counts and the
    verdict the model expects, and each learner's progress equals the sum of that learner's own first answers per
    objective. The steps themselves check each reply: a new session's size, a question shown with the bank's
    options and without its key, whether an answer was correct, the same answer accepted again, a different one
    and an unknown option refused, and another learner's session not found.

    What it tells: a pass says that on the running stack (real database, migration, roles, HTTP) no order of
    these calls makes the app count an answer twice, replace one, judge early or mix learners. A failure prints
    the sequence of calls and the first number that differs, which points to the counting or the lookup involved.

    Why it is worth having: the app's own tests check each rule on its own, on prepared states. A defect that
    needs a sequence (answer, repeat in another form, then another learner answers) or the real stack shows only
    here, and the answer key and the pass mark are read from the course files, not from the app.
    """

    def __init__(self) -> None:
        """Open a client and register two fresh learners for this sequence of steps."""
        super().__init__()
        self.ref = reference()
        self.client = ExamClient(self.ref.settings, self.ref.admin_key).__enter__()
        self.learners: list[Learner] = [self.client.register_learner(), self.client.register_learner()]
        self.sessions: list[SessionModel] = []
        # Sessions whose result was not compared with the model since their last step.
        self.touched: set[str] = set()

    def teardown(self) -> None:
        """Remove both learners, and with them every session and answer of this sequence, then close the client."""
        try:
            for learner in self.learners:
                self.client.remove_learner(learner)
        finally:
            self.client.__exit__(None, None, None)

    def _conforms(self, response: httpx2.Response, path: str, method: str) -> None:
        """Fail when a response is not what the app's API description declares for this operation: a status the
        operation does not list, or a body of another shape than the one listed for that status."""
        declared = self.ref.schema.raw_schema["paths"][path][method.lower()]["responses"]
        assert str(response.status_code) in declared, (
            f"{method} {path} answered {response.status_code}, which its API description does not list "
            f"({sorted(declared)})"
        )
        self.ref.schema[path][method].validate_response(response)

    def _questions_in_scope(self, objective_id: str | None) -> int:
        """Count the bank's questions a session of this scope may hold."""
        return sum(1 for question in self.ref.bank.values() if objective_id in (None, question.objective_id))

    def _answered(self, which: int) -> tuple[SessionModel, str]:
        """Pick, by a drawn number, one answered question among all sessions; the caller's precondition makes
        sure there is one."""
        pairs = [(session, item_id) for session in self.sessions for item_id in session.answers]
        return pairs[which % len(pairs)]

    @rule(
        learner=st.sampled_from((0, 1)),
        objective=st.one_of(st.none(), st.deferred(lambda: st.sampled_from(_objectives()))),
        size=st.one_of(st.none(), st.integers(1, 4)),
    )
    def start_a_session(self, learner: int, objective: str | None, size: int | None) -> None:
        """A learner starts a session on the whole bank or one objective; its size must be what was asked for,
        else the course's session size, and never more than the scope holds."""
        body: dict[str, JsonValue] = {"course_id": self.ref.course_id}
        if objective is not None:
            body["objective_id"] = objective
        if size is not None:
            body["size"] = size
        response = self.client.start_session(self.learners[learner], body)
        self._conforms(response, SESSIONS_PATH, "POST")
        assert response.status_code == 201, f"starting a session with {body} answered {response.status_code}"
        created = response.json()
        in_scope = self._questions_in_scope(objective)
        wanted = size if size is not None else self.ref.session_size
        expected = in_scope if wanted is None else min(wanted, in_scope)
        assert created["question_count"] == expected, (
            f"a session with {body} must hold {expected} questions, the app says {created['question_count']}"
        )
        self.sessions.append(SessionModel(learner, created["session_id"], created["question_count"], objective))
        self.touched.add(created["session_id"])

    @precondition(lambda self: bool(self.sessions))
    @rule(which=_which, right=st.booleans(), pattern=_pattern, writing=_writing)
    def answer_the_next_question(self, which: int, right: bool, pattern: int, writing: Writing) -> None:
        """A learner takes the next question of one of its sessions and answers it, with the key or with another
        set of options, written in any form. The question must be new to the session and inside its scope, shown
        with the bank's options, and the reply must say whether the answer was correct by the bank's key."""
        session = self.sessions[which % len(self.sessions)]
        learner = self.learners[session.learner]
        response = self.client.next_question(learner, session.session_id)
        self._conforms(response, _NEXT, "GET")
        shown = response.json()
        if len(session.answers) == session.question_count:
            assert shown == {"status": "finished", "question_count": session.question_count}, (
                f"every question of the session is answered, but next says {shown}"
            )
            return
        assert shown["status"] == "question", f"the session has questions without an answer, but next says {shown}"
        asked = shown["question"]
        assert set(asked) == {"item_id", "stem", "options", "select"}, (
            f"a question is shown with other fields than its text and options: {sorted(asked)}"
        )
        item_id = asked["item_id"]
        assert item_id in self.ref.bank, f"next shows {item_id!r}, which is not in the course's bank"
        assert item_id not in session.answers, f"next shows {item_id!r}, which the session already answered"
        question = self.ref.bank[item_id]
        assert session.objective_id in (None, question.objective_id), (
            f"a session on {session.objective_id!r} shows {item_id!r} of {question.objective_id!r}"
        )
        assert [option["id"] for option in asked["options"]] == list(question.option_ids), (
            f"{item_id!r} is shown with other options than the bank has"
        )
        assert asked["select"] == len(question.keys), f"{item_id!r} asks to select {asked['select']} options"
        chosen = question.keys if right else selection(question, pattern)
        sent = as_written(chosen, writing)
        reply = self.client.answer(learner, session.session_id, item_id, sent)
        self._conforms(reply, _ANSWERS, "POST")
        correct = is_correct(chosen, question)
        assert reply.status_code == 200, f"answering {item_id!r} with {sent} answered {reply.status_code}"
        assert reply.json() == {"item_id": item_id, "correct": correct, "keys": sorted(question.keys)}, (
            f"{item_id!r} has the key {sorted(question.keys)}; {sent} must be correct={correct}, got {reply.json()}"
        )
        session.record(item_id, chosen, correct)
        self.touched.add(session.session_id)

    @precondition(lambda self: any(session.answers for session in self.sessions))
    @rule(which=_which, writing=_writing)
    def send_an_answer_again(self, which: int, writing: Writing) -> None:
        """A learner repeats an answer it already gave, written in another form, as a client does after a lost
        reply. The app must answer as the first time; the model does not change, so a second count shows in the
        result."""
        session, item_id = self._answered(which)
        sent = as_written(session.selected[item_id], writing)
        reply = self.client.answer(self.learners[session.learner], session.session_id, item_id, sent)
        self._conforms(reply, _ANSWERS, "POST")
        keys = sorted(self.ref.bank[item_id].keys)
        assert reply.status_code == 200, f"repeating the answer to {item_id!r} as {sent} answered {reply.status_code}"
        assert reply.json() == {"item_id": item_id, "correct": session.answers[item_id], "keys": keys}, (
            f"the repeated answer to {item_id!r} got another reply than the first: {reply.json()}"
        )
        self.touched.add(session.session_id)

    @precondition(lambda self: any(session.answers for session in self.sessions))
    @rule(which=_which, pattern=_pattern)
    def send_a_different_answer(self, which: int, pattern: int) -> None:
        """A learner tries to replace an answer with another selection. The app must refuse it; the model keeps
        the first answer, so a replaced one shows in the result."""
        session, item_id = self._answered(which)
        first = session.selected[item_id]
        other = sorted(other_selection(self.ref.bank[item_id], pattern, first))
        reply = self.client.answer(self.learners[session.learner], session.session_id, item_id, other)
        self._conforms(reply, _ANSWERS, "POST")
        assert reply.status_code == 409, (
            f"{item_id!r} was answered {sorted(first)}; {other} must be refused, got {reply.status_code}"
        )
        assert reply.json()["error"]["code"] == "already_answered", f"wrong reason: {reply.json()['error']}"
        self.touched.add(session.session_id)

    @precondition(lambda self: bool(self.sessions))
    @rule(which=_which)
    def name_an_option_the_question_does_not_have(self, which: int) -> None:
        """A learner sends an answer that names an option outside the question, next to a real one. The app must
        refuse the whole answer and store nothing; the model does not change."""
        session = self.sessions[which % len(self.sessions)]
        learner = self.learners[session.learner]
        shown = self.client.next_question(learner, session.session_id).json()
        if shown["status"] != "question":
            return
        item_id = shown["question"]["item_id"]
        real = self.ref.bank[item_id].option_ids[0]
        reply = self.client.answer(learner, session.session_id, item_id, [real, "no-such-option"])
        self._conforms(reply, _ANSWERS, "POST")
        assert reply.status_code == 422, f"an unknown option for {item_id!r} must be refused, got {reply.status_code}"
        assert reply.json()["error"]["code"] == "unknown_option", f"wrong reason: {reply.json()['error']}"
        self.touched.add(session.session_id)

    @precondition(lambda self: bool(self.sessions))
    @rule(which=_which)
    def ask_for_another_learners_session(self, which: int) -> None:
        """The other learner asks for the result of a session and tries to answer in it. Both must be answered
        like a session that does not exist, and nothing may be stored."""
        session = self.sessions[which % len(self.sessions)]
        stranger = self.learners[1 - session.learner]
        result = self.client.result(stranger, session.session_id)
        self._conforms(result, _RESULT, "GET")
        item_id = next(iter(self.ref.bank))
        answer = self.client.answer(stranger, session.session_id, item_id, [self.ref.bank[item_id].option_ids[0]])
        for response in (result, answer):
            assert response.status_code == 404 and response.json()["error"]["code"] == "session_not_found", (
                f"another learner's session must look missing, got {response.status_code}: {response.text[:200]}"
            )
        self.touched.add(session.session_id)

    @invariant()
    def results_and_progress_equal_the_model(self) -> None:
        """After every step: each session touched since the last check reports the model's result, and each
        learner's progress is the sum of its own first answers per objective."""
        for session in self.sessions:
            if session.session_id not in self.touched:
                continue
            response = self.client.result(self.learners[session.learner], session.session_id)
            self._conforms(response, _RESULT, "GET")
            expected = expected_result(session, self.ref.pass_mark_percent)
            assert response.json() == expected, (
                f"session of learner {session.learner}: the app reports {response.json()}, "
                f"the answers sent give {expected}"
            )
        self.touched.clear()
        for number, learner in enumerate(self.learners):
            response = self.client.progress(learner, self.ref.course_id)
            self._conforms(response, "/v1/progress", "GET")
            expected_rows = expected_progress(self.sessions, number, self.ref.bank)
            assert response.json() == {"course_id": self.ref.course_id, "objectives": expected_rows}, (
                f"progress of learner {number}: the app reports {response.json()['objectives']}, "
                f"the learner's own answers give {expected_rows}"
            )


ExamSessions.TestCase.settings = settings(
    max_examples=HarnessSettings().stateful_examples,
    stateful_step_count=_STEPS,
    # Every step sends real requests, so there is no time limit per step.
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)

# The name pytest collects; the criteria are the ones the machine's steps and its check after every step prove.
TestExamSessions = pytest.mark.covers("US-020", "AC-3", "AC-4", "AC-5", "AC-6", "AC-7", "AC-8", "AC-9")(
    ExamSessions.TestCase
)
