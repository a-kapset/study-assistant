"""The practice exam endpoints over HTTP: who may call them, what each answers, and which error a refusal carries.

The app serves a copy of the demo course with a small question bank added, loaded the way an operator loads it:
through the ingestion endpoint. The stores are in-memory fakes.
"""

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from study_assistant.composition import build_app
from study_assistant.settings import CoursesSettings, Settings

if TYPE_CHECKING:
    from .conftest import FakeCorpusStore, FakeDatabase, FakeEmbedder, FakeExamStore, FakeLearnerStore

# The repository's own course folder: app/tests/unit/<this file> -> repository root.
REPO_COURSES = Path(__file__).resolve().parents[3] / "courses"
BANK_FILE = "bank.jsonl"
# Added to the demo manifest: the bank as one more source, and a pass mark of 50 percent.
BANK_SOURCE = f"  - file: {BANK_FILE}\n    role: question_bank\n    authority: primary\n"
EXAM_BLOCK = "exam:\n  pass_mark_percent: {mark}\n"
SESSIONS = "/v1/exams/sessions"


def _record(item_id: str, objective_id: str, keys: list[str]) -> dict[str, object]:
    """One question record of the bank, with the options a to c."""
    return {
        "id": item_id,
        "stem": f"Question {item_id} about Vardolin.",
        "options": [{"id": option_id, "text": f"Answer {option_id}"} for option_id in "abc"],
        "keys": keys,
        "objective_id": objective_id,
        "source": {"document": "vardolin_questions.pdf", "version": "1", "question": item_id},
    }


BANK = [_record("Q1", "OBJ-01-01", ["b"]), _record("Q2", "OBJ-01-01", ["a", "c"]), _record("Q3", "OBJ-02-01", ["a"])]


def _course_root(tmp_path: Path, mark: int = 50, bank_lines: list[str] | None = None) -> Path:
    """Make a folder with the demo course, its manifest extended by a question bank and a pass mark."""
    folder = tmp_path / "demo"
    folder.mkdir()
    for name in ("course.yaml", "vardolin_rules.pdf"):
        (folder / name).write_bytes((REPO_COURSES / "demo" / name).read_bytes())
    with (folder / "course.yaml").open("a", encoding="utf-8", newline="\n") as manifest:
        manifest.write(BANK_SOURCE + EXAM_BLOCK.format(mark=mark))
    lines = bank_lines if bank_lines is not None else [json.dumps(record) for record in BANK]
    (folder / BANK_FILE).write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return tmp_path


class Stack:
    """The app under test with its fakes, an admin key header, and helpers for the steps most tests repeat."""

    def __init__(
        self,
        settings: Settings,
        db: FakeDatabase,
        corpus: FakeCorpusStore,
        embedder: FakeEmbedder,
        exams: FakeExamStore,
        learners: FakeLearnerStore,
        root: Path,
    ) -> None:
        """Build the app serving the course under ``root`` on the given fakes."""
        courses = CoursesSettings(manifest_dir=root, source_dir=root, enabled=("demo",))
        app = build_app(
            settings.model_copy(update={"courses": courses}),
            db,
            corpus,
            embedder,
            exam_store=exams,
            learner_store=learners,
        )
        self.client = TestClient(app)
        self.admin = {"X-Admin-Key": settings.admin.api_key.get_secret_value()}
        self.corpus = corpus
        self.exams = exams
        self.learners = learners

    def ingest(self) -> dict[str, object]:
        """Ingest the course as an operator does and return the report; the bank is loaded by the same call."""
        response = self.client.post("/v1/admin/courses/demo/ingest", headers=self.admin)
        assert response.status_code == 200, f"ingesting the test course failed: {response.text}"
        report: dict[str, object] = response.json()
        return report

    def learner(self) -> dict[str, str]:
        """Register a learner as an operator does and return the header its requests carry."""
        response = self.client.post("/v1/admin/learners", headers=self.admin)
        assert response.status_code == 201, f"registering a learner failed: {response.text}"
        return {"Authorization": f"Bearer {response.json()['api_key']}"}

    def session(self, headers: dict[str, str], **scope: object) -> str:
        """Start a session over the whole bank (or ``scope``) and return its id."""
        response = self.client.post(SESSIONS, json={"course_id": "demo", **scope}, headers=headers)
        assert response.status_code == 201, f"starting a session failed: {response.text}"
        session_id: str = response.json()["session_id"]
        return session_id

    def answer(self, headers: dict[str, str], session_id: str, item_id: str, selected: list[str]) -> int:
        """Send an answer and return the status code."""
        body = {"item_id": item_id, "selected": selected}
        return self.client.post(f"{SESSIONS}/{session_id}/answers", json=body, headers=headers).status_code


@pytest.fixture
def stack(
    settings: Settings,
    fake_db: FakeDatabase,
    fake_store: FakeCorpusStore,
    fake_embedder: FakeEmbedder,
    fake_exam_store: FakeExamStore,
    fake_learner_store: FakeLearnerStore,
    tmp_path: Path,
) -> Stack:
    """The app serving the demo course with a bank of three questions and a pass mark of 50 percent."""
    root = _course_root(tmp_path)
    return Stack(settings, fake_db, fake_store, fake_embedder, fake_exam_store, fake_learner_store, root)


def _error_code(response_json: dict[str, dict[str, str]]) -> str:
    """The code of an error envelope."""
    return response_json["error"]["code"]


SOME_SESSION = "00000000-0000-4000-8000-000000000000"
LEARNER_CALLS = [
    ("post", SESSIONS, {"course_id": "demo"}),
    ("get", f"{SESSIONS}/{SOME_SESSION}/next", None),
    ("post", f"{SESSIONS}/{SOME_SESSION}/answers", {"item_id": "Q1", "selected": ["b"]}),
    ("get", f"{SESSIONS}/{SOME_SESSION}/result", None),
    ("get", "/v1/progress?course_id=demo", None),
]


@pytest.mark.covers("US-020", "AC-1")
@pytest.mark.parametrize(("method", "path", "body"), LEARNER_CALLS)
@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer not-a-learner-key"}, {"Authorization": "Basic abc"}])
def test_exam_endpoints_refuse_a_caller_without_a_learner_key(
    stack: Stack, method: str, path: str, body: dict[str, object] | None, headers: dict[str, str]
) -> None:
    """Calls every learner endpoint without a key, with a key no learner has and with another scheme. Catches an
    endpoint added without the key check: it would serve sessions to anyone who reaches the port."""
    stack.ingest()

    response = stack.client.request(method, path, json=body, headers=headers)

    assert response.status_code == 401, f"{method} {path} with {headers} must be refused, got {response.status_code}"
    assert response.headers.get("WWW-Authenticate") == "Bearer", "a 401 must say which scheme is expected"
    assert not stack.exams.sessions, "a refused caller must not create a session"


@pytest.mark.covers("US-020", "AC-1")
def test_key_is_checked_before_the_course_is_looked_up(stack: Stack) -> None:
    """Guards the order: a 404 for an unknown course without a key would tell anyone which course ids exist."""
    response = stack.client.post(SESSIONS, json={"course_id": "no_such_course"})

    assert response.status_code == 401, f"an unknown course without a key must be 401, got {response.status_code}"


@pytest.mark.covers("US-020", "AC-2")
def test_registering_a_learner_needs_the_admin_key(stack: Stack) -> None:
    """Catches an open registration: anyone could make themselves a learner."""
    response = stack.client.post("/v1/admin/learners")

    assert response.status_code == 401, f"registration without the admin key must be 401, got {response.status_code}"
    assert not stack.learners.by_hash, "a refused registration must not store a learner"


@pytest.mark.covers("US-020", "AC-2")
def test_only_the_hash_of_a_learner_key_is_stored(stack: Stack) -> None:
    """Registers a learner and looks at what the store was given. Catches a key stored as it is: a copy of the
    database would then open every learner's sessions."""
    key = stack.learner()["Authorization"].removeprefix("Bearer ")

    stored = list(stack.learners.by_hash)

    assert stored == [hashlib.sha256(key.encode()).hexdigest()], "the store must hold the SHA-256 of the key only"


@pytest.mark.covers("US-020", "AC-10")
def test_ingestion_loads_the_question_bank_and_reports_its_size(stack: Stack) -> None:
    """Catches an ingestion that skips the bank file: sessions would find no questions after a clean start."""
    report = stack.ingest()

    assert report["question_bank_items"] == 3, f"the bank has 3 questions, the report says {report}"
    assert list(stack.exams.items["demo"]) == ["Q1", "Q2", "Q3"], "the stored bank must hold the file's questions"


@pytest.mark.covers("US-020", "AC-10")
def test_an_unusable_bank_file_stops_ingestion_before_anything_is_written(
    settings: Settings,
    fake_db: FakeDatabase,
    fake_store: FakeCorpusStore,
    fake_embedder: FakeEmbedder,
    fake_exam_store: FakeExamStore,
    fake_learner_store: FakeLearnerStore,
    tmp_path: Path,
) -> None:
    """The bank's second line names a key that is not an option. The answer must be a 422 naming the file and the
    line, with neither the corpus nor the bank written. Catches a half-done ingestion reported after the fact."""
    broken = json.dumps(_record("Q2", "OBJ-01-01", ["z"]))
    root = _course_root(tmp_path, bank_lines=[json.dumps(BANK[0]), broken])
    stack = Stack(settings, fake_db, fake_store, fake_embedder, fake_exam_store, fake_learner_store, root)

    response = stack.client.post("/v1/admin/courses/demo/ingest", headers=stack.admin)

    assert response.status_code == 422, f"a broken bank must be a 422, got {response.status_code}"
    error = response.json()["error"]
    assert error["code"] == "question_bank_invalid", f"the code must name the bank, got {error['code']}"
    assert BANK_FILE in error["message"] and "line 2" in error["message"], f"file and line missing: {error['message']}"
    assert stack.corpus.writers_opened == 0 and not stack.exams.items, "nothing may be written before the bank is read"


@pytest.mark.covers("US-020", "AC-3")
def test_starting_a_session_answers_its_id_and_size(stack: Stack) -> None:
    """Guards the contract of a new session: 201 with the id and the number of questions, nothing else."""
    stack.ingest()

    response = stack.client.post(SESSIONS, json={"course_id": "demo"}, headers=stack.learner())

    assert response.status_code == 201, f"a new session must be 201, got {response.status_code}"
    body = response.json()
    assert set(body) == {"session_id", "question_count"}, f"unexpected fields: {sorted(body)}"
    assert body["question_count"] == 3, f"the bank has 3 questions, the session reports {body['question_count']}"


@pytest.mark.covers("US-020", "AC-3")
@pytest.mark.parametrize(
    ("ingested", "body", "status", "code"),
    [
        (True, {"course_id": "no_such_course"}, 404, "not_found"),
        (False, {"course_id": "demo"}, 409, "question_bank_empty"),
        (True, {"course_id": "demo", "objective_id": "OBJ-99-99"}, 422, "unknown_objective"),
    ],
)
def test_a_session_that_cannot_start_names_the_reason(
    stack: Stack, ingested: bool, body: dict[str, object], status: int, code: str
) -> None:
    """A course the app does not serve, a course whose bank is not loaded, and an objective without questions.
    Catches service errors that surface as a 500 or under one shared code a client cannot act on."""
    if ingested:
        stack.ingest()

    response = stack.client.post(SESSIONS, json=body, headers=stack.learner())

    assert (response.status_code, _error_code(response.json())) == (status, code), f"got {response.text}"


@pytest.mark.covers("US-020", "AC-4")
def test_next_shows_the_question_without_its_key(stack: Stack) -> None:
    """Compares the whole body of "next" with the expected one. Catches the key, or whether an option is right,
    leaking into the question a learner has not answered yet."""
    stack.ingest()
    learner = stack.learner()
    session_id = stack.session(learner, objective_id="OBJ-02-01")

    response = stack.client.get(f"{SESSIONS}/{session_id}/next", headers=learner)

    assert response.json() == {
        "status": "question",
        "question_count": 1,
        "position": 1,
        "question": {
            "item_id": "Q3",
            "stem": "Question Q3 about Vardolin.",
            "options": [{"id": option_id, "text": f"Answer {option_id}"} for option_id in "abc"],
            "select": 1,
        },
    }, f"the body of next is not the question alone: {response.text}"


@pytest.mark.covers("US-020", "AC-4")
def test_next_of_a_finished_session_says_finished_and_nothing_else(stack: Stack) -> None:
    """Catches empty question fields sent with "finished", which a client would have to tell from a question."""
    stack.ingest()
    learner = stack.learner()
    session_id = stack.session(learner, objective_id="OBJ-02-01")
    stack.answer(learner, session_id, "Q3", ["a"])

    response = stack.client.get(f"{SESSIONS}/{session_id}/next", headers=learner)

    assert response.json() == {"status": "finished", "question_count": 1}, f"got {response.text}"


@pytest.mark.covers("US-020", "AC-5")
def test_an_answer_is_told_whether_it_was_correct_with_the_key(stack: Stack) -> None:
    """Sends a wrong answer with spaces and capitals. The reply must be a 200 saying it was wrong and naming the
    key. Catches a reply that leaves the learner without the right answer to learn from."""
    stack.ingest()
    learner = stack.learner()
    session_id = stack.session(learner)

    response = stack.client.post(
        f"{SESSIONS}/{session_id}/answers", json={"item_id": "Q2", "selected": [" A "]}, headers=learner
    )

    assert response.status_code == 200, f"a valid answer must be 200, got {response.status_code}"
    assert response.json() == {"item_id": "Q2", "correct": False, "keys": ["a", "c"]}, f"got {response.text}"


@pytest.mark.covers("US-020", "AC-5", "AC-6")
@pytest.mark.parametrize(
    ("item_id", "selected", "status", "code"),
    [
        ("Q1", ["z"], 422, "unknown_option"),
        ("Q1", ["a"], 409, "already_answered"),
        ("Q9", ["a"], 404, "item_not_in_session"),
    ],
)
def test_a_refused_answer_names_the_reason(
    stack: Stack, item_id: str, selected: list[str], status: int, code: str
) -> None:
    """After Q1 was answered with b: an option Q1 does not have, another answer to Q1, and a question the session
    does not hold. Catches refusals that surface as a 500 or under a code a client cannot act on."""
    stack.ingest()
    learner = stack.learner()
    session_id = stack.session(learner)
    stack.answer(learner, session_id, "Q1", ["b"])

    response = stack.client.post(
        f"{SESSIONS}/{session_id}/answers", json={"item_id": item_id, "selected": selected}, headers=learner
    )

    assert (response.status_code, _error_code(response.json())) == (status, code), f"got {response.text}"


@pytest.mark.covers("US-020", "AC-6")
def test_the_same_answer_sent_again_is_answered_200_again(stack: Stack) -> None:
    """Catches a retry of the same answer refused like a different one: a client that lost the first reply could
    never learn the outcome."""
    stack.ingest()
    learner = stack.learner()
    session_id = stack.session(learner)

    statuses = [stack.answer(learner, session_id, "Q1", sent) for sent in (["b"], ["B "])]

    assert statuses == [200, 200], f"the same answer twice must be 200 both times, got {statuses}"
    assert len(stack.exams.answers) == 1, f"one answer must be stored, the store holds {len(stack.exams.answers)}"


@pytest.mark.covers("US-020", "AC-7")
@pytest.mark.parametrize(("mark", "verdict"), [(50, True), (51, False)])
def test_result_is_judged_by_the_pass_mark_in_the_course_manifest(
    settings: Settings,
    fake_db: FakeDatabase,
    fake_store: FakeCorpusStore,
    fake_embedder: FakeEmbedder,
    fake_exam_store: FakeExamStore,
    fake_learner_store: FakeLearnerStore,
    tmp_path: Path,
    mark: int,
    verdict: bool,
) -> None:
    """The same answers (1 of 2 correct) under a manifest with pass mark 50 and one with 51. Catches a pass mark
    that does not come from the course's manifest: both apps would give the same verdict."""
    root = _course_root(tmp_path, mark=mark)
    stack = Stack(settings, fake_db, fake_store, fake_embedder, fake_exam_store, fake_learner_store, root)
    stack.ingest()
    learner = stack.learner()
    session_id = stack.session(learner, objective_id="OBJ-01-01")
    stack.answer(learner, session_id, "Q1", ["b"])
    stack.answer(learner, session_id, "Q2", ["a"])

    response = stack.client.get(f"{SESSIONS}/{session_id}/result", headers=learner)

    assert response.json() == {
        "question_count": 2,
        "answered": 2,
        "correct": 1,
        "pass_mark_percent": mark,
        "passed": verdict,
    }, f"1 of 2 at {mark} percent must give passed={verdict}: {response.text}"


@pytest.mark.covers("US-020", "AC-9")
@pytest.mark.parametrize(
    ("method", "suffix", "body"),
    [("get", "next", None), ("post", "answers", {"item_id": "Q1", "selected": ["b"]}), ("get", "result", None)],
)
def test_a_session_of_another_learner_is_answered_like_one_that_does_not_exist(
    stack: Stack, method: str, suffix: str, body: dict[str, object] | None
) -> None:
    """A second learner calls each session endpoint with the first learner's session id, then with an id no
    session has. Both answers must be the same 404, so the id of a real session tells a stranger nothing."""
    stack.ingest()
    session_id = stack.session(stack.learner())
    stranger = stack.learner()

    foreign = stack.client.request(method, f"{SESSIONS}/{session_id}/{suffix}", json=body, headers=stranger)
    missing = stack.client.request(method, f"{SESSIONS}/{SOME_SESSION}/{suffix}", json=body, headers=stranger)

    assert (foreign.status_code, _error_code(foreign.json())) == (404, "session_not_found"), f"got {foreign.text}"
    assert (missing.status_code, _error_code(missing.json())) == (404, "session_not_found"), f"got {missing.text}"


@pytest.mark.covers("US-020", "AC-8")
def test_progress_lists_the_learner_answers_per_objective(stack: Stack) -> None:
    """Guards the contract of progress: the course and, per answered objective, how many answers and how many
    correct. A second learner's answers are in the store too and must not be counted."""
    stack.ingest()
    learner, other = stack.learner(), stack.learner()
    session_id, other_session = stack.session(learner), stack.session(other)
    stack.answer(learner, session_id, "Q1", ["b"])
    stack.answer(learner, session_id, "Q2", ["a"])
    stack.answer(other, other_session, "Q3", ["a"])

    response = stack.client.get("/v1/progress?course_id=demo", headers=learner)

    assert response.json() == {
        "course_id": "demo",
        "objectives": [{"objective_id": "OBJ-01-01", "answered": 2, "correct": 1}],
    }, f"progress is not the learner's own answers per objective: {response.text}"


def test_progress_of_a_course_not_served_answers_404(stack: Stack) -> None:
    """Catches an empty progress list for a mistyped course id, which reads as "nothing answered yet"."""
    response = stack.client.get("/v1/progress?course_id=no_such_course", headers=stack.learner())

    assert response.status_code == 404, f"an unknown course must be 404, got {response.status_code}"


@pytest.mark.covers("US-020", "AC-12")
def test_a_removed_learner_key_opens_nothing_and_a_second_removal_is_404(stack: Stack) -> None:
    """Registers a learner, removes it as an operator and calls an exam endpoint with its key; then removes it
    again. The key must be refused like one no learner ever had, and the second removal must say the learner is
    gone. Catches a removal that answers success but leaves the key usable."""
    registered = stack.client.post("/v1/admin/learners", headers=stack.admin).json()
    learner = {"Authorization": f"Bearer {registered['api_key']}"}
    removal_path = f"/v1/admin/learners/{registered['learner_id']}"

    first = stack.client.delete(removal_path, headers=stack.admin)
    after = stack.client.get("/v1/progress?course_id=demo", headers=learner)
    second = stack.client.delete(removal_path, headers=stack.admin)

    assert first.status_code == 204, f"removing a learner must be 204, got {first.status_code}: {first.text}"
    assert after.status_code == 401, f"the removed learner's key must be refused, got {after.status_code}"
    assert second.status_code == 404, f"removing a learner twice must be 404 the second time, got {second.status_code}"


@pytest.mark.covers("US-020", "AC-12")
def test_removing_a_learner_needs_the_admin_key(stack: Stack) -> None:
    """A learner tries to remove a learner with its own key and without any. Catches an open removal: anyone
    could erase a learner's sessions and progress."""
    registered = stack.client.post("/v1/admin/learners", headers=stack.admin).json()
    removal_path = f"/v1/admin/learners/{registered['learner_id']}"

    statuses = [
        stack.client.delete(removal_path, headers=headers).status_code
        for headers in ({}, {"Authorization": f"Bearer {registered['api_key']}"})
    ]

    assert statuses == [401, 401], f"a removal without the admin key must be 401, got {statuses}"
    assert len(stack.learners.by_hash) == 1, "a refused removal must leave the learner stored"
