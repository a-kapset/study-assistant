"""API tests for the explanation of a wrongly answered practice question on the running stack, with the real
search, the real question bank and the LLM proxy.

A learner the test registers starts a session on the learning objective of a chosen question of the invented
course ``lumen_moths``, answers the session's questions wrongly up to that question, and asks for its explanation.
The questions are chosen by their shape from the course's committed bank file: the first with one correct option
and the first with several. The wrong answer is fixed by the bank too, so the app sends its model the same request
every time and the proxy answers from a recording. The replies are real replies of the model, recorded once.
"""

import json
from collections.abc import Iterator
from functools import cache
from typing import NamedTuple, cast

import pytest
import yaml
from pydantic import JsonValue

from evalharness.datasets.question_bank import BankQuestion, read_question_bank
from evalharness.exam_client import ExamClient, Learner
from evalharness.oracles.explanation import options_not_named
from evalharness.proxy.ledger import read_ledger
from evalharness.settings import HarnessSettings
from evalharness.transport import REQUEST_ID_HEADER

# The invented course whose material and question bank are committed (courses/lumen_moths), so the recorded
# replies of the model can be committed too and the tests run the same way on every stack that serves it.
COURSE = "lumen_moths"
# Longest wait for one explanation: in replay it takes well under a second; a recording run calls the model.
_EXPLAIN_TIMEOUT_S = 90.0
_SHAPES = ("one correct option", "several correct options")


class Explained(NamedTuple):
    """One explanation the app gave, with what the test knows about the question from the bank file."""

    question: BankQuestion
    selected: tuple[str, ...]
    status_code: int
    body_text: str
    # Who answered the model calls of this request, by the proxy's ledger.
    sources: tuple[str, ...]


@cache
def _local_bank() -> tuple[str, dict[str, BankQuestion]]:
    """Return the id of the invented course and its question bank, read from the course's committed folder."""
    folder = HarnessSettings().stack_env_file.resolve().parent / "courses" / COURSE
    manifest = cast(dict[str, JsonValue], yaml.safe_load((folder / "course.yaml").read_text(encoding="utf-8")))
    bank: dict[str, BankQuestion] = {}
    for source in cast(list[dict[str, str]], manifest["sources"]):
        if source["role"] == "question_bank":
            bank.update(read_question_bank(folder / source["file"]))
    assert bank, f"the course {COURSE!r} lists no question bank, so nothing can be explained"
    return COURSE, bank


def _question_of(shape: str) -> BankQuestion:
    """Return the first question of the bank, in file order, with one correct option or with several."""
    _, bank = _local_bank()
    several = shape == "several correct options"
    found = next((question for question in bank.values() if (len(question.keys) > 1) == several), None)
    assert found is not None, f"the bank of {COURSE} has no question with {shape}"
    return found


def _wrong_selection(question: BankQuestion) -> tuple[str, ...]:
    """Return a wrong answer fixed by the bank: the first option that is not correct, and for a question with
    several correct options also the first correct one, so the answer is partly right."""
    wrong = next(option for option in question.option_ids if option not in question.keys)
    if len(question.keys) == 1:
        return (wrong,)
    right = next(option for option in question.option_ids if option in question.keys)
    return tuple(sorted((right, wrong), key=question.option_ids.index))


@pytest.fixture
def client(settings: HarnessSettings, admin_headers: dict[str, str]) -> Iterator[ExamClient]:
    """Open an exam client for one test and close it afterwards."""
    with ExamClient(settings, admin_headers["X-Admin-Key"]) as exam_client:
        yield exam_client


@pytest.fixture
def learner(client: ExamClient) -> Iterator[Learner]:
    """Register a learner for one test and remove it, with its sessions and answers, afterwards."""
    registered = client.register_learner()
    yield registered
    client.remove_learner(registered)


def _explain_a_wrong_answer(client: ExamClient, learner: Learner, settings: HarnessSettings, shape: str) -> Explained:
    """Answer the chosen question wrongly in a session on its objective and ask for the explanation.

    The session's questions are answered wrongly in the order the app shows them until the chosen one has its
    answer; a session that never shows it is an error of the setup, said as such.
    """
    course_id, bank = _local_bank()
    question = _question_of(shape)
    selected = _wrong_selection(question)
    started = client.start_session(learner, {"course_id": course_id, "objective_id": question.objective_id})
    assert started.status_code == 201, f"the session did not start: {started.status_code} {started.text}"
    session_id = started.json()["session_id"]
    for _ in range(started.json()["question_count"]):
        shown = client.next_question(learner, session_id).json()["question"]["item_id"]
        answer = selected if shown == question.item_id else _wrong_selection(bank[shown])
        answered = client.answer(learner, session_id, shown, list(answer))
        assert answered.status_code == 200 and answered.json()["correct"] is False, (
            f"the setup answer to {shown} must be accepted as wrong: {answered.status_code} {answered.text}"
        )
        if shown == question.item_id:
            break
    else:
        raise AssertionError(f"the session on {question.objective_id} never showed {question.item_id}")
    response = client.explain(learner, session_id, question.item_id, _EXPLAIN_TIMEOUT_S)
    request_id = response.headers.get(REQUEST_ID_HEADER)
    sources = tuple(line.source for line in read_ledger(settings.proxy_ledger_file) if line.request_id == request_id)
    return Explained(question, selected, response.status_code, response.text, sources)


def _explanation_text(explained: Explained) -> str:
    """Return the explanation's text after checking that the app explained from the recorded reply.

    The recording is checked first, so a changed request is reported as such and not as a changed explanation.
    """
    assert explained.sources == ("cassette",), (
        f"what the model receives for {explained.question.item_id} has changed (instructions, passages, question "
        f"block or model), so the recorded explanation no longer applies; record again and read the new one: "
        f"{explained.sources}"
    )
    assert explained.status_code == 200, f"no explanation: {explained.status_code} {explained.body_text}"
    body = cast(dict[str, JsonValue], json.loads(explained.body_text))
    assert body["abstain_reason"] is None, f"the explanation of {explained.question.item_id} declines: {body}"
    return cast(str, body["explanation"])


@pytest.mark.covers("US-021", "AC-3")
@pytest.mark.parametrize("shape", _SHAPES)
async def test_a_wrong_answer_is_explained_from_passages_the_model_cites(
    shape: str, client: ExamClient, learner: Learner, settings: HarnessSettings, proxy_in_front: None
) -> None:
    """Answers a question of the invented course wrongly and asks for its explanation, which the proxy answers from
    the recorded reply of the real model.

    What it checks: the app turns that reply into an explanation that does not decline and returns the passages
    the model was shown, each with its text, at least one of them marked as cited, and no citation of a passage
    that was not shown.
    What a result tells: a pass means the request the app sends today is the recorded one, the real search found
    passages above the stack's lowest score for the question's stem and key texts, and the app reads a real
    structured reply. A miss in the ledger means the request changed (prompt, reply form, passages found, model,
    library): a defect if nobody meant it, otherwise record again. A decline means the search found nothing for
    this question on this stack. No cited passage or an invalid citation points to how the app maps the model's
    passage numbers.
    Why it is worth having: the app's unit tests use a reply built by hand and a search store that returns chosen
    passages; only this test runs a real reply and the real search through the endpoint, at no cost on every
    push. It says nothing about whether the explanation is right: the reply is fixed.
    """
    explained = _explain_a_wrong_answer(client, learner, settings, shape)

    _explanation_text(explained)
    body = cast(dict[str, JsonValue], json.loads(explained.body_text))
    passages = cast(list[dict[str, JsonValue]], body["passages"])
    assert passages and all(passage["text"] for passage in passages), (
        f"the explanation of {explained.question.item_id} came without the passages it was written from: {body}"
    )
    cited = [passage["chunk_id"] for passage in passages if passage["cited"]]
    assert cited and body["invalid_citations"] == 0, (
        f"expected at least one shown passage cited and no citation of a passage not shown, got {len(cited)} "
        f"cited and {body['invalid_citations']} invalid"
    )


@pytest.mark.covers("US-021", "AC-4")
@pytest.mark.parametrize("shape", _SHAPES)
async def test_an_explanation_names_every_correct_option(
    shape: str, client: ExamClient, learner: Learner, settings: HarnessSettings, proxy_in_front: None
) -> None:
    """Answers a question of the invented course wrongly, asks for its explanation and looks in its text for every
    correct option of the question, written as "option" and the option's id. The correct options come from the
    bank file on this machine, not from the app.

    What it checks: the explanation names each correct option, also when the question has several.
    What a result tells: a pass means the recorded reply of the configured model told the learner the right
    answer in the form the app asks for. A failure names the options that are missing: the model ignored the
    form or named only one of several keys. A ledger miss means the request changed: record again and read the
    new explanation.
    Why it is worth having: it is an exact rule, so no judge is needed for it and the judge of explanations is
    never asked what a pattern can answer. The app's unit tests cannot see it: they build the model's reply by
    hand. With the instruction on how to name an option taken out of the app's prompt, the model named only one
    of two correct options this way and the test failed. Its limits: it finds an option that is named, not what
    is said about it, so an explanation that calls a correct option wrong passes; with a wrong key or a wrong
    selection given to the model, the text still named every option and the rule passed. Those defects are left
    to the judge. One reply of one model per question: it does not say how often the model keeps the form.
    """
    explained = _explain_a_wrong_answer(client, learner, settings, shape)

    text = _explanation_text(explained)
    missing = options_not_named(text, explained.question.keys)
    assert missing == [], (
        f"the explanation of {explained.question.item_id} (key {sorted(explained.question.keys)}) never names "
        f"option {missing}: {text!r}"
    )
