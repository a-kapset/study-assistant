"""API tests for asking a question about a course on the running stack, with the real search and the LLM proxy."""

import uuid
from typing import cast

import pytest
from pydantic import JsonValue

from evalharness.proxy.ledger import LedgerLine, read_ledger
from evalharness.settings import HarnessSettings
from evalharness.transport import SutTransport

# A question the demo course (a board game) has nothing about. Its closest demo passage scored 0.737 on 2026-10-28,
# well below the stack's lowest accepted score of 0.83, so search finds nothing close enough.
OFF_TOPIC_QUESTION = "Who wrote the novel War and Peace?"
# The fixed text the app answers with when search finds nothing. The model writes its own words, so this text shows
# that the app declined before any model call, even when the model would have declined too.
# (NO_PASSAGES_ANSWER in app/src/study_assistant/answer/service.py)
APP_DECLINE = "The course material has no passage close enough to this question, so it is not answered."
# A question the demo course answers: its best passage scored 0.852 on 2026-10-28, so the app calls the model.
ON_TOPIC_QUESTION = "How many barges does each player receive?"
# The passage the recorded reply to that question cites; a passage id comes from its text, so it stays the same.
RECORDED_CITATION = "7f172213d26adc5cd3a132bb78bd6f3ec667520f8b627d7664aa22023facd372"
# The address of the proxy inside the stack (evaluation-harness/compose.proxy.yaml).
PROXY_URL = "http://llm-proxy:8100/v1"


async def _require_the_proxy(sut: SutTransport) -> None:
    """Fail unless the app calls the model through the proxy: without it an empty ledger would prove nothing."""
    config = await sut.request("GET", "/v1/config")
    settings = cast(dict[str, dict[str, JsonValue]], config.body_json)["settings"]
    base_url = cast(dict[str, JsonValue], settings["openai"])["base_url"]
    assert base_url == PROXY_URL, f"start the stack with evaluation-harness/compose.proxy.yaml, the app uses {base_url}"


def _ledger_lines(settings: HarnessSettings, request_id: str) -> list[LedgerLine]:
    """Return the ledger lines of one request of the app."""
    return [line for line in read_ledger(settings.proxy_ledger_file) if line.request_id == request_id]


@pytest.mark.covers("US-009", "AC-5")
async def test_question_outside_the_course_is_declined_without_a_model_call(
    sut: SutTransport, settings: HarnessSettings
) -> None:
    """Asks the demo course a question it has nothing about and expects no model call and the app's own decline.

    What it checks: on the running stack, the ledger of the proxy has no line for this request, and the app answers
    with its fixed decline text.
    What a result tells: a pass means the real embedding server, the stored vectors, the stack's lowest score and
    the grounded mode work together so that such a question never reaches the model. A ledger line means a call was
    made: look at the stack's search settings (lowest score, embedding model) or the ask mode.
    Why it is worth having: the app's unit test proves the decline on a fake search that returns "empty" by
    itself; only the running stack shows that a real off-topic question becomes "empty", and only the proxy, which
    the app does not control, proves that the model was not called. A changed threshold or embedding model would
    send such questions to the model, which costs money and answers from unrelated passages.
    """
    await _require_the_proxy(sut)

    observation = await sut.request("POST", "/v1/ask", json={"course_id": "demo", "question": OFF_TOPIC_QUESTION})

    lines = _ledger_lines(settings, observation.request_id)
    assert lines == [], f"the model was called for a question the app must decline: {lines}"
    assert observation.status_code == 200, f"ask failed: {observation.body_text}"
    result = cast(dict[str, JsonValue], observation.body_json)
    assert (result["abstained"], result["answer"]) == (True, APP_DECLINE), (
        f"expected the app's own decline, which it gives without a model call, got {result}"
    )


@pytest.mark.covers("US-010", "AC-1", "AC-3")
@pytest.mark.covers("US-002", "AC-7")
async def test_a_call_without_a_recording_leaves_one_ledger_line_and_a_failed_answer(
    sut: SutTransport, settings: HarnessSettings
) -> None:
    """Asks a question that reaches the model and has no recording, with a new id in the text so none can exist.

    What it checks: the app answers with its 502 envelope, and the ledger has exactly one line with the id of this
    request, marked as a miss.
    What a result tells: a pass means the whole chain works on the running stack: the app sends its request id,
    the call goes through the proxy, and a miss is seen once. No line points to the id not being sent or the app
    not using the proxy; more than one line points to a miss status that the client repeats.
    Why it is worth having: the proxy's unit tests use a fake client, so they cannot show that the app sends the
    id or that the real client does not repeat a miss. It also proves that the ledger sees calls, which the empty
    ledger of the decline test relies on.
    """
    await _require_the_proxy(sut)
    question = f"{ON_TOPIC_QUESTION} (case {uuid.uuid4().hex})"

    observation = await sut.request("POST", "/v1/ask", json={"course_id": "demo", "question": question})

    assert observation.status_code == 502, f"expected the failed answer of a miss: {observation.body_text}"
    lines = _ledger_lines(settings, observation.request_id)
    assert [line.source for line in lines] == ["miss"], f"expected one miss for this request id, got {lines}"


@pytest.mark.covers("US-010", "AC-2")
async def test_a_recorded_question_is_answered_from_the_recording(sut: SutTransport, settings: HarnessSettings) -> None:
    """Asks the demo question whose real reply is recorded and expects that reply, without a model call.

    What it checks: the ledger has one line for this request, served from the recording, and the app turns the
    recorded reply into a successful answer that cites the passage the model chose.
    What a result tells: a pass means the request the app sends today is the one that was recorded, and the app
    still reads a real reply of the provider. A miss in the ledger means the request changed (prompt, reply schema,
    passages found, or a library update): a defect if nobody meant it, otherwise the recording is out of date. A
    wrong citation points to how the app maps the model's passage numbers to passages.
    Why it is worth having: the proxy's unit tests use a reply built by hand, so only this test runs a real reply
    through the app, at no cost and on every push. Without it a recording could go out of date and nobody would know.
    It says nothing about how good the answer is: the reply is fixed.
    """
    await _require_the_proxy(sut)

    observation = await sut.request("POST", "/v1/ask", json={"course_id": "demo", "question": ON_TOPIC_QUESTION})

    lines = _ledger_lines(settings, observation.request_id)
    assert [line.source for line in lines] == ["cassette"], f"the request no longer matches its recording: {lines}"
    assert observation.status_code == 200, f"the app could not use the recorded reply: {observation.body_text}"
    result = cast(dict[str, JsonValue], observation.body_json)
    citations = cast(list[dict[str, JsonValue]], result["citations"])
    assert [citation["chunk_id"] for citation in citations] == [RECORDED_CITATION], f"wrong passage cited: {citations}"
