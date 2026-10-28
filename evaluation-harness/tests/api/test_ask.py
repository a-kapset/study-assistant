"""API tests for asking a question about a course on the running stack, with the real search behind the answer."""

from typing import cast

import pytest
from pydantic import JsonValue

from evalharness.transport import SutTransport

# A question the demo course (a board game) has nothing about. Its closest demo passage scored 0.737 on 2026-10-28,
# well below the stack's lowest accepted score of 0.83, so search finds nothing close enough.
OFF_TOPIC_QUESTION = "Who wrote the novel War and Peace?"
# The fixed text the app answers with when search finds nothing. The model writes its own words, so this text shows
# that the app declined before any model call, even when the model would have declined too.
# (NO_PASSAGES_ANSWER in app/src/study_assistant/answer/service.py)
APP_DECLINE = "The course material has no passage close enough to this question, so it is not answered."


@pytest.mark.covers("US-009", "AC-5")
async def test_question_outside_the_course_is_declined_without_a_model_call(sut: SutTransport) -> None:
    """Asks the demo course a question it has nothing about and expects 200 with the app's own decline.

    What it checks: on the running stack, a question outside the course gets no passage above the lowest accepted
    score, and the app answers with its fixed decline text instead of calling the model.
    What a result tells: a pass means the real embedding server, the stored vectors, the stack's lowest score and
    the grounded mode work together so that such a question never reaches the model. A failure points to the
    stack's search settings (lowest score, embedding model) or the ask mode: a 502 means the model was called and
    failed (the placeholder key of CI), another answer text means the model was called and answered.
    Why it is worth having: the app's unit test proves the decline on a fake search that returns "empty" by
    itself; only the running stack shows that a real off-topic question becomes "empty". A changed threshold or
    embedding model would send such questions to the model, which costs money and answers from unrelated passages.
    """

    observation = await sut.request("POST", "/v1/ask", json={"course_id": "demo", "question": OFF_TOPIC_QUESTION})

    assert observation.status_code == 200, f"ask failed, a 502 means the model was called: {observation.body_text}"
    result = cast(dict[str, JsonValue], observation.body_json)
    assert (result["abstained"], result["answer"]) == (True, APP_DECLINE), (
        f"expected the app's own decline, which it gives without a model call, got {result}"
    )
