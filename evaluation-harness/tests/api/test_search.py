"""API tests for course search on the running stack, with the real embedding server and the stored vectors."""

from typing import cast

import pytest
from pydantic import JsonValue

from evalharness.transport import SutTransport

# A question whose answer the demo course states in one place: Unit 1 says that in a game for three players each
# player receives four barges (courses/demo/vardolin_rules.md). Its section id is the unit number.
QUESTION = "How many barges does each player receive in a game for three players?"
ANSWER_SECTION = "1"


@pytest.mark.covers("US-006", "AC-1")
async def test_question_from_the_demo_text_finds_its_section_first(sut: SutTransport) -> None:
    """Asks the demo course a question that Unit 1 answers and expects "ok" with section 1 first.

    The only search check with the real embedding server and stored vectors; it runs in CI. A failure shows the
    broken part: an error code, "empty" (lowest score too high) or another section first (wrong order or space).
    It is not a quality measure, and a new model may need a new expected section.
    """

    observation = await sut.request("POST", "/v1/search", json={"course_id": "demo", "query": QUESTION})

    assert observation.status_code == 200, f"search failed: {observation.body_text}"
    result = cast(dict[str, JsonValue], observation.body_json)
    hits = cast(list[dict[str, JsonValue]], result["hits"])
    first = hits[0]["section_id"] if hits else None
    assert (result["status"], first) == ("ok", ANSWER_SECTION), (
        f"expected status ok with section {ANSWER_SECTION} first, got {result['status']} with section {first}"
    )
