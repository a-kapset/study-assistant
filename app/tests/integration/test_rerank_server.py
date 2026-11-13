"""The running rerank server, called through the app's own client.

The unit tests of the client play a server that answers best first and takes at most 32 texts per request; this test
checks both against the real server. Needs the stack's rerank server (``docker compose up``) and runs with
``pytest -m rerank_server``. Without the server the test fails with a connection error; it is never skipped, so a
missing server cannot look like a green run.
"""

import os

import httpx2
import pytest

from study_assistant.retrieval.rerank import RemoteReranker

pytestmark = [pytest.mark.anyio, pytest.mark.rerank_server]

# The server's port on this machine, the same variable the stack's .env sets.
PORT = os.environ.get("RERANKER_PUBLISHED_PORT", "8082")
QUESTION = "How many dice does a player roll at the start of a turn?"
ANSWER = "At the start of a turn each player rolls two dice and moves that many fields."
# More texts than one request takes, with the answer in the second request, far from the first place.
TEXTS = [f"The market of town {n} opens on the third day of the month." for n in range(40)]
ANSWER_PLACE = 35
TEXTS[ANSWER_PLACE] = ANSWER


@pytest.fixture
def anyio_backend() -> str:
    """Run the test on asyncio, the loop the app server uses."""
    return "asyncio"


@pytest.mark.covers("US-018", "AC-6")
async def test_the_real_server_scores_every_text_in_place_and_the_answer_highest() -> None:
    """Sends 40 texts, the one answer among them at place 35, to the server published on this machine.

    Expects 40 scores and the highest at place 35. Catches what the fakes of the unit tests assume and only the real
    server can show: a reply order or a request limit other than the fakes play (the answer's score would land at
    another place, or the server would refuse the request), and a server running other model files or another
    runtime that no longer tells an answer from unrelated text.
    """
    async with httpx2.AsyncClient() as http_client:
        scores = await RemoteReranker(http_client, f"http://127.0.0.1:{PORT}", timeout_seconds=30).ascore(
            QUESTION, TEXTS
        )

    best = max(range(len(scores)), key=lambda place: scores[place])
    assert (len(scores), best) == (len(TEXTS), ANSWER_PLACE), f"{len(scores)} scores, highest at {best}: {scores}"
