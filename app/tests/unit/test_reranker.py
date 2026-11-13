"""The rerank client against a fake rerank server that answers the way the real one does: best first, each score with
the place of its text in the request, and no more than 32 texts per request."""

import json

import httpx2
import pytest

from study_assistant.retrieval.rerank import RemoteReranker, RerankServiceRequestError, RerankServiceUnavailableError

pytestmark = pytest.mark.anyio

# The most texts the rerank server accepts in one request; it refuses a larger request as a whole.
SERVER_BATCH_LIMIT = 32


class FakeRerankServer:
    """Scores each text by the number written in it ("text 7" scores 7.0) and answers best first, as the real server
    does; refuses more than 32 texts with 422, as the real server does; keeps the texts of every request."""

    def __init__(self) -> None:
        """Start with no requests seen."""
        self.batches: list[list[str]] = []

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        """Keep the request's texts and answer with their scores, highest first."""
        texts: list[str] = json.loads(request.content)["texts"]
        self.batches.append(texts)
        if len(texts) > SERVER_BATCH_LIMIT:
            return httpx2.Response(422, json={"error": "batch size too large", "error_type": "Validation"})
        ranks = [{"index": place, "score": float(text.split()[-1])} for place, text in enumerate(texts)]
        return httpx2.Response(200, json=sorted(ranks, key=lambda rank: -rank["score"]))


def _reranker(server: FakeRerankServer) -> RemoteReranker:
    """A rerank client whose every request is answered by ``server`` instead of a real one."""
    client = httpx2.AsyncClient(transport=httpx2.MockTransport(server.handle))
    return RemoteReranker(client, "http://reranker.invalid", timeout_seconds=5)


@pytest.mark.covers("US-018", "AC-6")
async def test_scores_come_back_in_the_order_the_texts_were_given() -> None:
    """Sends three texts whose scores are 1, 3 and 2; the server answers best first (3, 2, 1) with their places.

    Expects the scores 1, 3, 2: one per text, in the order the texts were given. Catches a client that reads the
    server's list in its own order, which would give each passage the score of another one and reorder the hits
    by nonsense while every request succeeds.
    """
    server = FakeRerankServer()

    scores = await _reranker(server).ascore("How many dice?", ["text 1", "text 3", "text 2"])

    assert scores == [1.0, 3.0, 2.0], f"each text should get its own score in the given order, got {scores}"


@pytest.mark.covers("US-018", "AC-6")
async def test_more_texts_than_one_request_takes_are_sent_in_several_and_keep_their_places() -> None:
    """Sends 40 texts scored 0..39, more than the 32 the server takes in one request.

    Expects requests of at most 32 texts and the 40 scores back in the given order. Catches a client that sends all
    texts at once (the server refuses the whole search as soon as more than 32 passages reach the reranker) and one
    that puts the scores of the second request at the places of the first.
    """
    server = FakeRerankServer()
    texts = [f"text {n}" for n in range(40)]

    scores = await _reranker(server).ascore("How many dice?", texts)

    sizes = [len(batch) for batch in server.batches]
    assert max(sizes) <= SERVER_BATCH_LIMIT, f"a request held more texts than the server takes: {sizes}"
    assert scores == [float(n) for n in range(40)], f"scores lost their places across requests: {scores}"


@pytest.mark.covers("US-018", "AC-7")
@pytest.mark.parametrize(
    ("answer", "error"),
    [
        (httpx2.ConnectError("connection refused"), RerankServiceUnavailableError),
        (httpx2.ReadTimeout("timed out"), RerankServiceUnavailableError),
        (httpx2.Response(500, json={"error": "inference failed", "error_type": "backend"}), RerankServiceRequestError),
    ],
    ids=["unreachable", "too slow", "error status"],
)
async def test_each_failure_of_the_server_is_raised_as_its_own_error(
    answer: httpx2.Response | Exception, error: type[Exception]
) -> None:
    """Lets the server refuse the connection, not answer in time, or answer with status 500.

    Expects ``RerankServiceUnavailableError`` for the first two and ``RerankServiceRequestError`` for the third.
    Catches the client library's own error leaking out, which the endpoints would answer as an app crash (500)
    instead of naming the rerank server, and the two failures mixed up (503 and 502 tell an operator different
    things: start the server, or read its log).
    """

    def reply(request: httpx2.Request) -> httpx2.Response:
        """Answer the request with ``answer``, or fail it with ``answer`` when it is an exception."""
        if isinstance(answer, Exception):
            raise answer
        return answer

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(reply))

    with pytest.raises(error):
        await RemoteReranker(client, "http://reranker.invalid", timeout_seconds=5).ascore("Dice?", ["text 1"])
