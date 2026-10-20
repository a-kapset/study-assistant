"""The running embedding server, called through the app's own client, against vectors of the model's reference code.

Needs the stack's embedding server (``docker compose up``) and runs with ``pytest -m embedding_server``. Without the
server the test fails with a connection error; it is never skipped, so a missing server cannot look like a green run.
"""

import json
import math
import os
from pathlib import Path

import httpx2
import pytest
from pydantic import HttpUrl

from study_assistant.embed.remote import RemoteEmbedder
from study_assistant.settings import EmbeddingServiceSettings, EmbeddingSettings

pytestmark = [pytest.mark.anyio, pytest.mark.embedding_server]

# Vectors made once by sentence-transformers on the same model revision, outside the app (the texts are invented).
# They include a Russian text and one longer than the model's limit of 512 tokens, which must be cut the same way.
REFERENCE = json.loads((Path(__file__).parent / "data" / "embedder_reference.json").read_text(encoding="utf-8"))
# Equal vectors give 1.0. The tolerance only allows rounding differences between runtimes; a lost or swapped prefix,
# OpenAI token ids sent instead of text, or other model files give far less.
MIN_COSINE = 0.9999
# The server's port on this machine, the same variable the stack's .env sets.
PORT = os.environ.get("EMBEDDER_PUBLISHED_PORT", "8081")


@pytest.fixture
def anyio_backend() -> str:
    """Run the test on asyncio, the loop the app server uses."""
    return "asyncio"


def _embedder(http_client: httpx2.AsyncClient) -> RemoteEmbedder:
    """The app's client for the reference model and its prefixes, pointed at the server published on this machine."""
    embedding = EmbeddingSettings(
        model=REFERENCE["model"],
        revision=REFERENCE["revision"],
        dimension=384,
        passage_prefix="passage: ",
        query_prefix="query: ",
    )
    service = EmbeddingServiceSettings(url=HttpUrl(f"http://127.0.0.1:{PORT}"))
    return RemoteEmbedder.from_settings(embedding, service, http_client)


def _cosine(first: list[float], second: list[float]) -> float:
    """How close two vectors point: 1.0 for the same direction."""
    dot = sum(a * b for a, b in zip(first, second, strict=True))
    return dot / math.sqrt(sum(a * a for a in first) * sum(b * b for b in second))


@pytest.mark.covers("US-004", "AC-21")
async def test_server_vectors_match_the_reference_implementation() -> None:
    """Embeds the reference passages in one call and the reference query, and expects each vector to match.

    Catches everything between our settings and the model that changes vectors while every call still succeeds:
    a lost or swapped prefix, token ids sent instead of text, a long text cut in another place, other model files in
    the server's folder, and an upgrade of the server image or of the client libraries. None of these is part of the
    space id, so their vectors would otherwise be stored and compared as if nothing had changed.
    """
    async with httpx2.AsyncClient() as http_client:
        embedder = _embedder(http_client)
        passages = await embedder.aembed_documents([item["text"] for item in REFERENCE["passages"]])
        queries = [await embedder.aembed_query(item["text"]) for item in REFERENCE["queries"]]
    expected = [item["vector"] for item in REFERENCE["passages"] + REFERENCE["queries"]]

    cosines = [_cosine(got, want) for got, want in zip(passages + queries, expected, strict=True)]
    assert min(cosines) >= MIN_COSINE, cosines
