"""The LLM proxy: the app calls it instead of the provider, and it logs, records and replays every call."""

import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx2
from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from evalharness.proxy.cassettes import Cassette, CassetteStore
from evalharness.proxy.keys import request_key
from evalharness.proxy.ledger import Ledger, LedgerLine, Source
from evalharness.proxy.prices import call_cost, load_prices
from evalharness.proxy.surfaces import openai_responses as surface
from evalharness.proxy.surfaces.openai_responses import ReplyFacts
from evalharness.settings import ProxySettings


@dataclass(frozen=True)
class _Call:
    """What every ledger line of one model call shares."""

    request_id: str | None
    key: str
    model: str
    started: float


def build_proxy(settings: ProxySettings, transport: httpx2.AsyncBaseTransport | None = None) -> Starlette:
    """Build the proxy from ``settings``; nothing is opened until the server starts.

    Tests pass ``transport`` as a fake provider, so every path runs without a network.
    """
    prices = load_prices()
    ledger = Ledger(settings.ledger_file)
    store = CassetteStore(settings.cassette_dir)
    upstream_url = str(settings.upstream_url).rstrip("/") + surface.UPSTREAM_PATH
    headers = {
        "Authorization": f"Bearer {settings.upstream_api_key.get_secret_value()}",
        "Content-Type": "application/json",
    }
    client = httpx2.AsyncClient(timeout=settings.upstream_timeout_s, transport=transport)

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncGenerator[None]:
        """Close the connections to the provider when the server stops."""
        yield
        await client.aclose()

    def log_call(
        call: _Call, source: Source, status: int, facts: ReplyFacts | None = None, error: str | None = None
    ) -> None:
        """Add the ledger line of one call."""
        usage = facts.usage if facts else None
        ledger.append(
            LedgerLine(
                at=datetime.now(UTC),
                request_id=call.request_id,
                key=call.key,
                source=source,
                status=status,
                model_requested=call.model,
                model_returned=facts.model_returned if facts else None,
                input_tokens=usage.input if usage else None,
                cached_tokens=usage.cached if usage else None,
                output_tokens=usage.output if usage else None,
                reasoning_tokens=usage.reasoning if usage else None,
                latency_ms=(time.perf_counter() - call.started) * 1000,
                cost_usd=call_cost(prices, call.model, usage) if usage else None,
                error=error,
            )
        )

    def successful_reply(call: _Call, source: Source, text: str) -> Response:
        """Log a successful reply and return it; a reply the proxy cannot read still reaches the app."""
        try:
            log_call(call, source, 200, facts=surface.read_reply(text.encode("utf-8")))
        except ValidationError:
            log_call(call, source, 200, error="unreadable_reply")
        return Response(text, media_type="application/json")

    async def handle_call(request: Request) -> Response:
        """Answer one model call from a recording, or from the provider when the mode allows it."""
        started = time.perf_counter()
        body = await request.body()
        key = request_key(surface.SURFACE, body)
        call = _Call(request.headers.get("X-Request-ID"), key, surface.requested_model(body), started)
        cassette = store.find(key)
        if cassette is not None:
            return successful_reply(call, "cassette", cassette.body)
        if settings.mode == "replay":
            log_call(call, "miss", 404, error="replay_miss")
            return JSONResponse({"error": {"type": "replay_miss"}}, status_code=404)
        try:
            reply = await client.post(upstream_url, content=body, headers=headers)
        except httpx2.HTTPError as error:
            log_call(call, "upstream", 502, error=type(error).__name__)
            return JSONResponse({"error": {"type": "upstream_unreachable"}}, status_code=502)
        if reply.status_code != 200:
            log_call(call, "upstream", reply.status_code)
            return Response(reply.content, status_code=reply.status_code, media_type="application/json")
        # Kept before anything reads it: the reply is paid for, and a repeat of this call must not pay again.
        store.save(key, Cassette(recorded_at=datetime.now(UTC), body=reply.text))
        return successful_reply(call, "upstream", reply.text)

    async def health(_request: Request) -> Response:
        """Answer the stack's start-up check."""
        return JSONResponse({"status": "ok", "mode": settings.mode})

    routes = [Route(surface.ROUTE, handle_call, methods=["POST"]), Route("/health", health)]
    return Starlette(routes=routes, lifespan=lifespan)
