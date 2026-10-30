"""Tests of the LLM proxy on a fake provider: no network, no key and no money."""

import json
from datetime import date
from pathlib import Path

import httpx2
import pytest
from pydantic import SecretStr
from starlette.testclient import TestClient

from evalharness.proxy.app import build_proxy
from evalharness.proxy.keys import request_key
from evalharness.proxy.ledger import read_ledger
from evalharness.proxy.prices import Price, TokenUsage, call_cost
from evalharness.settings import ProxySettings

REAL_KEY = "sk-real-key-of-the-proxy"
APP_PLACEHOLDER = "key-held-by-the-proxy"
ROUTE = "/v1/responses"
REQUEST = {"model": "gpt-5.4-mini", "input": "How many barges does each player receive?"}
# The shape of a reply as the types of the OpenAI client describe it.
REPLY = {
    "model": "gpt-5.4-mini-2026-03-17",
    "output": [],
    "usage": {
        "input_tokens": 1000,
        "input_tokens_details": {"cached_tokens": 200, "cache_write_tokens": 0},
        "output_tokens": 100,
        "output_tokens_details": {"reasoning_tokens": 40},
        "total_tokens": 1100,
    },
}
REPLY_COST = 0.001065


class FakeProvider:
    """Stands for the model provider: gives the planned replies in turn and remembers every call."""

    def __init__(self, *replies: httpx2.Response | httpx2.TransportError) -> None:
        """Keep the replies or transport errors to give; without any, every call gets the normal reply."""
        self.replies = list(replies)
        self.calls: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        """Remember the call and answer it, or fail the way the network would."""
        self.calls.append(request)
        reply = self.replies.pop(0) if self.replies else httpx2.Response(200, json=REPLY)
        if isinstance(reply, httpx2.TransportError):
            raise reply
        return reply


def settings_for(folder: Path, mode: str) -> ProxySettings:
    """Build proxy settings that keep the recordings and the ledger in ``folder``."""
    return ProxySettings.model_validate(
        {
            "mode": mode,
            "upstream_api_key": SecretStr(REAL_KEY),
            "cassette_dir": folder / "cassettes",
            "ledger_file": folder / "ledger.jsonl",
        }
    )


def call_proxy(
    folder: Path, mode: str, provider: FakeProvider, *bodies: dict[str, str], request_id: str = "case-1"
) -> list[httpx2.Response]:
    """Send ``bodies`` to one proxy in ``mode``, the way the app sends them, and return the replies."""
    headers = {"X-Request-ID": request_id, "Authorization": f"Bearer {APP_PLACEHOLDER}"}
    with TestClient(build_proxy(settings_for(folder, mode), httpx2.MockTransport(provider))) as client:
        return [client.post(ROUTE, json=body, headers=headers) for body in bodies]


@pytest.mark.covers("US-010", "AC-1")
def test_a_call_adds_a_ledger_line_with_tokens_by_type_and_cost(tmp_path: Path) -> None:
    """Sends one new call in record mode and reads the one line it leaves in the ledger.

    What it checks: the line holds the request id of the app, the model asked for and the model that answered,
    the tokens of each type and the cost.
    What it tells: a pass means the numbers of later cost reports come from the right fields of a reply. A failure
    points to the reading of the reply or to the building of the line.
    Why it is worth having: mixed-up fields give a believable wrong cost, and nothing else reads these numbers.
    """
    call_proxy(tmp_path, "record", FakeProvider(), REQUEST)

    [line] = read_ledger(tmp_path / "ledger.jsonl")
    tokens = (line.input_tokens, line.cached_tokens, line.output_tokens, line.reasoning_tokens)
    assert (line.request_id, line.source, line.status) == ("case-1", "upstream", 200), f"wrong call data: {line}"
    assert (line.model_requested, line.model_returned) == ("openai/gpt-5.4-mini", REPLY["model"]), f"models: {line}"
    assert tokens == (1000, 200, 100, 40), f"tokens read from the wrong fields: {tokens}"
    assert line.cost_usd == pytest.approx(REPLY_COST), f"wrong cost: {line.cost_usd}"


@pytest.mark.covers("US-010", "AC-2")
def test_replay_answers_from_the_recording_without_calling_the_provider(tmp_path: Path) -> None:
    """Records one call, then sends it again to a proxy in replay mode.

    What it checks: the second proxy returns the recorded body and never calls the provider.
    What it tells: a pass means a replayed run needs no network, no key and no money. A failure points to the
    search for a recording.
    Why it is worth having: a replay that quietly calls the provider spends money and gives an unstable test.
    """
    [recorded] = call_proxy(tmp_path, "record", FakeProvider(), REQUEST)
    provider = FakeProvider()

    [replayed] = call_proxy(tmp_path, "replay", provider, REQUEST)

    assert replayed.content == recorded.content, f"replay changed the reply: {replayed.text}"
    assert provider.calls == [], f"replay called the provider {len(provider.calls)} times"


@pytest.mark.covers("US-010", "AC-3")
def test_replay_without_a_recording_answers_replay_miss_and_calls_nobody(tmp_path: Path) -> None:
    """Sends a call that was never recorded to a proxy in replay mode.

    What it checks: the reply is 404 with the type "replay_miss", the provider gets no call and the ledger shows it.
    What it tells: a pass means a changed request is seen at once. A failure points to the branch of the mode, or
    to a status the client would repeat: it repeats every status from 500 up, so one miss would become three calls.
    Why it is worth having: a miss is the only signal that a recording no longer matches the app's request.
    """
    provider = FakeProvider()

    [reply] = call_proxy(tmp_path, "replay", provider, REQUEST)

    assert reply.status_code == 404, f"a miss must be a status the client does not repeat, got {reply.status_code}"
    assert reply.json() == {"error": {"type": "replay_miss"}}, f"a miss must be named: {reply.text}"
    assert provider.calls == [], f"a miss called the provider {len(provider.calls)} times"
    [line] = read_ledger(tmp_path / "ledger.jsonl")
    assert (line.source, line.error) == ("miss", "replay_miss"), f"the ledger hides the miss: {line}"


@pytest.mark.covers("US-010", "AC-4")
def test_the_key_follows_the_content_of_the_request_and_not_its_field_order() -> None:
    """Compares the keys of the same body in another field order and of a body with one nested value changed.

    What it checks: the first pair has one key, the second pair has two.
    What it tells: a pass means a recording fits only the request it was made for. A failure points to the
    canonical form in the key.
    Why it is worth having: without it a changed passage could get an old reply, or an updated library that orders
    fields differently would turn every recording into a miss.
    """
    body = {"model": "m", "input": [{"role": "user", "content": "question"}]}
    reordered = {"input": [{"content": "question", "role": "user"}], "model": "m"}
    changed = {"model": "m", "input": [{"role": "user", "content": "another question"}]}

    key = request_key("surface", json.dumps(body).encode())

    assert key == request_key("surface", json.dumps(reordered).encode()), "field order changed the key"
    assert key != request_key("surface", json.dumps(changed).encode()), "a changed value kept the same key"


@pytest.mark.covers("US-010", "AC-5")
def test_record_pays_once_for_the_same_request(tmp_path: Path) -> None:
    """Sends the same call twice to a proxy in record mode.

    What it checks: the provider gets one call and the second answer is marked as coming from the recording.
    What it tells: a pass means a larger question set does not pay again for the old questions. A failure points
    to the order of the checks in the proxy.
    Why it is worth having: double payment shows up nowhere else until the monthly budget is gone.
    """
    provider = FakeProvider()

    call_proxy(tmp_path, "record", provider, REQUEST, REQUEST)

    assert len(provider.calls) == 1, f"the same request reached the provider {len(provider.calls)} times"
    sources = [line.source for line in read_ledger(tmp_path / "ledger.jsonl")]
    assert sources == ["upstream", "cassette"], f"wrong sources in the ledger: {sources}"


@pytest.mark.covers("US-010", "AC-7")
@pytest.mark.parametrize("status", [401, 429, 500])
def test_a_failed_reply_is_not_recorded(status: int, tmp_path: Path) -> None:
    """Gets a failed reply from the provider, then sends the same call again in record mode.

    What it checks: the app gets the provider's status, and the second call goes to the provider again instead of a
    recording. The statuses stand for a wrong key, a rate limit and a fault of the provider.
    What it tells: a pass means only real answers are kept. A failure points to the status check before saving.
    Why it is worth having: recording with the placeholder key of the example file gives a 401, and a kept failure
    would be replayed for ever as if it were the provider's answer.
    """
    provider = FakeProvider(httpx2.Response(status, json={"error": {"message": "failed"}}))

    first, second = call_proxy(tmp_path, "record", provider, REQUEST, REQUEST)

    assert first.status_code == status, f"the provider's status must reach the app, got {first.status_code}"
    assert second.status_code == 200, f"the failure was kept as a recording, got {second.status_code}"
    assert len(provider.calls) == 2, f"expected a new call after the failure, got {len(provider.calls)}"


@pytest.mark.covers("US-010", "AC-1")
def test_a_provider_that_cannot_be_reached_leaves_a_ledger_line(tmp_path: Path) -> None:
    """Sends a call in record mode to a provider that times out.

    What it checks: the app gets 502 with the type "upstream_unreachable", and the ledger names the error.
    What it tells: a pass means a failed call is counted too. A failure points to the handling of network errors.
    Why it is worth having: a call that timed out may still be billed, so the ledger must not stay silent about it.
    """
    provider = FakeProvider(httpx2.ConnectTimeout("timed out"))

    [reply] = call_proxy(tmp_path, "record", provider, REQUEST)

    assert reply.status_code == 502, f"expected a failure the client may repeat, got {reply.status_code}"
    assert reply.json() == {"error": {"type": "upstream_unreachable"}}, f"the failure must be named: {reply.text}"
    [line] = read_ledger(tmp_path / "ledger.jsonl")
    assert (line.source, line.error) == ("upstream", "ConnectTimeout"), f"the ledger hides the failure: {line}"


@pytest.mark.covers("US-010", "AC-6")
def test_the_real_key_goes_to_the_provider_and_into_no_file(tmp_path: Path) -> None:
    """Records one call that arrives with the app's placeholder key.

    What it checks: the provider gets the real key, and neither key is in the recording or in the ledger.
    What it tells: a pass means recordings are safe to commit. A failure points to the headers of the proxy.
    Why it is worth having: recordings go into a public repository, and a key there cannot be taken back.
    """
    provider = FakeProvider()

    call_proxy(tmp_path, "record", provider, REQUEST)

    sent = provider.calls[0].headers["Authorization"]
    assert sent == f"Bearer {REAL_KEY}", f"the provider must get the proxy's key, got {sent}"
    written = "".join(path.read_text(encoding="utf-8") for path in tmp_path.rglob("*") if path.is_file())
    assert REAL_KEY not in written, "the real key was written to a file"
    assert APP_PLACEHOLDER not in written, "a request header was written to a file"


@pytest.mark.covers("US-010", "AC-8")
def test_a_paid_reply_the_proxy_cannot_read_is_kept_and_not_paid_for_again(tmp_path: Path) -> None:
    """Gets a successful reply of an unknown shape, then sends the same call again in record mode.

    What it checks: the app gets the reply, the ledger names the problem, and the provider is called only once.
    What it tells: a pass means a changed reply format costs one call. A failure points to the order of saving and
    reading in the proxy.
    Why it is worth having: if reading came first, the proxy would answer 500, the client would repeat the call
    and every repeat would be paid for.
    """
    unknown = httpx2.Response(200, json={"unexpected": True})
    provider = FakeProvider(unknown)

    first, _ = call_proxy(tmp_path, "record", provider, REQUEST, REQUEST)

    assert first.json() == {"unexpected": True}, f"the reply must reach the app, got {first.text}"
    assert len(provider.calls) == 1, f"a paid reply was lost, the provider got {len(provider.calls)} calls"
    errors = [line.error for line in read_ledger(tmp_path / "ledger.jsonl")]
    assert errors == ["unreadable_reply", "unreadable_reply"], f"the ledger hides the problem: {errors}"


def test_cached_input_costs_its_own_price() -> None:
    """Prices a call whose input came partly from the provider's cache.

    What it checks: cached tokens cost the cached price, the rest of the input the full price, and reasoning tokens
    are not added on top of the output.
    What it tells: a pass means the ledger's cost follows the provider's bill. A failure points to ``call_cost``.
    Why it is worth having: a wrong formula gives a believable number, and budget decisions rest on it.
    """
    prices = {"p/m": Price(input=1.0, cached_input=0.1, output=2.0, checked_on=date(2026, 10, 30))}

    cost = call_cost(prices, "p/m", TokenUsage(input=1_000_000, cached=400_000, output=1_000_000, reasoning=500_000))

    assert cost == pytest.approx(0.6 + 0.04 + 2.0), f"wrong cost: {cost}"


def test_a_model_without_a_price_has_no_cost_rather_than_zero() -> None:
    """Prices a call of a model that is not in the table.

    What it checks: the cost is ``None``, not 0.
    What it tells: a pass means a missing price stays visible. A failure points to ``call_cost``.
    Why it is worth having: a model changed in configuration must not look free in the ledger.
    """
    assert call_cost({}, "p/unknown", TokenUsage(input=10, cached=0, output=10, reasoning=0)) is None


@pytest.mark.parametrize("key", [None, ""])
def test_record_mode_stops_at_start_without_a_key(key: str | None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Builds the settings in record mode with no key and with the empty key Compose passes for an unset variable.

    What it checks: both stop with a message that names the variable.
    What it tells: a pass means a missing key is seen at start. A failure points to the check in the settings.
    Why it is worth having: otherwise the first call fails with a 401 that looks like a fault of the app.
    """
    monkeypatch.delenv("EVAL_PROXY_UPSTREAM_API_KEY", raising=False)
    values = {"mode": "record"} if key is None else {"mode": "record", "upstream_api_key": key}

    with pytest.raises(ValueError, match="EVAL_PROXY_UPSTREAM_API_KEY"):
        ProxySettings.model_validate(values)
