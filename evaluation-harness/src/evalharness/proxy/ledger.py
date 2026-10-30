"""The record of every model call: one JSON line per call, written by the proxy, so it does not depend on the app."""

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

type Source = Literal["upstream", "cassette", "miss"]


class LedgerLine(BaseModel):
    """One model call as the proxy saw it.

    ``source`` says who answered, and only "upstream" was paid for. ``cost_usd`` is what the reply costs at the
    listed price whoever answered, so a replayed run still shows the cost of each question; it is ``None`` when
    there is no price or no token count. Token fields are ``None`` when the reply reported no counts.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    at: datetime
    request_id: str | None
    key: str
    source: Source
    status: int
    model_requested: str
    model_returned: str | None
    input_tokens: int | None
    cached_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    latency_ms: float
    cost_usd: float | None
    error: str | None


class Ledger:
    """Appends lines to one file; the proxy is a single process, so whole lines never mix."""

    def __init__(self, path: Path) -> None:
        """Remember the file and make its folder, which a fresh clone does not have."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path

    def append(self, line: LedgerLine) -> None:
        """Add one line to the end of the file."""
        with self._path.open("a", encoding="utf-8") as file:
            file.write(line.model_dump_json() + "\n")


def read_ledger(path: Path) -> list[LedgerLine]:
    """Return every line of the file, oldest first, or nothing when no call has been made yet."""
    if not path.is_file():
        return []
    return [LedgerLine.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines()]
