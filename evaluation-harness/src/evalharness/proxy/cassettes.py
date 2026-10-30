"""Recorded replies: one file per request key, so a replay needs no network, no key and no money."""

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class Cassette(BaseModel):
    """One recorded reply: its body exactly as the provider sent it, and the day it was recorded."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    recorded_at: datetime
    body: str  # a string, not parsed JSON: the application receives exactly the bytes sent by the provider.


class CassetteStore:
    """Keeps recordings in one folder, with the request key as the file name."""

    def __init__(self, folder: Path) -> None:
        """Remember the folder and make it, since a local course uses a folder that is not committed."""
        folder.mkdir(parents=True, exist_ok=True)
        self._folder = folder

    def find(self, key: str) -> Cassette | None:
        """Return the recording of ``key``, or ``None`` when that request was never recorded."""
        path = self._path(key)
        if not path.is_file():
            return None
        return Cassette.model_validate_json(path.read_bytes())

    def save(self, key: str, cassette: Cassette) -> None:
        """Write the recording through a temporary file, so a stopped proxy cannot leave half a recording."""
        path = self._path(key)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(cassette.model_dump_json(indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    def _path(self, key: str) -> Path:
        """Return the file of ``key``."""
        return self._folder / f"{key}.json"
