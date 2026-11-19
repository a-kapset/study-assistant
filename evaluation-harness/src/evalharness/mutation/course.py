"""The stored passages of the course a mutant run works on: ingested again through the app's admin endpoint, and
named by a hash of their ids.

A setting of how sources are cut into passages changes nothing the checks measure until the course is ingested
again. Whether that happened is read from the database, not from the endpoint's answer: the passage ids are made from
their text, so the hash of the ids changes when the passages change and comes back when they are made as before.
"""

from typing import Protocol

from pydantic import SecretStr

from evalharness.corpus_db import CorpusReader
from evalharness.settings import HarnessSettings, StackSettings
from evalharness.transport import SutTransport

# Longest wait for one ingestion: a course of about 4,000 short passages takes some minutes to embed on a CPU.
_INGEST_TIMEOUT_S = 900.0


class IngestFailed(RuntimeError):
    """The course could not be ingested again; the message holds the app's answer."""


class StoredCourse(Protocol):
    """The stored passages of one course, as far as a mutant run changes them."""

    async def ingest(self) -> None:
        """Ingest the course again with the settings the app now runs with; raise IngestFailed when it fails."""
        ...

    def snapshot(self) -> str:
        """Name the stored passages of the course by a hash of their ids."""
        ...


class AdminIngestedCourse:
    """The stack's local course, ingested through the app's admin endpoint and read from its database."""

    def __init__(self, settings: HarnessSettings, stack: StackSettings, course_id: str, admin_key: SecretStr) -> None:
        """Keep where the app and its database are, the course and the admin key."""
        self._settings = settings.model_copy(update={"request_timeout_s": _INGEST_TIMEOUT_S})
        self._stack = stack
        self._course_id = course_id
        self._admin_key = admin_key

    async def ingest(self) -> None:
        """Call the admin endpoint for the course; anything but 200 raises IngestFailed with the app's answer."""
        async with SutTransport(self._settings) as sut:
            observation = await sut.request(
                "POST",
                f"/v1/admin/courses/{self._course_id}/ingest",
                headers={"X-Admin-Key": self._admin_key.get_secret_value()},
            )
        if observation.status_code != 200:
            reason = observation.transport_error or f"status {observation.status_code}: {observation.body_text[:300]}"
            raise IngestFailed(f"ingesting {self._course_id} failed: {reason}")

    def snapshot(self) -> str:
        """Read the hash of the course's passage ids from the database."""
        with CorpusReader(self._settings, self._stack) as corpus:
            return corpus.snapshot(self._course_id)
