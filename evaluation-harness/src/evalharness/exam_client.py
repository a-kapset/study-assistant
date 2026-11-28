"""A plain client of the app's practice exam endpoints, for checks that walk sessions step by step.

It is synchronous on purpose: a state machine test runs its steps one after another, outside any event loop.
Every method returns the raw response and judges nothing, so the check that called it decides what is right.
Learners are registered and removed through the app's own admin endpoints, the way an operator does it; a check
removes the learners it registered, and their sessions and answers go with them.
"""

from dataclasses import dataclass
from types import TracebackType
from typing import Self

import httpx2
from pydantic import JsonValue

from evalharness.settings import HarnessSettings

ADMIN_KEY_HEADER = "X-Admin-Key"
SESSIONS_PATH = "/v1/exams/sessions"
LEARNERS_PATH = "/v1/admin/learners"


@dataclass(frozen=True)
class Learner:
    """A learner the harness registered: its id, for the removal, and the header its requests carry."""

    learner_id: str
    headers: dict[str, str]


class ExamClient:
    """Sends the requests of practice sessions to the system under test; use it as a context manager."""

    def __init__(self, settings: HarnessSettings, admin_key: str) -> None:
        """Prepare the client for the app of ``settings``; the admin key is used only to register and remove
        learners."""
        self._client = httpx2.Client(
            base_url=str(settings.sut_base_url),
            timeout=settings.request_timeout_s,
            # Proxy variables from the environment must not silently reroute requests to the system under test.
            trust_env=False,
        )
        self._admin = {ADMIN_KEY_HEADER: admin_key}

    def __enter__(self) -> Self:
        """Open the connection pool."""
        self._client.__enter__()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        """Close the connection pool."""
        self._client.__exit__(exc_type, exc, tb)

    def register_learner(self) -> Learner:
        """Register a learner as an operator does; a refusal is an error, because no check can run without one."""
        response = self._client.post(LEARNERS_PATH, headers=self._admin)
        if response.status_code != 201:
            raise RuntimeError(f"registering a learner answered {response.status_code}: {response.text[:200]}")
        body = response.json()
        return Learner(learner_id=body["learner_id"], headers={"Authorization": f"Bearer {body['api_key']}"})

    def remove_learner(self, learner: Learner) -> None:
        """Remove a learner with its sessions and answers; a refusal is an error, so no test data stays unnoticed."""
        response = self._client.delete(f"{LEARNERS_PATH}/{learner.learner_id}", headers=self._admin)
        if response.status_code != 204:
            raise RuntimeError(f"removing learner {learner.learner_id} answered {response.status_code}")

    def start_session(self, learner: Learner, body: dict[str, JsonValue]) -> httpx2.Response:
        """Ask for a new session; ``body`` names the course and, optionally, an objective and a size."""
        return self._client.post(SESSIONS_PATH, json=body, headers=learner.headers)

    def next_question(self, learner: Learner, session_id: str) -> httpx2.Response:
        """Ask for the session's first question without an answer."""
        return self._client.get(f"{SESSIONS_PATH}/{session_id}/next", headers=learner.headers)

    def answer(self, learner: Learner, session_id: str, item_id: str, selected: list[str]) -> httpx2.Response:
        """Send the selected option ids for one question of the session, exactly as given."""
        body: dict[str, JsonValue] = {"item_id": item_id, "selected": list(selected)}
        return self._client.post(f"{SESSIONS_PATH}/{session_id}/answers", json=body, headers=learner.headers)

    def explain(self, learner: Learner, session_id: str, item_id: str, timeout_s: float) -> httpx2.Response:
        """Ask for the explanation of one question of the session; the app calls its model, so the caller says
        how long to wait."""
        body: dict[str, JsonValue] = {"item_id": item_id}
        return self._client.post(
            f"{SESSIONS_PATH}/{session_id}/explanations", json=body, headers=learner.headers, timeout=timeout_s
        )

    def result(self, learner: Learner, session_id: str) -> httpx2.Response:
        """Ask where the session stands."""
        return self._client.get(f"{SESSIONS_PATH}/{session_id}/result", headers=learner.headers)

    def progress(self, learner: Learner, course_id: str) -> httpx2.Response:
        """Ask for the learner's progress in a course."""
        return self._client.get("/v1/progress", params={"course_id": course_id}, headers=learner.headers)
