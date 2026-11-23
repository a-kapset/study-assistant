"""Who is asking: the learner behind a request, found by the key in its ``Authorization: Bearer`` header."""

from collections.abc import Awaitable, Callable
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from study_assistant.learners.service import Learners


def learner_dependency(learners: Learners) -> Callable[[HTTPAuthorizationCredentials], Awaitable[UUID]]:
    """Build the dependency that gives an endpoint the id of the learner whose key the request carries.

    A request without a bearer key, or with a key no learner has, is answered 401 before the endpoint runs, so
    nothing about a course or a session is told to a caller who is not a learner.
    """
    bearer = HTTPBearer(description="The key the learner got when registered.")

    async def current_learner(credentials: Annotated[HTTPAuthorizationCredentials, Depends(bearer)]) -> UUID:
        """Return the learner the key belongs to, or refuse the request."""
        learner_id = await learners.authenticate(credentials.credentials)
        if learner_id is None:
            raise HTTPException(
                status_code=401, detail="The bearer key belongs to no learner", headers={"WWW-Authenticate": "Bearer"}
            )
        return learner_id

    return current_learner
