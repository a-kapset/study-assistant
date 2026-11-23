"""Learner identity: an operator registers a learner and hands over the key; every request of the learner carries it.

The key is a long random value shown once, when the learner is registered. The app stores only its SHA-256, so
a copy of the database gives nobody a usable key. A fast hash is enough here: unlike a password, the key cannot
be guessed, so there is nothing a slow hash would protect.
"""

import hashlib
import secrets
from typing import Protocol
from uuid import UUID

# 32 random bytes: far beyond guessing, and short enough to paste into a header.
_KEY_BYTES = 32


class LearnerStore(Protocol):
    """Where learners are stored, each with the hash of its key."""

    async def add(self, key_sha256: str) -> UUID:
        """Store a new learner with this key hash and return the learner's id."""
        ...

    async def find(self, key_sha256: str) -> UUID | None:
        """Return the id of the learner whose key has this hash, or ``None`` when there is none."""
        ...

    async def remove(self, learner_id: UUID) -> bool:
        """Delete the learner with every session and answer it has; return whether there was such a learner."""
        ...


def key_digest(key: str) -> str:
    """Return the SHA-256 of a key as hex: the only form in which a key is stored or looked up."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class Learners:
    """Registers learners and tells which learner a key belongs to."""

    def __init__(self, store: LearnerStore) -> None:
        """Keep the store; nothing is read yet."""
        self._store = store

    async def register(self) -> tuple[UUID, str]:
        """Create a learner and return its id with its key; the key cannot be read again later."""
        key = secrets.token_urlsafe(_KEY_BYTES)
        return await self._store.add(key_digest(key)), key

    async def authenticate(self, key: str) -> UUID | None:
        """Return the learner a key belongs to, or ``None`` for a key no learner has."""
        return await self._store.find(key_digest(key))

    async def remove(self, learner_id: UUID) -> bool:
        """Delete a learner with its sessions and answers, so its key opens nothing any more; return whether the
        learner existed."""
        return await self._store.remove(learner_id)
