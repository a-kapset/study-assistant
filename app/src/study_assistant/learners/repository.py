"""Learners in PostgreSQL."""

from uuid import UUID

from study_assistant.db import Database


class PostgresLearnerStore:
    """The learner store of the running app, on the app's connection pool."""

    def __init__(self, database: Database) -> None:
        """Keep the database; connections are taken from it per call."""
        self._database = database

    async def add(self, key_sha256: str) -> UUID:
        """Insert a learner and return the id the database gave it."""
        async with self._database.connection() as conn:
            cursor = await conn.execute(
                "INSERT INTO learners (key_sha256) VALUES (%s) RETURNING learner_id", (key_sha256,)
            )
            row = await cursor.fetchone()
        if row is None:
            raise RuntimeError("the database returned no id for the new learner")
        learner_id: UUID = row[0]
        return learner_id

    async def find(self, key_sha256: str) -> UUID | None:
        """Look a learner up by the hash of its key."""
        async with self._database.connection() as conn:
            cursor = await conn.execute("SELECT learner_id FROM learners WHERE key_sha256 = %s", (key_sha256,))
            row = await cursor.fetchone()
        if row is None:
            return None
        learner_id: UUID = row[0]
        return learner_id
