"""PostgreSQL access: a connection pool that opens when the app starts and closes when it stops."""

import asyncio
from typing import Protocol

import psycopg
from psycopg.conninfo import make_conninfo
from psycopg_pool import AsyncConnectionPool

from study_assistant.settings import DbSettings

# One open connection is enough until real queries arrive; the pool opens more on demand, up to the maximum.
_POOL_MIN_SIZE = 1
_POOL_MAX_SIZE = 4


class DatabaseUnavailableError(Exception):
    """The database could not be reached, or did not answer in time."""


class Database(Protocol):
    """What the rest of the app needs from its database: open, close and a readiness ping.

    Code that uses the database (the readiness endpoint, the app's startup and shutdown) depends on
    this protocol, not on ``PostgresDatabase``. The running app passes ``PostgresDatabase``; unit tests
    pass a small fake that can be switched between "up" and "down", so the readiness logic is tested
    without a real database, and the type checker still confirms that the fake has the same methods.
    """

    async def open(self) -> None:
        """Start connecting; must not fail when the database is down."""
        ...

    async def close(self) -> None:
        """Release every connection."""
        ...

    async def ping(self) -> None:
        """Run a trivial query; raise ``DatabaseUnavailableError`` when the database does not answer."""
        ...


class PostgresDatabase:
    """The real database: a psycopg connection pool built from the settings."""

    def __init__(self, settings: DbSettings) -> None:
        """Keep the settings; nothing connects until ``open``."""
        self._settings = settings
        self._pool: AsyncConnectionPool | None = None

    async def open(self) -> None:
        """Create the pool and fill it in the background, so the app starts even while the database is down."""
        s = self._settings
        conninfo = make_conninfo(
            host=s.host,
            port=s.port,
            dbname=s.name,
            user=s.user,
            password=s.password.get_secret_value(),
            connect_timeout=s.connect_timeout_s,
        )
        self._pool = AsyncConnectionPool(
            conninfo,
            open=False,
            min_size=_POOL_MIN_SIZE,
            max_size=_POOL_MAX_SIZE,
            timeout=s.connect_timeout_s,
            # A connection broken by a database restart is replaced instead of being handed out.
            check=AsyncConnectionPool.check_connection,
        )
        await self._pool.open(wait=False)

    async def close(self) -> None:
        """Close every connection; calling it again, or before ``open``, does nothing."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def ping(self) -> None:
        """Run ``SELECT 1`` within the connect timeout; any database error becomes ``DatabaseUnavailableError``."""
        if self._pool is None:
            raise DatabaseUnavailableError("the connection pool is not open")
        try:
            async with asyncio.timeout(self._settings.connect_timeout_s), self._pool.connection() as conn:
                await conn.execute("SELECT 1")
        except (psycopg.Error, TimeoutError) as exc:
            # Only the error type: database messages can carry connection details that do not belong in a response.
            raise DatabaseUnavailableError(type(exc).__name__) from exc
