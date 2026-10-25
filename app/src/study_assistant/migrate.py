"""Database migrations: numbered SQL files applied in order by a one-shot run before the app server starts.

Run as ``python -m study_assistant.migrate`` with the database owner's connection. The run applies every file
the database has not seen yet, then prepares the role the app server connects as and a read-only role.
The app server itself never changes the schema and never knows the owner's password.
"""

import hashlib
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.resources import files
from importlib.resources.abc import Traversable

import psycopg
import structlog
from psycopg import sql
from psycopg.conninfo import make_conninfo

from study_assistant.embed.base import EmbeddingSpace
from study_assistant.logging_setup import configure_logging
from study_assistant.settings import (
    DbSettings,
    ReaderRoleSettings,
    RoleSettings,
    ServerRoleSettings,
    load_migration_settings,
)

# "002_corpus.sql": a three-digit number, then a lower-case name.
_FILE_NAME = re.compile(r"(?P<version>\d{3})_[a-z0-9_]+\.sql")

# The journal of applied files. The hash of each file's text is kept, so a file changed after it was applied
# is noticed instead of leaving databases migrated before and after the change with different schemas.
_JOURNAL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    integer PRIMARY KEY,
    name       text NOT NULL,
    sha256     text NOT NULL,
    applied_at timestamptz NOT NULL DEFAULT now()
)
"""


class MigrationError(Exception):
    """The migration files or the database's journal are in a state the run must not continue from."""


@dataclass(frozen=True)
class Migration:
    """One numbered SQL file: its number, its file name without ".sql", its text and the SHA-256 of that text."""

    version: int
    name: str
    sql: str
    sha256: str


def load_migrations(directory: Traversable) -> tuple[Migration, ...]:
    """Read the ".sql" files of ``directory`` in number order.

    A misnamed file or a gap or repeat in the numbers is an error, so no file is skipped or applied out of order.
    """
    migrations: list[Migration] = []
    for entry in directory.iterdir():
        if not entry.name.endswith(".sql"):
            continue
        match = _FILE_NAME.fullmatch(entry.name)
        if match is None:
            raise MigrationError(f"{entry.name}: migration files are named like 002_corpus.sql")
        data = entry.read_bytes()
        migrations.append(
            Migration(
                version=int(match["version"]),
                name=entry.name.removesuffix(".sql"),
                sql=data.decode("utf-8"),
                sha256=hashlib.sha256(data).hexdigest(),
            )
        )
    migrations.sort(key=lambda migration: migration.version)
    versions = [migration.version for migration in migrations]
    if versions != list(range(1, len(versions) + 1)):
        raise MigrationError(f"migration numbers must run 1, 2, 3 ... without gaps or repeats, found {versions}")
    return tuple(migrations)


def apply_migrations(conn: psycopg.Connection, migrations: Sequence[Migration]) -> list[str]:
    """Apply the files the journal does not list, each together with its journal row; return their names.

    Each file runs in its own transaction, so a failing file leaves neither its tables nor its journal row.
    The run stops when an applied file has changed, or when the database knows a file this code does not have.
    """
    with conn.transaction():
        conn.execute(_JOURNAL)
    rows = conn.execute("SELECT version, sha256 FROM schema_migrations").fetchall()
    applied: dict[int, str] = {version: sha256 for version, sha256 in rows}
    unknown = sorted(applied.keys() - {migration.version for migration in migrations})
    if unknown:
        raise MigrationError(f"the database has migrations {unknown} that this code does not have")
    newly_applied: list[str] = []
    for migration in migrations:
        if migration.version in applied:
            if applied[migration.version] != migration.sha256:
                raise MigrationError(f"{migration.name}.sql changed after it was applied; put the change in a new file")
            continue
        try:
            with conn.transaction():
                conn.execute(migration.sql)
                conn.execute(
                    "INSERT INTO schema_migrations (version, name, sha256) VALUES (%s, %s, %s)",
                    (migration.version, migration.name, migration.sha256),
                )
        except psycopg.Error as exc:
            raise MigrationError(f"{migration.name}.sql failed and was rolled back: {exc}") from exc
        newly_applied.append(migration.name)
    return newly_applied


def prepare_embedding_space(conn: psycopg.Connection, space: EmbeddingSpace, dimension: int) -> bool:
    """Register the configured embedding space and create its vector table; return True if it was new.

    A space that is already registered is left as it is, so a repeated run changes nothing. A space registered with
    another vector length stops the run: its table would refuse every vector of the configured length. Run before
    the role grants, so the app server gets rights on a table created here.
    """
    with conn.transaction():
        row = conn.execute("SELECT dimension FROM embedding_spaces WHERE space_id = %s", (space.id,)).fetchone()
        if row is not None:
            if row[0] != dimension:
                raise MigrationError(
                    f"embedding space {space.id} is registered with {row[0]} dimensions, "
                    f"the configuration says {dimension}"
                )
            return False
        conn.execute(
            "INSERT INTO embedding_spaces (space_id, model, revision, passage_prefix, query_prefix, dimension)"
            " VALUES (%s, %s, %s, %s, %s, %s)",
            (space.id, space.model, space.revision, space.passage_prefix, space.query_prefix, dimension),
        )
        table = sql.Identifier(space.vector_table)
        # The vector row points at its chunk and repeats the chunk's course, so deleting a chunk deletes its vector
        # and a search can filter by course in this table.
        conn.execute(
            sql.SQL(
                "CREATE TABLE {table} ("
                " chunk_id text PRIMARY KEY,"
                " course_id text NOT NULL,"
                " embedding vector({dimension}) NOT NULL,"
                " FOREIGN KEY (chunk_id, course_id) REFERENCES chunks (chunk_id, course_id) ON DELETE CASCADE)"
            ).format(table=table, dimension=sql.Literal(dimension))
        )
    return True


def prepare_server_role(conn: psycopg.Connection, role: ServerRoleSettings, owner: str) -> None:
    """Create the app server's role, or set its password if it exists, and let it read and write rows only.

    The role may select, insert, update and delete rows of every table, and only read the journal and the registry
    of embedding spaces; it cannot create, alter or drop anything. Run after the migrations and the embedding space,
    so tables they added are covered too.
    """
    with conn.transaction():
        name = _login_role(conn, role, owner)
        conn.execute(sql.SQL("GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {}").format(name))
        conn.execute(sql.SQL("REVOKE INSERT, UPDATE, DELETE ON schema_migrations FROM {}").format(name))
        conn.execute(sql.SQL("REVOKE INSERT, UPDATE, DELETE ON embedding_spaces FROM {}").format(name))


def prepare_reader_role(conn: psycopg.Connection, role: ReaderRoleSettings, owner: str) -> None:
    """Create the read-only role, or set its password if it exists, and let it read rows of every table only."""
    with conn.transaction():
        name = _login_role(conn, role, owner)
        conn.execute(sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA public TO {}").format(name))


def _login_role(conn: psycopg.Connection, role: RoleSettings, owner: str) -> sql.Identifier:
    """Create the login role or reset its password, and take back every table right it was given before.

    Starting from no rights means a role ends up with exactly what the caller grants next, even when it was
    given more under an earlier configuration. Must run inside the caller's transaction.
    """
    if role.name == owner:
        raise MigrationError(f"the role {role.name!r} must not be the database owner")
    name = sql.Identifier(role.name)
    exists = conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role.name,)).fetchone() is not None
    command = sql.SQL("ALTER ROLE" if exists else "CREATE ROLE")
    password = sql.Literal(role.password.get_secret_value())
    conn.execute(sql.SQL("{} {} LOGIN PASSWORD {}").format(command, name, password))
    conn.execute(sql.SQL("REVOKE ALL ON ALL TABLES IN SCHEMA public FROM {}").format(name))
    return name


def main() -> int:
    """Apply the migrations, prepare the configured embedding space and the two roles; return the exit code."""
    settings = load_migration_settings()
    configure_logging(settings.log_level)
    log = structlog.get_logger().bind(git_commit=settings.git_commit)
    migrations_dir = files("study_assistant") / "migrations"
    space = EmbeddingSpace.from_settings(settings.embedding)
    try:
        migrations = load_migrations(migrations_dir)
        with psycopg.connect(_conninfo(settings.db), autocommit=True) as conn:
            applied = apply_migrations(conn, migrations)
            space_created = prepare_embedding_space(conn, space, settings.embedding.dimension)
            prepare_server_role(conn, settings.server_role, owner=settings.db.user)
            prepare_reader_role(conn, settings.reader_role, owner=settings.db.user)
    except (MigrationError, psycopg.Error) as exc:
        log.error("migration_failed", error=str(exc))
        return 1
    log.info(
        "migration_done",
        applied=applied,
        known=len(migrations),
        embedding_space=space.id,
        embedding_space_created=space_created,
        server_role=settings.server_role.name,
        reader_role=settings.reader_role.name,
    )
    return 0


def _conninfo(db: DbSettings) -> str:
    """Build the connection string for the owner's connection."""
    return make_conninfo(
        host=db.host,
        port=db.port,
        dbname=db.name,
        user=db.user,
        password=db.password.get_secret_value(),
        connect_timeout=db.connect_timeout_s,
    )


if __name__ == "__main__":
    sys.exit(main())
