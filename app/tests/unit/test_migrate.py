import hashlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from importlib.resources import files
from pathlib import Path
from typing import cast

import psycopg
import pytest
from psycopg import sql
from pydantic import SecretStr

from study_assistant.migrate import (
    Migration,
    MigrationError,
    apply_migrations,
    load_migrations,
    prepare_reader_role,
    prepare_server_role,
)
from study_assistant.settings import ReaderRoleSettings, RoleSettings, ServerRoleSettings


def _write(directory: Path, files_and_text: dict[str, str]) -> None:
    """Create each named file in ``directory`` with the given text."""
    for name, text in files_and_text.items():
        (directory / name).write_text(text, encoding="utf-8")


def _migration(version: int, name: str, sql: str) -> Migration:
    """Build a migration the way loading a file would, hash included."""
    return Migration(version=version, name=name, sql=sql, sha256=hashlib.sha256(sql.encode("utf-8")).hexdigest())


class _Result:
    """The rows one fake statement returns."""

    def __init__(self, rows: list[tuple[object, ...]]) -> None:
        """Keep the rows."""
        self._rows = rows

    def fetchall(self) -> list[tuple[object, ...]]:
        """Return every row."""
        return self._rows

    def fetchone(self) -> tuple[object, ...] | None:
        """Return the first row, or None when there is none."""
        return self._rows[0] if self._rows else None


class _FakeConnection:
    """Stands in for a database connection: a journal of applied versions and a record of the SQL run.

    A statement equal to ``failing_sql`` raises the way PostgreSQL rejects a broken file.
    """

    def __init__(self, journal: dict[int, str], failing_sql: str | None = None) -> None:
        """Start with ``journal`` (version -> SHA-256) as the applied files."""
        self.journal = dict(journal)
        self.failing_sql = failing_sql
        self.executed: list[str] = []

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Run the block; the fake keeps no state that a rollback would have to undo."""
        yield

    def execute(self, query: str | sql.Composable, params: tuple[object, ...] | None = None) -> _Result:
        """Answer the journal query, record journal inserts and run (record) everything else.

        A statement composed from parts is recorded as the text it would send. No role exists yet.
        """
        if isinstance(query, sql.Composable):
            query = query.as_string(None)
        if query == self.failing_sql:
            raise psycopg.errors.UndefinedObject("type does not exist")
        if query.startswith("SELECT version, sha256 FROM schema_migrations"):
            return _Result(list(self.journal.items()))
        if query.startswith("INSERT INTO schema_migrations") and params is not None:
            version, _name, sha256 = params
            self.journal[cast(int, version)] = cast(str, sha256)
            return _Result([])
        self.executed.append(query)
        return _Result([])


def _apply(conn: _FakeConnection, migrations: list[Migration]) -> list[str]:
    """Run ``apply_migrations`` against the fake connection."""
    return apply_migrations(cast(psycopg.Connection, conn), migrations)


def test_files_load_in_number_order_with_name_and_hash(tmp_path: Path) -> None:
    """Guards ordering by number (not by directory listing), the stored name and the hash of the exact bytes."""
    _write(tmp_path, {"002_second.sql": "SELECT 2;", "001_first.sql": "SELECT 1;", "notes.txt": "not a migration"})

    migrations = load_migrations(tmp_path)

    assert [(m.version, m.name, m.sql) for m in migrations] == [
        (1, "001_first", "SELECT 1;"),
        (2, "002_second", "SELECT 2;"),
    ]
    assert migrations[0].sha256 == hashlib.sha256(b"SELECT 1;").hexdigest()


@pytest.mark.parametrize("name", ["1_short.sql", "003-dash.sql", "003_Upper.sql", "003_.sql", "v003_prefix.sql"])
def test_misnamed_sql_file_stops_the_run(tmp_path: Path, name: str) -> None:
    """Guards the name rule: a misnamed file must stop the run instead of being skipped or applied out of order."""
    _write(tmp_path, {"001_first.sql": "SELECT 1;", "002_second.sql": "SELECT 2;", name: "SELECT 3;"})

    with pytest.raises(MigrationError, match=name):
        load_migrations(tmp_path)


@pytest.mark.parametrize(
    "names",
    [
        pytest.param(["001_a.sql", "003_c.sql"], id="gap"),
        pytest.param(["001_a.sql", "001_b.sql"], id="repeat"),
        pytest.param(["002_b.sql"], id="not-starting-at-one"),
    ],
)
def test_numbers_must_run_without_gaps_or_repeats(tmp_path: Path, names: list[str]) -> None:
    """Guards the numbering check: a gap or a repeat would apply files in an order nobody intended."""
    _write(tmp_path, dict.fromkeys(names, "SELECT 1;"))

    with pytest.raises(MigrationError, match="without gaps or repeats"):
        load_migrations(tmp_path)


def test_packaged_migrations_are_found_and_numbered() -> None:
    """Guards the folder shipped inside the package: a moved folder or misnamed new file fails here, not in Compose."""
    migrations = load_migrations(files("study_assistant") / "migrations")

    assert [m.name for m in migrations][:2] == ["001_vector", "002_corpus"]


def test_only_files_missing_from_the_journal_are_applied_in_order() -> None:
    """Guards re-running: applied files must be skipped and new ones applied in order and written to the journal."""
    first, second, third = (_migration(n, f"00{n}_m", f"CREATE TABLE t{n} ();") for n in (1, 2, 3))
    conn = _FakeConnection(journal={1: first.sha256})

    applied = _apply(conn, [first, second, third])

    assert applied == ["002_m", "003_m"]
    assert conn.executed[1:] == [second.sql, third.sql]
    assert conn.journal == {1: first.sha256, 2: second.sha256, 3: third.sha256}


def test_nothing_is_applied_when_the_journal_is_complete() -> None:
    """Guards idempotence: a second run over the same files must change nothing."""
    first = _migration(1, "001_m", "CREATE TABLE t ();")
    conn = _FakeConnection(journal={1: first.sha256})

    assert _apply(conn, [first]) == []
    assert first.sql not in conn.executed


def test_changed_applied_file_stops_the_run() -> None:
    """Guards the hash check: an edited applied file would leave old and new databases with different schemas."""
    edited = _migration(1, "001_m", "CREATE TABLE t (); -- edited")
    newer = _migration(2, "002_m", "CREATE TABLE u ();")
    conn = _FakeConnection(journal={1: _migration(1, "001_m", "CREATE TABLE t ();").sha256})

    with pytest.raises(MigrationError, match=r"001_m\.sql changed after it was applied"):
        _apply(conn, [edited, newer])
    assert newer.sql not in conn.executed


def test_database_ahead_of_the_code_stops_the_run() -> None:
    """Guards running older code on a newer schema: a journal entry with no file must stop the run."""
    first = _migration(1, "001_m", "CREATE TABLE t ();")
    conn = _FakeConnection(journal={1: first.sha256, 2: "f" * 64})

    with pytest.raises(MigrationError, match=r"\[2\]"):
        _apply(conn, [first])


def test_failing_file_is_named_and_later_files_are_not_applied() -> None:
    """Guards the error path: the failing file must be named, left out of the journal, and stop the files after it."""
    broken = _migration(1, "001_broken", "CREATE TABLE t (a nosuchtype);")
    later = _migration(2, "002_later", "CREATE TABLE u ();")
    conn = _FakeConnection(journal={}, failing_sql=broken.sql)

    with pytest.raises(MigrationError, match=r"001_broken\.sql failed and was rolled back"):
        _apply(conn, [broken, later])
    assert conn.journal == {}
    assert later.sql not in conn.executed


def test_app_role_named_like_the_owner_is_refused() -> None:
    """Guards the owner check: the same name would give the owner the app's password and the app a superuser."""
    conn = _FakeConnection(journal={})
    role = ServerRoleSettings(name="owner_user", password=SecretStr("app-password"))

    with pytest.raises(MigrationError, match="must not be the database owner"):
        prepare_server_role(cast(psycopg.Connection, conn), role, owner="owner_user")
    assert conn.executed == []


def _role_statements(prepare: object, role: RoleSettings) -> list[str]:
    """Run a role preparation against a fresh fake connection and return the statements it sent."""
    conn = _FakeConnection(journal={})
    cast(Callable[..., None], prepare)(cast(psycopg.Connection, conn), role, owner="owner_user")
    return [statement for statement in conn.executed if not statement.startswith("SELECT 1 FROM pg_roles")]


def test_reader_role_may_only_read_rows() -> None:
    """Guards the reader's grants: any write right would let a checking tool change the data it checks."""
    role = ReaderRoleSettings(password=SecretStr("reader-password"))

    statements = _role_statements(prepare_reader_role, role)

    assert statements == [
        "CREATE ROLE \"app_ro\" LOGIN PASSWORD 'reader-password'",
        'REVOKE ALL ON ALL TABLES IN SCHEMA public FROM "app_ro"',
        'GRANT SELECT ON ALL TABLES IN SCHEMA public TO "app_ro"',
    ]


def test_server_role_starts_from_no_rights_before_its_grants() -> None:
    """Guards the reset: without it, a role that had more rights under an earlier name or setup would keep them."""
    role = ServerRoleSettings(password=SecretStr("app-password"))

    statements = _role_statements(prepare_server_role, role)

    revoke_all = statements.index('REVOKE ALL ON ALL TABLES IN SCHEMA public FROM "app_rw"')
    grant = statements.index('GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO "app_rw"')
    assert revoke_all < grant


def test_reader_role_named_like_the_owner_is_refused() -> None:
    """Guards the owner check for the reader: it would reset the owner's password and strip its rights."""
    conn = _FakeConnection(journal={})
    role = ReaderRoleSettings(name="owner_user", password=SecretStr("reader-password"))

    with pytest.raises(MigrationError, match="must not be the database owner"):
        prepare_reader_role(cast(psycopg.Connection, conn), role, owner="owner_user")
    assert conn.executed == []
