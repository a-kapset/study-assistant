"""Removing all harness test data from the system under test, and checking that none is left.

This is the one place that knows every kind of test data. Tests call it before and after they write data,
and the command ``python -m evalharness.testdata`` calls it by hand. Each kind of data has its own module;
a new one is added to both functions below.
"""

import psycopg

from evalharness.corpus_db import conninfo
from evalharness.settings import HarnessSettings, StackSettings
from evalharness.testdata import corpus


def remove_all(settings: HarnessSettings, stack: StackSettings) -> dict[str, int]:
    """Delete every piece of test data in every store and return how many items each store removed."""
    with corpus.connect_writer(settings, stack) as conn:
        return {"corpus": corpus.remove_all(conn)}


def leftovers(settings: HarnessSettings, stack: StackSettings) -> dict[str, list[str]]:
    """List the test data still present, per store; only stores with leftovers are in the result.

    Reads through the read-only role, like every check of the harness.
    """
    reader = conninfo(settings, stack, stack.db_reader_user, stack.db_reader_password)
    with psycopg.connect(reader, autocommit=True) as conn:
        found = {"corpus": corpus.leftovers(conn)}
    return {store: items for store, items in found.items() if items}
