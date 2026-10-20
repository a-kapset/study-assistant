"""Removing all harness test data from the system under test, and checking that none is left.

This is the one place that knows every kind of test data. Tests call it before and after they write data,
and the command ``python -m evalharness.testdata`` calls it by hand. Each kind of data has its own module;
a new one is added to both functions below.

No kind exists at the moment: every state the current checks need is reached through the app itself, so the
harness writes nothing into the app's stores, and both functions report nothing.
"""

from evalharness.settings import HarnessSettings, StackSettings


def remove_all(settings: HarnessSettings, stack: StackSettings) -> dict[str, int]:
    """Delete every piece of test data in every store and return how many items each store removed."""
    return {}


def leftovers(settings: HarnessSettings, stack: StackSettings) -> dict[str, list[str]]:
    """List the test data still present, per store; only stores with leftovers are in the result.

    Reads through the read-only role, like every check of the harness.
    """
    return {}
