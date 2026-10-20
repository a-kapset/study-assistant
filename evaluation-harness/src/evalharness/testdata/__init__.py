"""Test data that the harness writes into the system under test, created and removed in one place.

Some checks may need data that the app cannot be brought to through its own inputs, for example recorded model
answers or uploaded files for the AI components later on. All of it is made by the modules of this package,
never by single test files. A state the app reaches through its configuration, its API or its input files is
reached that way instead.

The contract: every name the harness writes starts with ``PREFIX`` wherever the field allows it (course ids,
file names, marker values). Removal deletes everything that carries the prefix, so a new kind of test data
cannot be forgotten by the cleanup. Real courses and files must never use the prefix; a unit test checks the
repository's course manifests for it.

Each kind of data has its own module with a ``remove_all`` and a ``leftovers`` function; ``cleanup`` runs
them all. A new kind adds its module and one line in each function of ``cleanup``.
"""

PREFIX = "evalharness_"


def name(suffix: str) -> str:
    """Build a test data name from ``suffix``: ``name("foreign")`` gives ``"evalharness_foreign"``."""
    return PREFIX + suffix
