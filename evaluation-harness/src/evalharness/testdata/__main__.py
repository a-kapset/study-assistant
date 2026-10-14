"""Command for removing or finding harness test data by hand.

    poetry run python -m evalharness.testdata clean   # delete all test data, then check that none is left
    poetry run python -m evalharness.testdata check   # only list what is left; exit code 1 if anything is

Useful after a test run was killed before its cleanup ran. Reads the stack's settings like the tests do.
"""

import sys

from evalharness.settings import HarnessSettings, load_stack_settings
from evalharness.testdata.cleanup import leftovers, remove_all

USAGE = "usage: python -m evalharness.testdata {clean|check}"


def main(argv: list[str]) -> int:
    """Run the command named in ``argv`` and return the exit code: 0 when no test data is left, 1 otherwise."""
    if argv not in (["clean"], ["check"]):
        print(USAGE, file=sys.stderr)
        return 2
    settings = HarnessSettings()
    stack = load_stack_settings(settings.stack_env_file)
    if argv == ["clean"]:
        print(f"removed: {remove_all(settings, stack)}")
    left = leftovers(settings, stack)
    print(f"left: {left}" if left else "no test data left")
    return 1 if left else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
