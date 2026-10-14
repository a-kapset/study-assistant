from collections.abc import Callable
from pathlib import Path
from typing import cast

import psycopg
import pytest

from evalharness.testdata import PREFIX
from evalharness.testdata.corpus import CorpusTestData

# evaluation-harness/tests/unit/<this file> -> repository root.
REPO_COURSES = Path(__file__).resolve().parents[3] / "courses"


def test_no_real_course_uses_the_test_data_prefix() -> None:
    """No course folder and no course manifest in the repository may use the test data prefix.

    The cleanup deletes everything that starts with the prefix. A real course or file named like that would be
    deleted after every test run that writes test data.
    """
    manifests = sorted(REPO_COURSES.glob("*/course.yaml"))

    assert manifests, f"no course manifests found under {REPO_COURSES}"
    for manifest in manifests:
        assert not manifest.parent.name.startswith(PREFIX), f"course folder {manifest.parent.name} uses {PREFIX}"
        assert PREFIX not in manifest.read_text(encoding="utf-8"), f"{manifest} uses {PREFIX}"


class _NoConnection:
    """Stands in for a database connection that must not be used: any statement fails the test."""

    def execute(self, *args: object) -> None:
        """Fail, because a refused name must be refused before anything is written."""
        raise AssertionError("nothing may be written for a name without the prefix")


@pytest.mark.parametrize(
    "create",
    [
        lambda data: data.copy_course("demo", "demo_copy"),
        lambda data: data.add_document("demo", "extra.pdf"),
    ],
    ids=["course id", "file name"],
)
def test_test_data_without_the_prefix_is_refused(create: Callable[[CorpusTestData], None]) -> None:
    """Creating a course or a document whose name lacks the prefix must fail before anything is written.

    Such data would never be found by the cleanup and would stay in the app's database.
    """
    data = CorpusTestData(cast(psycopg.Connection, _NoConnection()))

    with pytest.raises(ValueError, match=PREFIX):
        create(data)
