from pathlib import Path

from evalharness.testdata import PREFIX

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
