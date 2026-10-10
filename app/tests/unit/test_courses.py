from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

from study_assistant.composition import build_app
from study_assistant.courses.registry import CourseConfigError, load_course, load_courses
from study_assistant.db import Database
from study_assistant.settings import CoursesSettings, DbSettings, Settings

# The repository's own course folder: app/tests/unit/<this file> -> repository root.
REPO_COURSES = Path(__file__).resolve().parents[3] / "courses"


def _manifest(course_id: str, **changes: object) -> dict[str, object]:
    """Return a valid manifest for ``course_id`` as plain data, with ``changes`` applied on top."""
    manifest: dict[str, object] = {
        "id": course_id,
        "title": f"Course {course_id}",
        "description": "A course used only by this test.",
        "language": "en",
        "structure": {"section_heading": r"^Part (\d+)", "objective_id": r"OBJ-\d+"},
        "sources": [{"file": "text.md", "role": "material", "authority": "primary"}],
    }
    manifest.update(changes)
    return manifest


def _write(root: Path, folder: str, content: str) -> Path:
    """Write ``content`` as ``root/folder/course.yaml`` and return the file's path."""
    path = root / folder / "course.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_enabled_courses_are_served_in_settings_order(settings: Settings, fake_db: Database, tmp_path: Path) -> None:
    """Guards the wiring and the contract: courses in the order enabled, only the summary fields, no internals."""
    for course_id in ("alpha", "beta", "unused"):
        _write(tmp_path, course_id, yaml.safe_dump(_manifest(course_id)))
    configured = settings.model_copy(
        update={"courses": CoursesSettings(manifest_dir=tmp_path, enabled=("beta", "alpha"))}
    )

    with TestClient(build_app(configured, fake_db)) as client:
        response = client.get("/v1/courses")

    assert response.status_code == 200
    courses = response.json()["courses"]
    assert [course["id"] for course in courses] == ["beta", "alpha"]
    assert set(courses[0]) == {"id", "title", "description", "language"}


def test_bad_manifest_stops_the_app_from_being_built(settings: Settings, fake_db: Database, tmp_path: Path) -> None:
    """Guards startup: a course that fails to load must stop the app, not be skipped with a log line."""
    _write(tmp_path, "alpha", "id: [")
    configured = settings.model_copy(update={"courses": CoursesSettings(manifest_dir=tmp_path, enabled=("alpha",))})

    with pytest.raises(CourseConfigError, match="alpha"):
        build_app(configured, fake_db)


def test_enabled_courses_are_read_as_a_comma_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guards the variable format: without the raw-string handling the value would have to be a JSON list."""
    monkeypatch.setenv("APP_COURSES__ENABLED", " alpha, beta ,")

    settings = Settings(db=DbSettings(password=SecretStr("unit-test-password")))

    assert settings.courses.enabled == ("alpha", "beta")


@pytest.mark.parametrize("value", ["../alpha", "Alpha", "alpha,alpha"])
def test_unsafe_or_repeated_course_ids_are_refused(value: str) -> None:
    """Guards the ids from the environment: they become path parts, and a repeated one would load a course twice."""
    with pytest.raises(ValidationError, match="enabled"):
        CoursesSettings.model_validate({"enabled": value})


def test_manifest_id_must_equal_its_folder(tmp_path: Path) -> None:
    """Guards a copied folder whose manifest still carries the old id, so two courses would share one id."""
    _write(tmp_path, "beta", yaml.safe_dump(_manifest("alpha")))

    with pytest.raises(CourseConfigError, match="declares id 'alpha'"):
        load_course(tmp_path, "beta")


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        (None, "cannot read"),
        ("id: [", "not valid YAML"),
        (yaml.safe_dump(_manifest("alpha", titel="misspelt key")), "titel"),
        (
            yaml.safe_dump(_manifest("alpha", structure={"section_heading": "(", "objective_id": "x"})),
            "section_heading",
        ),
        (yaml.safe_dump(_manifest("alpha", sources=[])), "sources"),
    ],
    ids=["missing", "broken-yaml", "unknown-key", "broken-regex", "no-sources"],
)
def test_bad_manifest_error_names_the_course_the_file_and_the_reason(
    tmp_path: Path, content: str | None, reason: str
) -> None:
    """Guards the startup message: it must say which course, which file and what is wrong, not only that it failed."""
    if content is not None:
        _write(tmp_path, "alpha", content)

    with pytest.raises(CourseConfigError) as caught:
        load_courses(tmp_path, ["alpha"])

    message = str(caught.value)
    assert "Course 'alpha'" in message
    assert str(tmp_path / "alpha" / "course.yaml") in message
    assert reason in message


def test_committed_course_manifests_are_valid() -> None:
    """Guards the repository's own manifests: an edit that breaks one must fail here, not at someone's startup."""
    course_ids = sorted(path.parent.name for path in REPO_COURSES.glob("*/course.yaml"))

    loaded = load_courses(REPO_COURSES, course_ids)

    assert "demo" in loaded
    assert list(loaded) == course_ids
