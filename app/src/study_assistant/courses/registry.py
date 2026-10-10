"""Loading course manifests at startup: every enabled course must have a valid <manifest_dir>/<course_id>/course.yaml.

A course that cannot be loaded stops the app instead of being skipped, so the app never runs without a course
it was configured to serve.
"""

from collections.abc import Iterable, Mapping
from pathlib import Path

import yaml
from pydantic import ValidationError

from study_assistant.courses.models import CourseManifest

MANIFEST_FILE = "course.yaml"


class CourseConfigError(ValueError):
    """A course manifest is missing, unreadable or invalid; the message names the course and the file."""


def load_courses(manifest_dir: Path, enabled: Iterable[str]) -> Mapping[str, CourseManifest]:
    """Load the manifest of every enabled course, keyed by course id in the order given; the first bad one stops it."""
    return {course_id: load_course(manifest_dir, course_id) for course_id in enabled}


def load_course(manifest_dir: Path, course_id: str) -> CourseManifest:
    """Read and validate one course's manifest; the id inside must equal the course's folder name.

    The file is read as bytes so that YAML detects its encoding and reports bytes it cannot decode as a YAML error.
    """
    path = (manifest_dir / course_id / MANIFEST_FILE).resolve()
    try:
        # safe_load builds only plain data; the full loader can create arbitrary Python objects from tags.
        raw = yaml.safe_load(path.read_bytes())
    except OSError as exc:
        raise CourseConfigError(f"Course '{course_id}': cannot read {path}: {exc.strerror}") from exc
    except yaml.YAMLError as exc:
        raise CourseConfigError(f"Course '{course_id}': {path} is not valid YAML: {exc}") from exc

    try:
        manifest = CourseManifest.model_validate(raw)
    except ValidationError as exc:
        raise CourseConfigError(f"Course '{course_id}': {path} does not match the manifest format: {exc}") from exc

    if manifest.id != course_id:
        raise CourseConfigError(
            f"Course '{course_id}': {path} declares id '{manifest.id}', but the id must equal its folder name"
        )
    return manifest
