"""Proof that the running app is in the state a mutant run claims: read from the settings the app itself reports.

Starting the app again with other values only commands a change. Whether the app runs with them is seen in what it
reports at GET /v1/config, compared with the reading taken before the mutant. A mutant counts only when exactly its
own settings differ, each with its declared value: a mutant that was not applied would read as a defect no check
sees, and one more changed setting would make it unclear which change a check noticed. After the run the reading
must equal the first one again.
"""

import json
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, JsonValue

# The app reads its settings with this prefix and joins a group and a name with this delimiter.
_PREFIX = "APP_"
_DELIMITER = "__"


class Change(BaseModel):
    """One setting that differs between two readings of the app's settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    before: JsonValue
    after: JsonValue


def setting_path(name: str) -> str:
    """Return where a setting appears in the app's reported settings: APP_SEARCH__MIN_SCORE is search.min_score."""
    if not name.startswith(_PREFIX):
        raise ValueError(f"not an app setting: {name!r}")
    return ".".join(part.lower() for part in name.removeprefix(_PREFIX).split(_DELIMITER))


def declared_value(text: str) -> JsonValue:
    """Read a declared value as the app would type it: a number, true / false or a list as JSON, else the text."""
    try:
        value: JsonValue = json.loads(text)
    except json.JSONDecodeError:
        return text
    return value


def flatten(settings: Mapping[str, JsonValue], prefix: str = "") -> dict[str, JsonValue]:
    """Turn nested settings into one level keyed by their dotted path, e.g. {"search": {"k": 5}} → {"search.k": 5}."""
    flat: dict[str, JsonValue] = {}
    for key, value in settings.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{path}."))
        else:
            flat[path] = value
    return flat


def changes(before: Mapping[str, JsonValue], after: Mapping[str, JsonValue]) -> dict[str, Change]:
    """Return every setting whose value differs between two readings, a setting present in only one included."""
    old, new = flatten(before), flatten(after)
    return {
        path: Change(before=old.get(path), after=new.get(path))
        for path in sorted(old.keys() | new.keys())
        if old.get(path) != new.get(path)
    }


def apply_problem(declared: Mapping[str, str], changed: Mapping[str, Change]) -> str | None:
    """Say why the app is not running the mutant; None when exactly the declared settings changed to their values.

    For the baseline nothing is declared, so any change is a problem.
    """
    expected = {setting_path(name): declared_value(value) for name, value in declared.items()}
    problems = []
    unchanged = sorted(path for path in expected if path not in changed)
    if unchanged:
        problems.append(f"not changed: {unchanged}")
    wrong = sorted(
        f"{path} is {changed[path].after!r}, declared {value!r}"
        for path, value in expected.items()
        if path in changed and changed[path].after != value
    )
    if wrong:
        problems.append(f"other value: {wrong}")
    extra = sorted(path for path in changed if path not in expected)
    if extra:
        problems.append(f"changed but not declared: {extra}")
    return "; ".join(problems) or None


def restore_problem(changed: Mapping[str, Change]) -> str | None:
    """Say why the app is not back in its first state; None when no setting differs from the first reading."""
    if not changed:
        return None
    return "settings differ from the first reading: " + ", ".join(
        f"{path} {change.before!r} -> {change.after!r}" for path, change in changed.items()
    )
