import json
from pathlib import Path

import pytest

from study_assistant.exams.bank import BankError, read_bank


def _item(
    item_id: str,
    options: tuple[str, ...] = ("a", "b", "c", "d"),
    keys: tuple[str, ...] = ("b",),
    **extra: object,
) -> dict[str, object]:
    """Build one question record of an invented course; ``extra`` adds or replaces fields."""
    record: dict[str, object] = {
        "id": item_id,
        "stem": f"Question {item_id} about the river board.",
        "options": [{"id": option_id, "text": f"Answer {option_id}"} for option_id in options],
        "keys": list(keys),
        "objective_id": "OBJ-01-02",
        "source": {"document": "rules_questions.pdf", "version": "1", "question": item_id, "page": 3},
    }
    record.update(extra)
    return record


def _bank(tmp_path: Path, *lines: str) -> Path:
    """Write the given lines as a bank file and return its path."""
    path = tmp_path / "bank.jsonl"
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return path


def _refusal(path: Path) -> BankError:
    """Read a bank that must be refused and return the error."""
    with pytest.raises(BankError) as caught:
        read_bank(path)
    return caught.value


@pytest.mark.covers("US-005", "AC-1")
def test_a_bank_of_any_shape_is_read_whole_in_file_order(tmp_path: Path) -> None:
    """A bank with a 3-option question, a 5-option question and a question with two keys, with a blank line
    between them, is read completely and in file order, with both keys kept. Catches a reader tied to four options
    or one key, one that stops at a blank line, or one that drops or reorders items."""
    path = _bank(
        tmp_path,
        json.dumps(_item("q1", options=("a", "b", "c"))),
        "",
        json.dumps(_item("q2", options=("a", "b", "c", "d", "e"), keys=("e",))),
        json.dumps(_item("q3", keys=("a", "c"))),
    )

    items = read_bank(path)

    assert [item.id for item in items] == ["q1", "q2", "q3"], f"read {[item.id for item in items]}, not the whole bank"
    assert items[2].keys == ("a", "c"), f"the two-key question was read with keys {items[2].keys}"


@pytest.mark.covers("US-005", "AC-2")
def test_a_key_that_is_not_an_option_is_refused(tmp_path: Path) -> None:
    """The second question's key is "e", but it has options a to d: reading stops at line 2 with
    ``invalid_item``. Catches a reader that accepts a question no answer could ever pass."""
    path = _bank(tmp_path, json.dumps(_item("q1")), json.dumps(_item("q2", keys=("e",))))

    error = _refusal(path)

    assert (error.code, error.line) == ("invalid_item", 2), f"got {error.code} at line {error.line}: {error}"


@pytest.mark.covers("US-005", "AC-2")
def test_repeated_option_ids_are_refused(tmp_path: Path) -> None:
    """A question has two options with the id "a": reading stops with ``invalid_item``. Catches a reader that
    accepts a question where the key cannot say which of the two options is meant."""
    path = _bank(tmp_path, json.dumps(_item("q1", options=("a", "a", "b"), keys=("b",))))

    error = _refusal(path)

    assert (error.code, error.line) == ("invalid_item", 1), f"got {error.code} at line {error.line}: {error}"


@pytest.mark.covers("US-005", "AC-2")
def test_repeated_keys_are_refused(tmp_path: Path) -> None:
    """A question lists the key "b" twice: reading stops with ``invalid_item``. Catches a reader that keeps such
    a key, so a later comparison of the answer with the key would expect two selections of one option."""
    path = _bank(tmp_path, json.dumps(_item("q1", keys=("b", "b"))))

    error = _refusal(path)

    assert (error.code, error.line) == ("invalid_item", 1), f"got {error.code} at line {error.line}: {error}"


@pytest.mark.covers("US-005", "AC-3")
def test_an_unknown_field_is_refused(tmp_path: Path) -> None:
    """A question carries a misspelt field ``objectve_id`` next to the right one: reading stops with
    ``invalid_item``. Catches a reader that silently ignores unknown fields, so a typo in a converter's output
    would lose data without anyone noticing."""
    path = _bank(tmp_path, json.dumps(_item("q1", objectve_id="OBJ-01-03")))

    error = _refusal(path)

    assert (error.code, error.line) == ("invalid_item", 1), f"got {error.code} at line {error.line}: {error}"


@pytest.mark.covers("US-005", "AC-4")
def test_a_broken_line_is_refused_with_its_number(tmp_path: Path) -> None:
    """The second line is not JSON: reading stops with ``invalid_json`` at line 2. Catches line numbers counted
    from 0 and broken JSON reported as a wrong question, which would send the reader to the wrong fix."""
    path = _bank(tmp_path, json.dumps(_item("q1")), '{"id": "q2", ')

    error = _refusal(path)

    assert (error.code, error.line) == ("invalid_json", 2), f"got {error.code} at line {error.line}: {error}"


@pytest.mark.covers("US-005", "AC-5")
def test_a_repeated_item_id_is_refused_at_its_second_line(tmp_path: Path) -> None:
    """Two questions share the id "q1", with a blank line between them: reading stops with
    ``duplicate_item_id`` at line 3, the second one. Catches a reader that keeps both, so answers and results that
    refer to "q1" could mean either question."""
    path = _bank(tmp_path, json.dumps(_item("q1")), "", json.dumps(_item("q1")))

    error = _refusal(path)

    assert (error.code, error.line) == ("duplicate_item_id", 3), f"got {error.code} at line {error.line}: {error}"
