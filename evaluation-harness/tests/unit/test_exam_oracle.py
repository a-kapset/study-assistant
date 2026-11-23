"""The harness's own expectations for practice sessions: the bank it reads and the result and progress it computes.

The state machine test of the running app trusts these functions as its reference, so a mistake here would make
it pass a wrong app or fail a right one.
"""

import json
from pathlib import Path

import pytest

from evalharness.datasets.question_bank import BankQuestion, QuestionBankError, read_question_bank
from evalharness.oracles.exam_session import SessionModel, expected_progress, expected_result, is_correct


def _line(item_id: str, keys: list[str], objective_id: str = "OBJ-1", options: str = "abcd") -> str:
    """One bank line with the given key, objective and option ids, plus fields the exam checks do not read."""
    return json.dumps(
        {
            "id": item_id,
            "stem": "A question of an invented course.",
            "options": [{"id": option_id, "text": f"Answer {option_id}"} for option_id in options],
            "keys": keys,
            "objective_id": objective_id,
            "source": {"document": "questions.pdf", "version": "1", "question": item_id},
        }
    )


def _bank(tmp_path: Path, *lines: str) -> Path:
    """Write the lines as a bank file and return its path."""
    path = tmp_path / "bank.jsonl"
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return path


def _question(item_id: str, objective_id: str, keys: tuple[str, ...] = ("b",)) -> BankQuestion:
    """A question with the options a to d."""
    return BankQuestion(
        item_id=item_id, option_ids=("a", "b", "c", "d"), keys=frozenset(keys), objective_id=objective_id
    )


def test_bank_is_read_with_options_keys_and_objective_of_every_question(tmp_path: Path) -> None:
    """Reads a bank of two questions, one with two keys and five options. A pass says the reference of the exam
    checks holds what the file says; a failure means every exam check would judge answers by a wrong key or miss
    an option. Worth having because nothing else compares the harness's reading with the file."""
    path = _bank(tmp_path, _line("Q1", ["b"]), "", _line("Q2", ["a", "e"], "OBJ-2", options="abcde"))

    bank = read_question_bank(path)

    assert list(bank) == ["Q1", "Q2"], f"the questions must come in file order, got {list(bank)}"
    assert bank["Q2"] == BankQuestion(
        item_id="Q2", option_ids=("a", "b", "c", "d", "e"), keys=frozenset({"a", "e"}), objective_id="OBJ-2"
    ), f"Q2 was read as {bank['Q2']}"


@pytest.mark.parametrize(
    ("lines", "reason"),
    [
        ([_line("Q1", ["z"])], "not among its options"),
        ([_line("Q1", ["b"]), _line("Q1", ["c"])], "used twice"),
        ([_line("Q1", ["b"]), "{not json"], "line 2"),
    ],
)
def test_a_bank_that_cannot_serve_as_reference_is_refused(tmp_path: Path, lines: list[str], reason: str) -> None:
    """A key outside the options, a repeated id and a broken line. A pass says the checks never run on a
    reference with a hole; a failure means a later question could silently replace an earlier one, or a key no
    answer can match would make every answer to it look wrong."""
    with pytest.raises(QuestionBankError) as caught:
        read_question_bank(_bank(tmp_path, *lines))

    assert reason in str(caught.value), f"the refusal must say why ({reason}): {caught.value}"


@pytest.mark.parametrize(
    ("chosen", "expected"), [(["a", "c"], True), (["c", "a"], True), (["a"], False), (["a", "b", "c"], False)]
)
def test_an_answer_is_correct_only_as_the_exact_key(chosen: list[str], expected: bool) -> None:
    """A question with the key a and c. A pass says the model judges an answer by the whole key, whatever the
    order; a failure means the state machine test would accept an app that takes a part of the key as correct."""
    verdict = is_correct(chosen, _question("Q1", "OBJ-1", keys=("a", "c")))

    assert verdict is expected, f"choosing {chosen} for the key a, c must be correct={expected}"


def test_the_model_keeps_the_first_answer_of_a_question() -> None:
    """Records a wrong answer and then a right one for the same question. A pass says the model counts the first
    only, as the app must; a failure means an app that lets answers be replaced would agree with the model."""
    session = SessionModel(learner=0, session_id="s", question_count=2)

    session.record("Q1", frozenset({"a"}), False)
    session.record("Q1", frozenset({"b"}), True)

    assert session.answers == {"Q1": False}, f"the first answer must stand, the model holds {session.answers}"
    assert session.selected == {"Q1": frozenset({"a"})}, f"the first selection must stand: {session.selected}"


@pytest.mark.parametrize(
    ("answers", "count", "mark", "passed"),
    [
        ({"Q1": True}, 2, 50, None),
        ({"Q1": True, "Q2": False}, 2, 50, True),
        ({"Q1": True, "Q2": False}, 2, 51, False),
        ({}, 0, 50, True),
    ],
)
def test_expected_result_judges_a_session_only_when_it_is_complete(
    answers: dict[str, bool], count: int, mark: int, passed: bool | None
) -> None:
    """An open session, a finished one exactly on the pass mark, one just under it, and a session without
    questions. A pass says the model gives no verdict early and takes "exactly on the mark" as passed; a failure
    means the state machine test would demand a wrong verdict from the app."""
    session = SessionModel(learner=0, session_id="s", question_count=count, answers=dict(answers))

    result = expected_result(session, mark)

    assert result == {
        "question_count": count,
        "answered": len(answers),
        "correct": sum(answers.values()),
        "pass_mark_percent": mark,
        "passed": passed,
    }, f"{sum(answers.values())} of {count} at {mark} percent must give passed={passed}: {result}"


def test_expected_progress_sums_one_learner_answers_per_objective() -> None:
    """Two sessions of learner 0 and one of learner 1 on the same questions. A pass says the model sums only the
    asked learner's answers, over all its sessions, per objective, and leaves out objectives without answers; a
    failure means the state machine test could not see an app that mixes learners or drops a session."""
    bank = {"Q1": _question("Q1", "OBJ-1"), "Q2": _question("Q2", "OBJ-1"), "Q3": _question("Q3", "OBJ-2")}
    sessions = [
        SessionModel(0, "a", 2, answers={"Q1": True, "Q3": False}),
        SessionModel(0, "b", 2, answers={"Q1": False, "Q2": True}),
        SessionModel(1, "c", 3, answers={"Q1": True, "Q2": True, "Q3": True}),
    ]

    progress = expected_progress(sessions, 0, bank)

    assert progress == [
        {"objective_id": "OBJ-1", "answered": 3, "correct": 2},
        {"objective_id": "OBJ-2", "answered": 1, "correct": 0},
    ], f"learner 0 answered OBJ-1 three times (two right) and OBJ-2 once (wrong): {progress}"
    assert expected_progress(sessions[:0], 0, bank) == [], "a learner without answers has no progress rows"
