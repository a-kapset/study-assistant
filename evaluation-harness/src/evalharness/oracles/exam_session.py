"""What practice sessions must report, computed by the harness from the answers it sent and the bank's own key.

A check keeps one ``SessionModel`` per session it started: the questions the app put into the session and, for
each question it answered, whether the first accepted answer was correct by the bank file. From these models the
functions here give the result every session must report and the progress every learner must have. They know
nothing about how the app stores or counts answers, so a counting defect in the app cannot repeat itself here.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from pydantic import JsonValue

from evalharness.datasets.question_bank import BankQuestion


@dataclass
class SessionModel:
    """One session as the check knows it: whose it is, what it holds and what was answered."""

    learner: int
    session_id: str
    # How many questions the app said the session holds.
    question_count: int
    # The learning objective the session was started on; None for a session on the whole bank.
    objective_id: str | None = None
    # Question id to whether its first accepted answer was correct; a later answer never changes it.
    answers: dict[str, bool] = field(default_factory=dict)
    # Question id to the options of that first answer, as the bank writes them.
    selected: dict[str, frozenset[str]] = field(default_factory=dict)

    def record(self, item_id: str, chosen: frozenset[str], correct: bool) -> None:
        """Keep the first accepted answer of a question; a question that already has one is left as it is."""
        if item_id not in self.answers:
            self.answers[item_id] = correct
            self.selected[item_id] = chosen


def is_correct(chosen: Iterable[str], question: BankQuestion) -> bool:
    """Tell whether the chosen options are exactly the key of the question, whatever their order."""
    return frozenset(chosen) == question.keys


def expected_result(session: SessionModel, pass_mark_percent: int) -> dict[str, JsonValue]:
    """Return the result the session must report.

    ``passed`` is set only when every question has an answer; then it says whether the correct answers reach the
    pass mark, compared in whole numbers so that a session exactly on the mark is passed.
    """
    answered = len(session.answers)
    correct = sum(session.answers.values())
    complete = answered == session.question_count
    return {
        "question_count": session.question_count,
        "answered": answered,
        "correct": correct,
        "pass_mark_percent": pass_mark_percent,
        "passed": correct * 100 >= pass_mark_percent * session.question_count if complete else None,
    }


def expected_progress(
    sessions: Iterable[SessionModel], learner: int, bank: Mapping[str, BankQuestion]
) -> list[dict[str, JsonValue]]:
    """Return the progress rows a learner must have: per objective with at least one answer, how many answers
    the learner gave over all its sessions and how many were correct, ordered by objective id.

    Sessions of other learners are passed in too and must not count: the caller gives every session it knows.
    """
    counts: dict[str, list[int]] = {}
    for session in sessions:
        if session.learner != learner:
            continue
        for item_id, correct in session.answers.items():
            count = counts.setdefault(bank[item_id].objective_id, [0, 0])
            count[0] += 1
            count[1] += int(correct)
    return [
        {"objective_id": objective_id, "answered": counts[objective_id][0], "correct": counts[objective_id][1]}
        for objective_id in sorted(counts)
    ]
