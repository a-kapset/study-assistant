"""Unit tests of loading a paraphrase pair file: which pairs are refused before a run, on small files in a temporary
folder instead of the committed set."""

from pathlib import Path

import pytest

from evalharness.datasets.paraphrase_set import load_paraphrase_set

QA = """
course_id = "evalharness_course"
[targets]
answered_min = 0.85
gold_cited_min = 0.75
unsupported_answers_max = 0
[[cases]]
id = "QA-1"
question = "How does boundary value analysis work?"
gold_section = "4.2"
key_terms = ["boundary value analysis"]
[[cases]]
id = "QA-2"
question = "Why do old tests find fewer bugs?"
gold_section = "1.3"
key_terms = ["tests wear out"]
[[cases]]
id = "QA-U1"
question = "How do I write a Jenkins pipeline?"
key_terms = ["jenkins"]
"""


def _write(folder: Path, pairs: str) -> Path:
    """Write the Q&A set above and a pair file with ``pairs`` into ``folder``; return the pair file."""
    (folder / "qa.toml").write_text(QA, encoding="utf-8")
    path = folder / "pairs.toml"
    path.write_text(
        'course_id = "evalharness_course"\nqa_file = "qa.toml"\n'
        "[relations]\nmin_meaningful_change = 0.05\nmin_overlap = 0.6\nstable_share_min = 0.7\n" + pairs,
        encoding="utf-8",
    )
    return path


@pytest.mark.covers("US-015", "AC-1")
@pytest.mark.parametrize(
    "rewording",
    ["When is boundary value analysis worth it?", "How does Boundary Value\nAnalysis work for ages?"],
)
def test_a_rewording_that_still_contains_a_key_term_is_refused(tmp_path: Path, rewording: str) -> None:
    """Writes a pair whose rewording still holds the key term "boundary value analysis", once as written and once
    in other capitals across a line break.

    A pass means the loader refuses such a pair, so every pair that reaches a run really asks without the course
    term. A failure means a rewording with the term would be counted as a term-free rewording, and the relation
    would look better than search really treats a learner who does not know the term. The match rule is the one of
    the Q&A set check, so the two cannot drift apart.
    """
    path = _write(tmp_path, f'[[pairs]]\nid = "QA-1"\nparaphrase = """{rewording}"""\n')

    with pytest.raises(ValueError, match="still contains the key terms") as error:
        load_paraphrase_set(path)

    assert "QA-1" in str(error.value), f"the refusal does not name the pair: {error.value}"


@pytest.mark.covers("US-015", "AC-1")
@pytest.mark.parametrize(("case_id", "reason"), [("QA-9", "no such question"), ("QA-U1", "unanswerable")])
def test_a_pair_without_an_answerable_question_is_refused(tmp_path: Path, case_id: str, reason: str) -> None:
    """Writes a pair for a question id the Q&A set does not have, and one for an unanswerable question.

    A pass means both are refused with the reason, before any search. A failure means a run would score a pair
    with no gold section: it could never keep an answer, and every such pair would count against search.
    """
    path = _write(tmp_path, f'[[pairs]]\nid = "{case_id}"\nparaphrase = "How do I automate a build?"\n')

    with pytest.raises(ValueError, match=reason):
        load_paraphrase_set(path)


@pytest.mark.covers("US-015", "AC-5")
def test_a_pair_says_whether_its_question_held_a_key_term(tmp_path: Path) -> None:
    """Loads two valid pairs: QA-1, whose question contains its key term, and QA-2, whose question does not.

    A pass means the loader marks QA-1 as "term removed" and QA-2 as not, which is what the run's two slices are
    built from. A failure means the slices mix the two kinds of pair, so a search that depends on the course term
    would not show in the slice made to see it.
    """
    path = _write(
        tmp_path,
        '[[pairs]]\nid = "QA-1"\nparaphrase = "How do I pick inputs at the edges of a range?"\n'
        '[[pairs]]\nid = "QA-2"\nparaphrase = "Why do tests run many times stop finding bugs?"\n',
    )

    pairs = {pair.case_id: pair.term_removed for pair in load_paraphrase_set(path).pairs}

    assert pairs == {"QA-1": True, "QA-2": False}, f"term removed per pair: {pairs}"
