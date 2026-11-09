"""A paraphrase relation run: search every question of a pair set and its rewording, and judge whether search
treats them alike.

Both texts of a pair are searched twice. The primary pass sends them as a user does, so the app's own number of
hits and lowest score apply; the answer-kept relation is judged on it. The diagnostic pass asks for the same number
of hits with no lowest score: it shows which losses the lowest score causes, and the top-k stable relation is judged
on it, because the lowest score cuts the two lists to different lengths and would mix its own effect into the
overlap.

    poetry run python -m evalharness.evaluation.paraphrase_run datasets/<course>/paraphrase_v0.toml

Run it from evaluation-harness/ on the stack with the course ingested; no model is called. The run is written to
runs/relations/ as a new JSON file and its summary is printed. Exit code 0 when the run completed, whatever its
verdicts; 2 when it was refused.
"""

import argparse
import asyncio
import logging
import subprocess
import sys
import uuid
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import structlog
from pydantic import JsonValue

from evalharness.corpus_db import CorpusReader
from evalharness.datasets.paraphrase_set import ParaphrasePair, ParaphraseSet, load_paraphrase_set
from evalharness.evaluation.paraphrase_record import PairPass, PairRecord, ParaphraseRunRecord, write_record
from evalharness.evaluation.record import PassName
from evalharness.evaluation.retrieval import (
    Corpus,
    RunRefused,
    Searched,
    SearchResult,
    pass_result,
    read_app_settings,
    search,
)
from evalharness.relations.paraphrase import (
    PairOutcome,
    RelationSummary,
    answer_kept,
    answer_kept_verdict,
    pair_outcome,
    summarise,
    top_k_stable_verdict,
)
from evalharness.settings import HarnessSettings, load_stack_settings
from evalharness.stats.intervals import share_interval
from evalharness.transport import SutTransport

RUNS_DIR = Path("runs/relations")

# The extra request fields of each pass: none for the primary pass, no lowest score for the diagnostic one.
_PASSES: dict[PassName, dict[str, JsonValue]] = {"primary": {}, "diagnostic": {"min_score": -1}}


def check_gold_sections(pairs: ParaphraseSet, section_ids: set[str]) -> None:
    """Refuse the run when a gold section of the pairs is not a section of the stored course: no search could find
    it, so every such pair would count as "neither" and hide what the rewording does."""
    missing = sorted({pair.gold_section for pair in pairs.pairs} - section_ids)
    if missing:
        raise RunRefused("gold_section_missing", f"gold sections {missing} are not sections of the stored course")


async def run(
    pairs: ParaphraseSet,
    sut: SutTransport,
    corpus: Corpus,
    *,
    pair_file: str,
    harness_commit: str,
    harness_dirty: bool,
) -> ParaphraseRunRecord:
    """Check the gold sections, make both passes over every pair one search at a time, and return the run's record.

    The diagnostic pass leaves the number of hits to the app, as the primary pass does, so both are scored over the
    same k places.
    """
    course_id = pairs.course_id
    started_at = datetime.now(UTC)
    check_gold_sections(pairs, corpus.section_ids(course_id))
    snapshot = corpus.snapshot(course_id)
    app_settings = await read_app_settings(sut)
    searched: dict[PassName, list[tuple[Searched, Searched]]] = {}
    for name, extra in _PASSES.items():
        searched[name] = [
            (
                await search(sut, course_id, pair.question, dict(extra)),
                await search(sut, course_id, pair.paraphrase, dict(extra)),
            )
            for pair in pairs.pairs
        ]
    reported = _first_result(searched["primary"])
    # The number of places both passes are scored over; when every search of the primary pass failed, the
    # diagnostic pass reports it. With no successful search at all every pair is "failed" and k is never used.
    any_result = reported or _first_result(searched["diagnostic"])
    k = any_result.k if any_result is not None else 0
    records = tuple(
        PairRecord(
            case_id=pair.case_id,
            gold_section=pair.gold_section,
            passes={name: _pair_pass(pair, searched[name][i], k) for name in _PASSES},
        )
        for i, pair in enumerate(pairs.pairs)
    )
    return ParaphraseRunRecord(
        run_id=uuid.uuid4().hex,
        started_at=started_at,
        harness_commit=harness_commit,
        harness_dirty=harness_dirty,
        pair_file=pair_file,
        set_sha256=pairs.sha256,
        course_id=course_id,
        corpus_snapshot=snapshot,
        app_settings=app_settings,
        embedding_space=reported.embedding_space if reported is not None else None,
        k=reported.k if reported is not None else None,
        min_score=reported.min_score if reported is not None else None,
        summaries={name: summaries(pairs.pairs, records, name, pairs.rules.min_overlap) for name in _PASSES},
        pairs=records,
    )


def _first_result(searched: Sequence[tuple[Searched, Searched]]) -> SearchResult | None:
    """Return the first successful search result of a pass, or None when every search of it failed."""
    return next((one.result for pair in searched for one in pair if one.result is not None), None)


def _pair_pass(pair: ParaphrasePair, searched: tuple[Searched, Searched], k: int) -> PairPass:
    """Score both searches of one pair against its gold section and classify the pair."""
    question = pass_result(pair.gold_section, searched[0], k)
    rewording = pass_result(pair.gold_section, searched[1], k)
    outcome = pair_outcome(
        question.score,
        rewording.score,
        [hit.chunk_id for hit in question.hits],
        [hit.chunk_id for hit in rewording.hits],
        k,
    )
    return PairPass(question=question, rewording=rewording, outcome=outcome)


def summaries(
    pairs: Sequence[ParaphrasePair], records: Sequence[PairRecord], name: PassName, min_overlap: float
) -> dict[str, RelationSummary]:
    """Summarise one pass over all pairs, then over the pairs whose question lost a key term and the others."""
    groups: defaultdict[str, list[PairOutcome]] = defaultdict(list)
    for pair, record in zip(pairs, records, strict=True):
        outcome = record.passes[name].outcome
        groups["all"].append(outcome)
        groups["key term removed" if pair.term_removed else "no key term in question"].append(outcome)
    return {key: summarise(group, min_overlap) for key, group in groups.items()}


def main(argv: Sequence[str]) -> int:
    """Run the command and return its exit code: 0 when the run completed, whatever its verdicts; 2 when refused."""
    parser = argparse.ArgumentParser(prog="python -m evalharness.evaluation.paraphrase_run")
    parser.add_argument("pair_file", type=Path, help="the paraphrase pair file")
    args = parser.parse_args(argv)
    # One log line per request would bury the summary; failed requests still show as warnings.
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))
    settings = HarnessSettings()
    pairs = load_paraphrase_set(args.pair_file)
    commit, dirty = _harness_version()
    try:
        record = asyncio.run(_run(pairs, args.pair_file, settings, commit, dirty))
    except RunRefused as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    path = write_record(record, RUNS_DIR)
    print_summary(record, pairs, path)
    return 0


async def _run(
    pairs: ParaphraseSet, pair_file: Path, settings: HarnessSettings, commit: str, dirty: bool
) -> ParaphraseRunRecord:
    """Make the run against the stack: searches over HTTP, the corpus read through the read-only role."""

    with CorpusReader(settings, load_stack_settings(settings.stack_env_file)) as corpus:
        async with SutTransport(settings) as sut:
            return await run(
                pairs, sut, corpus, pair_file=pair_file.as_posix(), harness_commit=commit, harness_dirty=dirty
            )


def _harness_version() -> tuple[str, bool]:
    """Return the git commit of the checkout and whether the harness folder has changes not yet committed."""
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "--", "."], capture_output=True, text=True, check=True)
    return commit, bool(status.stdout.strip())


def print_summary(record: ParaphraseRunRecord, pairs: ParaphraseSet, path: Path) -> None:
    """Print the run's settings, a table per pass, both relation verdicts and the pairs that lost their answer."""
    rules = pairs.rules
    dirty = " + uncommitted changes" if record.harness_dirty else ""
    print(
        f"run {record.run_id[:8]}  {record.course_id}  {len(record.pairs)} pairs  "
        f"harness {record.harness_commit[:8]}{dirty}"
    )
    print(f"record: {path}")
    labels: dict[PassName, str] = {"primary": f"min_score={record.min_score}", "diagnostic": "no lowest score"}
    for name, label in labels.items():
        print(f"\n{name} pass (k={record.k}, {label})")
        print(
            f"  {'slice':<26}{'pairs':>6}{'recall q':>10}{'recall rw':>11}{'kept':>6}{'lost':>6}{'gained':>8}"
            f"{'neither':>9}{'stable':>8}{'failed':>8}"
        )
        for key, s in record.summaries[name].items():
            measured = s.pairs - s.failed
            recall_q = (s.kept + s.lost) / measured if measured else 0.0
            recall_rw = (s.kept + s.gained) / measured if measured else 0.0
            print(
                f"  {key:<26}{s.pairs:>6}{recall_q:>10.3f}{recall_rw:>11.3f}{s.kept:>6}{s.lost:>6}{s.gained:>8}"
                f"{s.neither:>9}{s.stable:>8}{s.failed:>8}"
            )
    print()
    for name in ("primary", "diagnostic"):
        _print_answer_kept(record.summaries[name]["all"], name, rules.min_meaningful_change)
    _print_stable(record.summaries["diagnostic"]["all"], rules.min_overlap, rules.stable_share_min, record.k)
    for name in ("primary", "diagnostic"):
        lost = [p.case_id for p in record.pairs if p.passes[name].outcome.found == "lost"]
        print(f"lost in the {name} pass: {' '.join(lost) or 'none'}")


def _print_answer_kept(s: RelationSummary, name: PassName, min_meaningful: float) -> None:
    """Print the answer-kept relation of one pass: the change of recall, its interval, McNemar and the verdict."""
    if s.failed:
        print(f"answer kept ({name}): invalid, {s.failed} pair(s) had a failed search; no verdict")
        return
    change, flips = answer_kept(s)
    interval = "none" if change.interval is None else f"{change.interval.low:+.3f}..{change.interval.high:+.3f}"
    p = "n/a" if flips.p_value is None else f"{flips.p_value:.3f}"
    verdict = answer_kept_verdict(s, min_meaningful)
    print(
        f"answer kept ({name}): recall change {change.change:+.3f}, 95% CI {interval}, lost {flips.lost} "
        f"gained {flips.gained}, McNemar p={p}, smallest meaningful change {min_meaningful}: {verdict}"
    )


def _print_stable(s: RelationSummary, min_overlap: float, stable_share_min: float, k: int | None) -> None:
    """Print the top-k stable relation: the share of stable pairs with its interval and the verdict."""
    if s.failed:
        print(f"top-k stable (diagnostic): invalid, {s.failed} pair(s) had a failed search; no verdict")
        return
    interval = share_interval(s.stable, s.pairs)
    print(
        f"top-{k} stable (diagnostic): {s.stable}/{s.pairs} pairs share >= {min_overlap} of their passages = "
        f"{s.stable / s.pairs:.3f}, 95% CI {interval.low:.3f}-{interval.high:.3f}, target {stable_share_min}: "
        f"{top_k_stable_verdict(s, stable_share_min)}"
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
