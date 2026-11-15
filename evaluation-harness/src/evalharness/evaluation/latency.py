"""A latency run: how long one search of a course takes, as a whole and stage by stage.

    poetry run python -m evalharness.evaluation.latency datasets/<course>/<gold set>.toml
    poetry run python -m evalharness.evaluation.latency <declaration> --budget-s 1.0
    poetry run python -m evalharness.evaluation.latency <declaration> --allow-degraded

Run it from evaluation-harness/ on the running stack; no model is called. The questions are those of the gold set's
main result, sent one at a time as a learner sends them, so latency is measured on the requests recall is measured
on. The first requests after a start are slower (the first database connection, the first call to a model server),
so the first questions are sent once before the measurement and left out; how many is fixed before the run, not
chosen from its numbers. The time of each stage comes from the request's trace in the app's trace store, so the
numbers say where a search spends its time, not only how long the caller waited.

One client and one request at a time: the numbers say how long one learner waits, and nothing about many learners
at once. The run is written to runs/latency/ as a new JSON file and its summary is printed. Exit code 0 when the run
completed, whatever its numbers; 2 when it was refused. With --budget-s the summary also says whether the 95th
percentile of the search stage is within that many seconds. With --allow-degraded the run also starts while a model
server of the search is down, to measure the weaker path the app then answers on.
"""

import argparse
import asyncio
import logging
import os
import platform
import statistics
import subprocess
import sys
import uuid
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import structlog
from pydantic import BaseModel, ConfigDict, JsonValue

from evalharness.datasets.retrieval_gold import GoldCase, GoldSet, GoldSetError, build_gold_set, load_declaration
from evalharness.evaluation.retrieval import (
    RunRefused,
    Searched,
    bank_folder,
    check_ready,
    read_app_settings,
    search,
    search_path,
    select_cases,
)
from evalharness.settings import HarnessSettings, load_stack_settings
from evalharness.traces import Span, Trace, TraceNotFoundError, TraceReader
from evalharness.transport import SutTransport

RUNS_DIR = Path("runs/latency")
# How many questions are sent once before the measurement and left out of it.
WARM_UP_REQUESTS = 10
# Longest wait for a whole trace: the app sends its spans to the store in batches every few seconds.
_TRACE_WAIT_S = 30.0
# The stages in the order they are printed: the caller's wait, the app's work on the request, the search inside
# it, and the calls the search makes.
STAGES = ("client", "request", "search", "embedder", "database", "rerank")
# The span the app writes around a whole search, and the one around its call to the reranker.
_SEARCH_SPAN = "search"
_RERANK_SPAN = "rerank"

type BudgetVerdict = Literal["within budget", "over budget"]


class TraceShapeError(Exception):
    """A trace does not look like the trace of one search, so its stage times cannot be read."""


class _Strict(BaseModel):
    """Base for the record: an unknown key is an error, and a record never changes after it is made."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class RequestLatency(_Strict):
    """One search of the run: which question, whether it was measured, and the seconds of each stage."""

    case_id: str
    # The same id is in the app's log lines for this search.
    request_id: str
    # The request's trace in the app's trace store; None when the response named none.
    trace_id: str | None
    # True for a request sent before the measurement; it is left out of the summary.
    warm_up: bool
    # Why the request has no stage times: the search's own failure, "no_trace_id", "trace_not_found", or
    # "trace_shape: ..." with what was wrong; None when the times were read.
    error: str | None
    # Why the app answered on a weaker path than the configured one, as the app names it; None otherwise.
    degraded_reason: str | None
    # Seconds per stage; a stage the search did not go through is missing. Empty when ``error`` is set.
    stages: dict[str, float]


class StageSummary(_Strict):
    """One stage over the measured requests that went through it: how many, and their seconds."""

    n: int
    p50: float
    p95: float
    max: float


class LatencyRecord(_Strict):
    """Everything one latency run measured, and what is needed to say which code and search it measured where."""

    run_id: str
    # In UTC.
    started_at: datetime
    # The harness's git commit, and whether its files had changes not yet committed (the harness folder only).
    harness_commit: str
    harness_dirty: bool
    declaration_file: str
    course_id: str
    sets: tuple[str, ...]
    # The settings the app reported at GET /v1/config before the first search; "git_commit" names the app's code.
    app_settings: dict[str, JsonValue]
    # The search path the app reported in its responses; None when no measured search succeeded.
    strategy: str | None
    reranked: bool | None
    # The machine the requests were sent from. Latency belongs to a machine: the same stack is several times
    # faster or slower elsewhere, so a number is read only together with this line.
    client_machine: str
    # How many requests were sent before the measurement and left out.
    warm_up: int
    # In the order they were sent, the warm-up requests first.
    requests: tuple[RequestLatency, ...]
    # Per stage, over the measured requests whose times were read.
    summary: dict[str, StageSummary]


def stage_times(trace: Trace) -> dict[str, float]:
    """Read from the trace of one search how long the request, the search stage and each kind of call inside the
    search took, in seconds; several calls of one kind add up.

    Raises ``TraceShapeError`` when the trace has no single request span or search span, or when the search made a
    call that is none of the known kinds: its time would otherwise be left out without a sign.
    """
    requests = [span for span in trace.spans if span.parent_id is None]
    searches = trace.named(_SEARCH_SPAN)
    if len(requests) != 1 or len(searches) != 1:
        raise TraceShapeError(f"{len(requests)} request span(s) and {len(searches)} search span(s), expected one each")
    times = {"request": requests[0].duration_s, "search": searches[0].duration_s}
    for call in trace.children(searches[0]):
        stage = _stage_of(call)
        times[stage] = times.get(stage, 0.0) + call.duration_s
    return times


def _stage_of(call: Span) -> str:
    """Name the stage a call made inside the search belongs to, from what its instrumentation recorded: a database
    query, the call to the reranker, or an HTTP call, which at that level only the embedding client makes."""
    if "db.system" in call.attributes:
        return "database"
    if call.name == _RERANK_SPAN:
        return "rerank"
    if "http.method" in call.attributes:
        return "embedder"
    raise TraceShapeError(f"a call of an unknown kind inside the search: {call.name!r}")


def percentile(values: Sequence[float], p: int) -> float:
    """Return the ``p``-th percentile of ``values``: the value at position (n - 1) * p / 100 of the sorted values,
    counted from 0, taken between its two neighbours in proportion when the position is not whole."""
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[p - 1]


def summarise(requests: Sequence[RequestLatency]) -> dict[str, StageSummary]:
    """Sum up each stage over the measured requests whose times were read; warm-up requests and requests without
    times are left out, and a stage no request went through gets no entry."""
    summary: dict[str, StageSummary] = {}
    for stage in STAGES:
        values = [one.stages[stage] for one in requests if not one.warm_up and stage in one.stages]
        if values:
            summary[stage] = StageSummary(
                n=len(values), p50=percentile(values, 50), p95=percentile(values, 95), max=max(values)
            )
    return summary


def budget_verdict(summary: dict[str, StageSummary], budget_s: float) -> BudgetVerdict | None:
    """Say whether the 95th percentile of the search stage is within ``budget_s`` seconds; None when no search
    was measured. The budget is for the search stage: the time a learner waits for it in every mode, before any
    model is called."""
    search_stage = summary.get(_SEARCH_SPAN)
    if search_stage is None:
        return None
    return "within budget" if search_stage.p95 <= budget_s else "over budget"


async def run(
    gold: GoldSet,
    cases: Sequence[GoldCase],
    sut: SutTransport,
    traces: TraceReader,
    *,
    declaration_file: str,
    harness_commit: str,
    harness_dirty: bool,
    client_machine: str,
    warm_up: int = WARM_UP_REQUESTS,
    allow_degraded: bool = False,
) -> LatencyRecord:
    """Send the first ``warm_up`` questions once, then every question of ``cases`` once, one at a time, and return
    the record with the stage times of each request.

    The traces are read after the last search, so reading them never competes with a measured request. With
    ``allow_degraded`` the run also starts while a model server of the search is down.
    """
    course_id = gold.declaration.course_id
    started_at = datetime.now(UTC)
    app_settings = await read_app_settings(sut)
    await check_ready(sut, allow_degraded=allow_degraded)

    planned = [(case, True) for case in cases[:warm_up]] + [(case, False) for case in cases]
    sent = [(case, warm, await search(sut, course_id, case.query, {})) for case, warm in planned]
    requests = tuple([await _request(case, warm, searched, traces) for case, warm, searched in sent])

    reported = next((one.result for _, warm, one in sent if not warm and one.result is not None), None)
    return LatencyRecord(
        run_id=uuid.uuid4().hex,
        started_at=started_at,
        harness_commit=harness_commit,
        harness_dirty=harness_dirty,
        declaration_file=declaration_file,
        course_id=course_id,
        sets=tuple(sorted({case.exam_set for case in cases})),
        app_settings=app_settings,
        strategy=reported.strategy if reported is not None else None,
        reranked=reported.reranked if reported is not None else None,
        client_machine=client_machine,
        warm_up=min(warm_up, len(cases)),
        requests=requests,
        summary=summarise(requests),
    )


async def _request(case: GoldCase, warm_up: bool, searched: Searched, traces: TraceReader) -> RequestLatency:
    """Record one search: the caller's wait and the stage times of its trace, or why they could not be read."""
    stages: dict[str, float] = {}
    error = searched.error
    if error is None:
        if searched.trace_id is None:
            error = "no_trace_id"
        else:
            try:
                stages = {"client": searched.elapsed_s, **stage_times(await traces.fetch(searched.trace_id))}
            except TraceNotFoundError:
                error = "trace_not_found"
            except TraceShapeError as shape:
                error = f"trace_shape: {shape}"
    return RequestLatency(
        case_id=case.case_id,
        request_id=searched.request_id,
        trace_id=searched.trace_id,
        warm_up=warm_up,
        error=error,
        degraded_reason=searched.result.degraded_reason if searched.result is not None else None,
        stages=stages,
    )


def write_record(record: LatencyRecord, runs_dir: Path) -> Path:
    """Write the record as a new JSON file in ``runs_dir`` and return its path; an existing file is never replaced."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{record.started_at:%Y%m%dT%H%M%SZ}-{record.course_id}-{record.run_id[:8]}.json"
    with path.open("x", encoding="utf-8") as file:
        file.write(record.model_dump_json(indent=2))
    return path


def main(argv: Sequence[str]) -> int:
    """Run the command and return its exit code."""
    parser = argparse.ArgumentParser(prog="python -m evalharness.evaluation.latency")
    parser.add_argument("declaration", type=Path, help="the gold set's declaration file")
    parser.add_argument("--sets", nargs="+", help="exam sets to run (default: the sets of the declared main result)")
    parser.add_argument(
        "--budget-s", type=float, help="say whether the 95th percentile of the search stage is within this time"
    )
    parser.add_argument(
        "--allow-degraded",
        action="store_true",
        help="start although a model server of the search is down, to measure the weaker path",
    )
    args = parser.parse_args(argv)

    # One log line per request would bury the summary; failed requests still show as warnings.
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))
    settings = HarnessSettings()
    stack = load_stack_settings(settings.stack_env_file)
    declaration = load_declaration(args.declaration)
    # The trace store's port is published on the same host as the stack's database port.
    trace_store = f"http://{settings.db_host}:{stack.jaeger_published_port}"
    try:
        folder = bank_folder(declaration, stack.local_course_id, stack.local_course_sources, settings.stack_env_file)
        gold = build_gold_set(declaration, folder)
        cases = select_cases(gold, args.sets or declaration.primary.sets, include_held_out=False)
        record = asyncio.run(_run(gold, cases, settings, trace_store, args.declaration, args.allow_degraded))
    except (RunRefused, GoldSetError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    path = write_record(record, RUNS_DIR)
    print_summary(record, path, args.budget_s)
    return 0


async def _run(
    gold: GoldSet,
    cases: Sequence[GoldCase],
    settings: HarnessSettings,
    trace_store: str,
    declaration_file: Path,
    allow_degraded: bool,
) -> LatencyRecord:
    """Make the run against the stack: searches over HTTP, the traces from the stack's trace store."""
    commit, dirty = _harness_version()
    async with SutTransport(settings) as sut, TraceReader(trace_store, wait_s=_TRACE_WAIT_S) as traces:
        return await run(
            gold,
            cases,
            sut,
            traces,
            declaration_file=declaration_file.as_posix(),
            harness_commit=commit,
            harness_dirty=dirty,
            client_machine=_client_machine(),
            allow_degraded=allow_degraded,
        )


def _harness_version() -> tuple[str, bool]:
    """Return the git commit of the checkout and whether the harness folder has changes not yet committed."""
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "--", "."], capture_output=True, text=True, check=True)
    return commit, bool(status.stdout.strip())


def _client_machine() -> str:
    """Describe this machine by its operating system, processor and number of logical processors."""
    return f"{platform.system()} {platform.release()}, {platform.processor()}, {os.cpu_count()} logical CPUs"


def print_summary(record: LatencyRecord, path: Path, budget_s: float | None) -> None:
    """Print what was measured where, one line per stage, the requests left out and why, and the budget line."""
    dirty = " + uncommitted changes" if record.harness_dirty else ""
    sets = " ".join(record.sets)
    print(f"run {record.run_id[:8]}  {record.course_id}  sets {sets}  harness {record.harness_commit[:8]}{dirty}")
    print(f"record: {path}")
    print(f"search: {search_path(record.strategy, record.reranked)}")
    print(f"sent from: {record.client_machine}; one request at a time")
    measured = [one for one in record.requests if not one.warm_up]
    print(f"first {record.warm_up} request(s) left out as warm-up, {len(measured)} measured")
    first = record.requests[0].stages.get("client") if record.requests else None
    if first is not None:
        print(f"first request of the run: {first:.3f} s for the caller")

    print(f"\n  {'stage':<10}{'n':>6}{'p50, s':>10}{'p95, s':>10}{'max, s':>10}")
    for stage, s in record.summary.items():
        print(f"  {stage:<10}{s.n:>6}{s.p50:>10.3f}{s.p95:>10.3f}{s.max:>10.3f}")
    failed = Counter(one.error.split(":", 1)[0] for one in measured if one.error is not None)
    if failed:
        listed = ", ".join(f"{reason} {n}" for reason, n in sorted(failed.items()))
        print(f"  requests without stage times, left out: {listed}")
    weaker = Counter(one.degraded_reason for one in measured if one.degraded_reason is not None)
    if weaker:
        listed = ", ".join(f"{reason} {n}" for reason, n in sorted(weaker.items()))
        print(f"  answered on a weaker path, not the configured search: {listed}")

    timed = [one for one in measured if _SEARCH_SPAN in one.stages]
    if timed:
        slowest = max(timed, key=lambda one: one.stages[_SEARCH_SPAN])
        print(f"\nslowest search: {slowest.case_id}, {slowest.stages[_SEARCH_SPAN]:.3f} s, trace {slowest.trace_id}")
    if budget_s is not None:
        verdict = budget_verdict(record.summary, budget_s)
        if verdict is None:
            print(f"no budget line: no search was measured (budget {budget_s:.3f} s)")
        else:
            p95 = record.summary[_SEARCH_SPAN].p95
            print(f"p95 of the search stage = {p95:.3f} s (n={len(timed)}), budget {budget_s:.3f} s: {verdict}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
