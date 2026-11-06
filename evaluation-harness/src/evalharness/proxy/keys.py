"""The name of a recording: the same request always gets the same key, and any change of content gets another.

A repeated answer run asks the same questions again, so its model calls are identical to the first ones. The runner
marks each repetition at the end of the request id it sends to the app; the app passes that id on to the proxy,
and the proxy puts the repetition into the key, so every repetition gets its own recording.
"""

import hashlib
import json
import re

# "<id>.rep2": the second repetition. An id without the mark is the first repetition.
_REPETITION = re.compile(r"\.rep([1-9][0-9]*)$")


def repetition_suffix(repetition: int) -> str:
    """Return what the runner appends to a request id for ``repetition`` (counted from 1); nothing for the first.

    The first repetition keeps the plain id, so its key is the key of a run made without repetitions and the
    recordings of such a run serve it.
    """
    if repetition < 1:
        raise ValueError(f"repetitions are counted from 1, got {repetition}")
    return "" if repetition == 1 else f".rep{repetition}"


def repetition_of(request_id: str | None) -> int:
    """Return the repetition a request id is marked with; 1 when it has no mark or there is no id."""
    if request_id is None:
        return 1
    match = _REPETITION.search(request_id)
    return 1 if match is None else int(match.group(1))


def request_key(surface: str, body: bytes, repetition: int = 1) -> str:
    """Return ``surface`` plus the SHA-256 of ``body`` in canonical form, and the repetition when it is not the first.

    The body is compared by content: the order of JSON fields and spacing mean nothing, and a library update may
    change them. Every field counts, so a changed prompt, passage, model or reply schema is a different request.
    Headers are left out: each call has a new request id, and the key of the provider must not matter. Only the
    repetition mark of the request id counts, as a suffix of the key.
    """
    canonical = json.dumps(json.loads(body), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    suffix = "" if repetition == 1 else f"-rep{repetition}"
    return f"{surface}-{digest}{suffix}"
