"""The name of a recording: the same request always gets the same key, and any change of content gets another."""

import hashlib
import json


def request_key(surface: str, body: bytes) -> str:
    """Return ``surface`` plus the SHA-256 of ``body`` in canonical form.

    The body is compared by content: the order of JSON fields and spacing mean nothing, and a library update may
    change them. Every field counts, so a changed prompt, passage, model or reply schema is a different request.
    Headers are left out: each call has a new request id, and the key of the provider must not matter.
    """
    canonical = json.dumps(json.loads(body), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{surface}-{digest}"
