"""Starts the proxy: python -m evalharness.proxy."""

import uvicorn

from evalharness.proxy.app import build_proxy
from evalharness.settings import ProxySettings

# The port inside the stack's network; only the app reaches it.
_PORT = 8100


def main() -> None:
    """Build the proxy from the environment and serve it until stopped."""
    uvicorn.run(build_proxy(ProxySettings()), host="0.0.0.0", port=_PORT)


if __name__ == "__main__":
    main()
