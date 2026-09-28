"""Safe demo startup: validate host publication before importing the application."""

from __future__ import annotations

import argparse
import os
import sys
from ipaddress import ip_address


def validate_publication(address: str, token: str | None) -> None:
    """Only literal loopback IPs may use the development token or disabled admin API."""
    try:
        local = ip_address(address).is_loopback
    except ValueError:
        # A hostname can resolve elsewhere; never infer trust from its spelling or DNS.
        local = False
    if not local and (not token or not token.strip() or token.strip() == "dev-token"):
        raise ValueError("Remote startup requires a configured non-default admin token")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--container",
        action="store_true",
        help="listen internally on 0.0.0.0; MODELGATE_HOST is the host publication address",
    )
    args = parser.parse_args()
    address = os.environ.get("MODELGATE_HOST") or "127.0.0.1"
    token = os.environ.get("MODELGATE_ADMIN_TOKEN", "dev-token")
    try:
        validate_publication(address, token)
        port = int(os.environ.get("PORT", "8000"))
        if not 1 <= port <= 65535:
            raise ValueError("PORT must be between 1 and 65535")
    except ValueError as error:
        parser.error(str(error))

    # An explicitly empty token stays empty, preserving the application's disabled-admin mode.
    # No shell evaluates either the token or the address, and no token enters argv or logs.
    os.environ.setdefault("MODELGATE_ADMIN_TOKEN", token)
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "uvicorn",
            "modelgate.serving.app:app",
            "--host",
            "0.0.0.0" if args.container else address,
            "--port",
            str(port),
        ],
    )


if __name__ == "__main__":
    main()
