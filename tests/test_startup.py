"""Exercise the real Make launcher, including the address passed to Uvicorn."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest

ROOT = Path(__file__).resolve().parents[1]


def clean_env(**overrides: str) -> dict[str, str]:
    # Keep tool lookup, not a developer's tokens, .env settings, or Python startup hooks.
    return {"PATH": os.environ["PATH"], "UV_NO_SYNC": "1", **overrides}


@contextmanager
def serve(
    tmp_path: Path,
    command: list[str] | Callable[[int], list[str]] | None = None,
    **overrides: str,
):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    if callable(command):
        command = command(port)
    log_path = tmp_path / "server.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command or ["make", "--no-print-directory", "serve", f"PORT={port}"],
            cwd=ROOT,
            env=clean_env(**overrides, PORT=str(port)),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    break
                try:
                    with urlopen(f"http://127.0.0.1:{port}/readyz", timeout=0.2) as response:
                        if response.status == 200:
                            break
                except (URLError, TimeoutError):
                    time.sleep(0.05)
            else:
                pytest.fail(f"server did not become ready: {log_path.read_text()}")
            yield process, port, log_path
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)


def test_make_serve_defaults_to_loopback(tmp_path):
    # Regresses the old public listener, not just the text in the Makefile.
    with serve(tmp_path) as (process, port, log):
        assert process.poll() is None, log.read_text()
        assert f"Uvicorn running on http://127.0.0.1:{port}" in log.read_text()


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.0.2.9", "localhost"])
@pytest.mark.parametrize("token", [None, "", "   ", "dev-token", " dev-token "])
def test_remote_make_serve_refuses_missing_or_development_token(tmp_path, host, token):
    variables = {"MODELGATE_HOST": host}
    if token is not None:
        variables["MODELGATE_ADMIN_TOKEN"] = token
    with serve(tmp_path, **variables) as (process, _port, log):
        assert process.poll() not in (None, 0), log.read_text()
        assert "Remote startup requires" in log.read_text()
        assert "Uvicorn running" not in log.read_text()


def test_local_explicit_empty_token_leaves_admin_disabled(tmp_path):
    with serve(tmp_path, MODELGATE_ADMIN_TOKEN="") as (process, port, log):
        assert process.poll() is None, log.read_text()
        with pytest.raises(HTTPError) as error:
            urlopen(f"http://127.0.0.1:{port}/admin/versions", timeout=2)
        assert error.value.code == 503


def test_remote_make_serve_accepts_literal_configured_token(tmp_path):
    # Neither shell syntax nor whitespace may be evaluated or rewritten by the launcher.
    token = "caller-configured-'token'; $(printf should-not-run) $literal"
    with serve(tmp_path, MODELGATE_HOST="0.0.0.0", MODELGATE_ADMIN_TOKEN=token) as (
        process,
        port,
        log,
    ):
        assert process.poll() is None, log.read_text()
        assert f"Uvicorn running on http://0.0.0.0:{port}" in log.read_text()
        request = Request(
            f"http://127.0.0.1:{port}/admin/versions", headers={"X-Admin-Token": token}
        )
        with urlopen(request, timeout=2) as response:
            assert response.status == 200
        assert token not in log.read_text()
