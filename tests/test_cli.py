"""Run the installed `modelgate serve` command and check the address Uvicorn reports."""

from __future__ import annotations

from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from modelgate.cli import main
from tests.test_startup import serve


def modelgate_serve(*args: str):
    return lambda port: ["uv", "run", "modelgate", "serve", "--port", str(port), *args]


def test_modelgate_serve_defaults_to_loopback(tmp_path):
    with serve(tmp_path, command=modelgate_serve()) as (process, port, log):
        assert process.poll() is None, log.read_text()
        assert f"Uvicorn running on http://127.0.0.1:{port}" in log.read_text()


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "localhost"])
@pytest.mark.parametrize("token", [None, "", "   ", "dev-token", " dev-token "])
def test_remote_modelgate_serve_refuses_missing_or_development_token(tmp_path, host, token):
    variables = {} if token is None else {"MODELGATE_ADMIN_TOKEN": token}
    with serve(tmp_path, command=modelgate_serve("--host", host), **variables) as (
        process,
        _port,
        log,
    ):
        assert process.poll() not in (None, 0), log.read_text()
        assert "Remote startup requires" in log.read_text()
        assert "Uvicorn running" not in log.read_text()


@pytest.mark.parametrize("token", [None, "", "   ", "dev-token", " dev-token "])
def test_cli_rejects_unsafe_publication_before_launch(monkeypatch, token):
    # Intercept the network boundary so the regression's red phase never opens a public port.
    if token is None:
        monkeypatch.delenv("MODELGATE_ADMIN_TOKEN", raising=False)
    else:
        monkeypatch.setenv("MODELGATE_ADMIN_TOKEN", token)

    def unsafe_launch(*args, **kwargs):
        pytest.fail("CLI attempted to launch an unprotected public listener")

    monkeypatch.setattr("uvicorn.run", unsafe_launch)
    with pytest.raises(SystemExit) as error:
        main(["serve", "--host", "0.0.0.0"])
    assert error.value.code == 2


def test_remote_modelgate_serve_accepts_literal_configured_token(tmp_path):
    token = "caller-configured-'token'; $(printf should-not-run) $literal"
    command = modelgate_serve("--host", "0.0.0.0")
    with serve(tmp_path, command=command, MODELGATE_ADMIN_TOKEN=token) as (
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


def test_local_modelgate_serve_empty_token_keeps_admin_disabled(tmp_path):
    with serve(tmp_path, command=modelgate_serve(), MODELGATE_ADMIN_TOKEN="") as (
        process,
        port,
        log,
    ):
        assert process.poll() is None, log.read_text()
        with pytest.raises(HTTPError) as error:
            urlopen(f"http://127.0.0.1:{port}/admin/versions", timeout=2)
        assert error.value.code == 503


def test_cli_preserves_explicit_listener_options(monkeypatch):
    # Uvicorn owns the socket; inspect its boundary arguments without opening a listener.
    launched = []
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: launched.append((app, kwargs)))
    monkeypatch.setenv("MODELGATE_ADMIN_TOKEN", "")
    assert main(["serve", "--host", "::1", "--port", "8123", "--log-level", "warning"]) == 0
    assert launched == [
        ("modelgate.serving.app:app", {"host": "::1", "port": 8123, "log_level": "warning"})
    ]


def test_cli_default_listener_is_private_before_launch(monkeypatch):
    launched = []
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: launched.append(kwargs["host"]))
    monkeypatch.delenv("MODELGATE_ADMIN_TOKEN", raising=False)
    assert main(["serve"]) == 0
    assert launched == ["127.0.0.1"]
