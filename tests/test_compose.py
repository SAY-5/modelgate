"""Check Docker's resolved publication rules; no daemon or developer .env needed."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from tests.test_startup import serve

ROOT = Path(__file__).resolve().parents[1]


def compose_config(**variables):
    result = subprocess.run(
        [
            "docker",
            "compose",
            "--env-file",
            "/dev/null",
            "-f",
            str(ROOT / "docker-compose.yml"),
            "config",
            "--format",
            "json",
        ],
        cwd=ROOT,
        env={"PATH": os.environ["PATH"], "COMPOSE_DISABLE_ENV_FILE": "1", **variables},
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)["services"]


@pytest.mark.parametrize(
    "service,port", [("modelgate", 8000), ("prometheus", 9090), ("grafana", 3000)]
)
def test_compose_default_publications_are_loopback(service, port):
    ports = compose_config()[service]["ports"]
    assert len(ports) == 1
    assert ports[0].get("host_ip") == "127.0.0.1"
    assert ports[0]["target"] == port


def test_compose_remote_api_does_not_publish_observability():
    services = compose_config(MODELGATE_HOST="0.0.0.0", MODELGATE_ADMIN_TOKEN="caller-token")
    assert services["modelgate"]["ports"][0].get("host_ip") == "0.0.0.0"
    for service in ("prometheus", "grafana"):
        assert services[service]["ports"][0].get("host_ip") == "127.0.0.1"


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.0.2.9"])
@pytest.mark.parametrize("token", [None, "", "   ", "dev-token"])
def test_composed_remote_command_refuses_unsafe_token(tmp_path, host, token):
    variables = {"MODELGATE_HOST": host}
    if token is not None:
        variables["MODELGATE_ADMIN_TOKEN"] = token
    service = compose_config(**variables)["modelgate"]
    assert service["environment"]["MODELGATE_HOST"] == service["ports"][0]["host_ip"]
    # Execute the actual rendered command and environment; a missing/bypassed validator fails.
    with serve(tmp_path, command=service["command"], **service["environment"]) as (
        process,
        _port,
        log,
    ):
        assert process.poll() not in (None, 0), log.read_text()
        assert "Remote startup requires" in log.read_text()
        assert "Uvicorn running" not in log.read_text()


@pytest.mark.parametrize("host", ["127.0.0.1", "0.0.0.0"])
def test_composed_command_keeps_container_network_listener(tmp_path, host):
    service = compose_config(MODELGATE_HOST=host, MODELGATE_ADMIN_TOKEN="caller-token")["modelgate"]
    assert service["environment"]["MODELGATE_HOST"] == service["ports"][0]["host_ip"]
    with serve(tmp_path, command=service["command"], **service["environment"]) as (
        process,
        port,
        log,
    ):
        assert process.poll() is None, log.read_text()
        assert f"Uvicorn running on http://0.0.0.0:{port}" in log.read_text()
