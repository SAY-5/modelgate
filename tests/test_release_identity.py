"""The public API must identify the release that was packaged."""

import tomllib
from pathlib import Path


def test_health_and_openapi_report_packaged_release(client):
    metadata = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    packaged_version = metadata["project"]["version"]
    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["version"] == packaged_version
    schema = client.get("/openapi.json")
    assert schema.status_code == 200
    assert schema.json()["info"]["version"] == packaged_version
