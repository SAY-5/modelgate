"""FastAPI application: prediction, admin, health, and metrics endpoints."""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
import uuid
from contextlib import asynccontextmanager

import torch
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from modelgate import __version__
from modelgate.model.features import encode
from modelgate.serving import metrics
from modelgate.serving.config import Settings
from modelgate.serving.registry import ModelRegistry, NoPrimaryError, UnknownVersionError
from modelgate.serving.schemas import (
    PredictRequest,
    PredictResponse,
    PromoteRequest,
    ShadowRequest,
    rejection_reason,
)
from modelgate.serving.shadow import ShadowRecord, ShadowTracker

log = logging.getLogger("modelgate")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    # One-row inference is faster without the intra-op thread pool spinning up.
    torch.set_num_threads(1)
    registry = ModelRegistry(settings.artifacts_dir)
    shadow_tracker = ShadowTracker(settings.shadow_threshold_minutes, settings.shadow_log_size)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        primary = settings.primary_version
        if primary is None:
            available = registry.available_versions()
            primary = available[0] if available else None
        if primary is not None:
            await asyncio.to_thread(registry.promote, primary)
            log.info("primary model loaded: %s", primary)
        if settings.shadow_version:
            await asyncio.to_thread(registry.set_shadow, settings.shadow_version)
            log.info("shadow model loaded: %s", settings.shadow_version)
        yield

    app = FastAPI(
        title="ModelGate",
        version=__version__,
        description="ETA model serving with shadow runs and zero-drop version swaps.",
        lifespan=lifespan,
    )
    app.state.registry = registry
    app.state.shadow_tracker = shadow_tracker
    app.state.settings = settings

    # ---- auth ------------------------------------------------------------

    def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
        if not settings.admin_token:
            raise HTTPException(503, "admin API disabled: MODELGATE_ADMIN_TOKEN is not set")
        if not x_admin_token or not secrets.compare_digest(x_admin_token, settings.admin_token):
            raise HTTPException(401, "invalid admin token")

    # ---- error handling --------------------------------------------------

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request: Request, exc: RequestValidationError):
        details = []
        reasons: set[str] = set()
        for err in exc.errors():
            reason = rejection_reason(err.get("type", ""))
            reasons.add(reason)
            loc = ".".join(str(p) for p in err.get("loc", ()) if p != "body")
            details.append({"field": loc or "body", "reason": reason, "message": err.get("msg")})
        if request.url.path == "/predict":
            for reason in reasons:
                metrics.INPUT_REJECTIONS.labels(reason=reason).inc()
            primary = registry.primary
            metrics.REQUESTS.labels(
                version=primary.version if primary else "none", outcome="rejected"
            ).inc()
        return JSONResponse(
            status_code=422,
            content={"error": "invalid input", "rejections": details},
        )

    @app.exception_handler(Exception)
    async def on_unhandled(request: Request, exc: Exception):
        if request.url.path == "/predict":
            metrics.DROPPED_REQUESTS.inc()
            primary = registry.primary
            metrics.REQUESTS.labels(
                version=primary.version if primary else "none", outcome="error"
            ).inc()
        log.exception("unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500, content={"error": "internal error"})

    # ---- prediction ------------------------------------------------------

    @app.post("/predict", response_model=PredictResponse)
    async def predict(body: PredictRequest, x_request_id: str | None = Header(default=None)):
        started = time.perf_counter()
        request_id = x_request_id or uuid.uuid4().hex[:16]
        try:
            primary = registry.require_primary()
        except NoPrimaryError:
            metrics.DROPPED_REQUESTS.inc()
            metrics.REQUESTS.labels(version="none", outcome="error").inc()
            raise HTTPException(503, "no primary model loaded") from None

        features = torch.tensor([encode(**body.model_dump())], dtype=torch.float32)
        eta = primary.predict(features)

        metrics.REQUESTS.labels(version=primary.version, outcome="ok").inc()
        metrics.PREDICTIONS_ETA.labels(version=primary.version).observe(eta)

        shadow = registry.shadow
        if shadow is not None and shadow.version != primary.version:
            _run_shadow(shadow, primary, features, eta, request_id)

        metrics.REQUEST_LATENCY.labels(version=primary.version).observe(
            time.perf_counter() - started
        )
        return PredictResponse(
            eta_minutes=round(eta, 2), model_version=primary.version, request_id=request_id
        )

    def _run_shadow(shadow, primary, features, primary_eta: float, request_id: str) -> None:
        # The shadow path must never affect the client response.
        try:
            shadow_eta = shadow.predict(features)
        except Exception:  # noqa: BLE001
            metrics.SHADOW_REQUESTS.labels(shadow=shadow.version, outcome="error").inc()
            shadow_tracker.record_error()
            log.exception("shadow inference failed for %s", shadow.version)
            return
        metrics.SHADOW_REQUESTS.labels(shadow=shadow.version, outcome="ok").inc()
        metrics.SHADOW_DIVERGENCE.labels(primary=primary.version, shadow=shadow.version).observe(
            abs(shadow_eta - primary_eta)
        )
        shadow_tracker.record(
            ShadowRecord(
                request_id=request_id,
                primary_version=primary.version,
                shadow_version=shadow.version,
                primary_eta=primary_eta,
                shadow_eta=shadow_eta,
                at=time.time(),
            )
        )

    # ---- health and metrics ---------------------------------------------

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "version": __version__}

    @app.get("/readyz")
    async def readyz():
        if not registry.ready:
            return JSONResponse(status_code=503, content={"ready": False, "primary": None})
        return {"ready": True, "primary": registry.primary.version}

    @app.get("/metrics")
    async def prometheus_metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # ---- admin -----------------------------------------------------------

    @app.get("/admin/versions", dependencies=[Depends(require_admin)])
    async def admin_versions():
        return registry.describe()

    @app.post("/admin/shadow", dependencies=[Depends(require_admin)])
    async def admin_shadow(body: ShadowRequest):
        if body.version is not None and not registry.is_known(body.version):
            raise HTTPException(404, f"unknown version {body.version!r}")
        if (
            body.version is not None
            and registry.primary
            and body.version == registry.primary.version
        ):
            raise HTTPException(409, "shadow version must differ from the primary")
        loaded = await asyncio.to_thread(registry.set_shadow, body.version)
        shadow_tracker.reset()
        return {"shadow": loaded.version if loaded else None}

    @app.get("/admin/shadow/report", dependencies=[Depends(require_admin)])
    async def admin_shadow_report():
        report = shadow_tracker.report()
        report["primary"] = registry.primary.version if registry.primary else None
        report["shadow"] = registry.shadow.version if registry.shadow else None
        return report

    @app.post("/admin/promote", dependencies=[Depends(require_admin)])
    async def admin_promote(body: PromoteRequest):
        if not registry.is_known(body.version):
            raise HTTPException(404, f"unknown version {body.version!r}")
        try:
            record = await asyncio.to_thread(registry.promote, body.version)
        except UnknownVersionError:
            raise HTTPException(404, f"unknown version {body.version!r}") from None
        return record

    @app.post("/admin/rollback", dependencies=[Depends(require_admin)])
    async def admin_rollback():
        try:
            return await asyncio.to_thread(registry.rollback)
        except NoPrimaryError as exc:
            raise HTTPException(409, str(exc)) from None

    return app


app = create_app()
