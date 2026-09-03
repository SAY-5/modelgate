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
from modelgate.serving.batching import Batcher
from modelgate.serving.canary import CanaryRouter, CanaryThresholds
from modelgate.serving.config import Settings
from modelgate.serving.drift import DriftMonitor
from modelgate.serving.registry import ModelRegistry, NoPrimaryError, UnknownVersionError
from modelgate.serving.reqlog import RequestLog
from modelgate.serving.schemas import (
    CanaryRequest,
    PredictRequest,
    PredictResponse,
    PromoteRequest,
    ShadowRequest,
    WarmRequest,
    rejection_reason,
)
from modelgate.serving.shadow import ShadowRecord, ShadowTracker

log = logging.getLogger("modelgate")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    # One-row inference is faster without the intra-op thread pool spinning up.
    torch.set_num_threads(1)
    canary = CanaryRouter(
        CanaryThresholds(
            window_seconds=settings.canary_window_seconds,
            min_samples=settings.canary_min_samples,
            error_rate_delta=settings.canary_error_rate_delta,
            latency_ratio=settings.canary_latency_ratio,
            latency_floor_ms=settings.canary_latency_floor_ms,
        )
    )
    registry = ModelRegistry(
        settings.artifacts_dir,
        pool_size=settings.model_pool_size,
        pad_rows=settings.batch_max_size,
        extra_pins=lambda: {canary.candidate.version} if canary.candidate else set(),
    )
    shadow_tracker = ShadowTracker(settings.shadow_threshold_minutes, settings.shadow_log_size)
    batcher = Batcher(settings.batch_max_size, settings.batch_max_wait_ms / 1000.0)
    request_log = RequestLog(settings.request_log_path, settings.request_log_sample_rate)

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
        for version in settings.warm_versions:
            if registry.is_known(version):
                await asyncio.to_thread(registry.warm, version)
                log.info("warm pool loaded: %s", version)
        yield
        request_log.close()

    drift = DriftMonitor(
        registry.manifest.get("training_stats"),
        window_size=settings.drift_window_size,
        min_samples=settings.drift_min_samples,
        warn_threshold=settings.drift_warn_threshold,
        alert_threshold=settings.drift_alert_threshold,
        refresh_every=settings.drift_refresh_every,
    )

    app = FastAPI(
        title="ModelGate",
        version=__version__,
        description="ETA model serving with shadow runs and zero-drop version swaps.",
        lifespan=lifespan,
    )
    app.state.registry = registry
    app.state.shadow_tracker = shadow_tracker
    app.state.canary = canary
    app.state.drift = drift
    app.state.batcher = batcher
    app.state.request_log = request_log
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
            loc = ".".join(p for p in err.get("loc", ()) if isinstance(p, str) and p != "body")
            details.append({"field": loc or "body", "reason": reason, "message": err.get("msg")})
        if request.url.path == "/predict":
            for reason in reasons:
                metrics.INPUT_REJECTIONS.labels(reason=reason).inc()
            for d in details:
                if d["reason"] == "unknown_zone":
                    drift.record_unknown(d["field"])
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

        payload = body.model_dump()
        features = torch.tensor([encode(**payload)], dtype=torch.float32)
        drift.observe(payload)
        served = primary
        candidate = canary.pick(primary)
        if candidate is not None:
            served, eta = await _serve_canary(candidate, primary, features)
        else:
            result = await batcher.submit(primary, features)
            eta = result.eta
            canary.record(primary.version, result.infer_s, True, primary.version)

        metrics.REQUESTS.labels(version=served.version, outcome="ok").inc()
        metrics.PREDICTIONS_ETA.labels(version=served.version).observe(eta)
        request_log.record(payload, served.version, round(eta, 2))

        shadow = registry.shadow
        if shadow is not None and shadow.version != served.version:
            await _run_shadow(shadow, served, features, eta, request_id)

        metrics.REQUEST_LATENCY.labels(version=served.version).observe(
            time.perf_counter() - started
        )
        return PredictResponse(
            eta_minutes=round(eta, 2), model_version=served.version, request_id=request_id
        )

    async def _serve_canary(candidate, primary, features):
        """Run the candidate; on failure answer from the primary and count it against
        the candidate. The client never sees a canary failure."""
        t0 = time.perf_counter()
        try:
            result = await batcher.submit(candidate, features)
        except Exception:  # noqa: BLE001
            elapsed = time.perf_counter() - t0
            metrics.CANARY_REQUESTS.labels(version=candidate.version, outcome="fallback").inc()
            log.exception("canary inference failed for %s", candidate.version)
            canary.record(candidate.version, elapsed, False, primary.version)
            fallback = await batcher.submit(primary, features)
            return primary, fallback.eta
        metrics.CANARY_REQUESTS.labels(version=candidate.version, outcome="ok").inc()
        canary.record(candidate.version, result.infer_s, True, primary.version)
        return candidate, result.eta

    async def _run_shadow(shadow, primary, features, primary_eta: float, request_id: str):
        # The shadow path must never affect the client response.
        try:
            shadow_eta = (await batcher.submit(shadow, features)).eta
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

    @app.get("/admin/request-log", dependencies=[Depends(require_admin)])
    async def admin_request_log():
        request_log.flush()
        return request_log.describe()

    @app.post("/admin/warm", dependencies=[Depends(require_admin)])
    async def admin_warm(body: WarmRequest):
        if not registry.is_known(body.version):
            raise HTTPException(404, f"unknown version {body.version!r}")
        return await asyncio.to_thread(registry.warm, body.version)

    @app.post("/admin/canary", dependencies=[Depends(require_admin)])
    async def admin_canary(body: CanaryRequest):
        if body.version is None:
            canary.clear()
            return {"candidate": None, "weight": 0.0, "status": canary.status}
        if not registry.is_known(body.version):
            raise HTTPException(404, f"unknown version {body.version!r}")
        if registry.primary and body.version == registry.primary.version:
            raise HTTPException(409, "canary version must differ from the primary")
        candidate = await asyncio.to_thread(registry.load, body.version)
        canary.start(candidate, body.weight)
        return {"candidate": candidate.version, "weight": body.weight, "status": canary.status}

    @app.get("/admin/canary/report", dependencies=[Depends(require_admin)])
    async def admin_canary_report():
        return canary.report(registry.primary.version if registry.primary else None)

    @app.get("/admin/drift", dependencies=[Depends(require_admin)])
    async def admin_drift():
        return drift.report()

    @app.post("/admin/drift/reset", dependencies=[Depends(require_admin)])
    async def admin_drift_reset():
        drift.reset()
        return {"reset": True}

    @app.post("/admin/promote", dependencies=[Depends(require_admin)])
    async def admin_promote(body: PromoteRequest):
        if not registry.is_known(body.version):
            raise HTTPException(404, f"unknown version {body.version!r}")
        try:
            record = await asyncio.to_thread(registry.promote, body.version)
        except UnknownVersionError:
            raise HTTPException(404, f"unknown version {body.version!r}") from None
        # A version cannot be both the primary and its own canary.
        if canary.candidate is not None and canary.candidate.version == body.version:
            canary.clear()
        return record

    @app.post("/admin/rollback", dependencies=[Depends(require_admin)])
    async def admin_rollback():
        try:
            return await asyncio.to_thread(registry.rollback)
        except NoPrimaryError as exc:
            raise HTTPException(409, str(exc)) from None

    return app


app = create_app()
