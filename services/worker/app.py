from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Dict
from urllib.parse import urljoin

import httpx
from fastapi import FastAPI, HTTPException, status

from .config import WorkerSettings, get_settings
from .schemas import StreamStartRequest, StreamStatus, StreamStopRequest
from .sessions import StreamManager
from .storage import S3StorageClient

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="MuSc Worker", version="1.0.0")

settings = get_settings()
async_http_client = httpx.AsyncClient(timeout=30.0)
sync_http_client = httpx.Client(timeout=30.0)
storage_client = S3StorageClient(settings.default_s3_bucket, settings.default_s3_region)
manager = StreamManager(storage_client, sync_http_client, settings)


async def _register_worker() -> None:
    registration_url = urljoin(str(settings.orchestrator_url), "workers/register")
    payload = {
        "worker_id": settings.worker_id,
        "endpoint": str(settings.endpoint),
        "max_streams": settings.max_streams,
        "gpu_uuid": settings.gpu_uuid,
    }
    response = await async_http_client.post(registration_url, json=payload)
    response.raise_for_status()


async def _heartbeat_loop() -> None:
    heartbeat_url = urljoin(str(settings.orchestrator_url), "workers/heartbeat")
    while True:
        try:
            slots = await manager.slots_available()
            payload = {
                "worker_id": settings.worker_id,
                "available_stream_slots": slots,
            }
            response = await async_http_client.post(heartbeat_url, json=payload)
            response.raise_for_status()
        except asyncio.CancelledError:
            raise
        except httpx.HTTPError as exc:  # pragma: no cover - network unreliability
            logger.warning("Heartbeat failed: %s", exc)
        await asyncio.sleep(settings.heartbeat_interval_seconds)


@app.on_event("startup")
async def startup_event() -> None:
    await _register_worker()
    app.state.heartbeat_task = asyncio.create_task(_heartbeat_loop())
    logger.info("Worker %s registered", settings.worker_id)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    heartbeat_task = getattr(app.state, "heartbeat_task", None)
    if heartbeat_task:
        heartbeat_task.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeat_task
    await async_http_client.aclose()
    sync_http_client.close()


@app.get("/healthz")
async def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/streams/start", response_model=StreamStatus, status_code=status.HTTP_202_ACCEPTED)
async def start_stream(request: StreamStartRequest) -> StreamStatus:
    try:
        return await manager.start_stream(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))


@app.post("/streams/stop")
async def stop_stream(request: StreamStopRequest) -> Dict[str, str]:
    await manager.stop_stream(request.stream_id)
    return {"status": "stopped"}


@app.get("/streams")
async def active_streams() -> Dict[str, StreamStatus]:
    return await manager.active_streams()
