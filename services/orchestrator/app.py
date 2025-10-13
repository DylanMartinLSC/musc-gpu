from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import JSONResponse

from .autoscaler import AutoScaler
from .clients import WorkerClient
from .config import OrchestratorSettings, get_settings
from .schemas import (
    StreamCreateRequest,
    StreamResponse,
    StreamResult,
    StreamState,
    WorkerHeartbeat,
    WorkerRegistration,
)
from .streams import StreamRegistry
from .workers import WorkerRegistry

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="MuSc Orchestrator", version="1.0.0")


async def get_orchestrator_settings() -> OrchestratorSettings:
    return get_settings()


http_client = httpx.AsyncClient(timeout=60.0)
settings = get_settings()
worker_registry = WorkerRegistry(settings)
stream_registry = StreamRegistry()
autoscaler = AutoScaler(settings)
worker_client = WorkerClient(http_client, settings)


@app.on_event("shutdown")
async def _shutdown_http_client() -> None:
    await http_client.aclose()


@app.get("/healthz")
async def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/workers/register", status_code=status.HTTP_201_CREATED)
async def register_worker(registration: WorkerRegistration) -> Dict[str, Any]:
    info = await worker_registry.register(
        worker_id=registration.worker_id,
        endpoint=registration.endpoint,
        max_streams=registration.max_streams,
        gpu_uuid=registration.gpu_uuid,
    )
    return {
        "worker_id": info.worker_id,
        "endpoint": str(info.endpoint),
        "max_streams": info.max_streams,
    }


@app.post("/workers/heartbeat")
async def heartbeat(payload: WorkerHeartbeat) -> Dict[str, Any]:
    info = await worker_registry.heartbeat(
        worker_id=payload.worker_id, available_slots=payload.available_stream_slots
    )
    if info is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="worker unknown")
    return {
        "worker_id": info.worker_id,
        "available_stream_slots": info.available_stream_slots,
        "active_streams": len(info.active_streams),
    }


@app.get("/workers")
async def list_workers() -> Dict[str, Any]:
    workers = await worker_registry.all_workers()
    return {
        worker_id: {
            "endpoint": str(info.endpoint),
            "max_streams": info.max_streams,
            "available_stream_slots": info.available_stream_slots,
            "active_streams": list(info.active_streams),
        }
        for worker_id, info in workers.items()
    }


@app.post("/streams", response_model=StreamResponse, status_code=status.HTTP_201_CREATED)
async def create_stream(payload: StreamCreateRequest) -> StreamResponse:
    active_for_camera = await stream_registry.by_camera(payload.camera_id)
    if any(record.state in {StreamState.STARTING, StreamState.RUNNING} for record in active_for_camera):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="stream already running")

    record = await stream_registry.create_stream(payload.camera_id, payload.config)

    worker = await worker_registry.select_worker()
    if worker is None:
        await autoscaler.ensure_capacity(worker_registry)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="no worker capacity available",
        )

    await worker_registry.mark_assignment(worker.worker_id, record.stream_id)
    await stream_registry.assign_worker(record.stream_id, worker.worker_id)
    try:
        await worker_client.start_stream(worker, record)
    except httpx.HTTPError as exc:  # pragma: no cover - network failure handling
        logger.exception("Failed to dispatch stream %s", record.stream_id)
        await stream_registry.update_state(record.stream_id, StreamState.FAILED)
        await worker_registry.release_stream(worker.worker_id, record.stream_id)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))

    refreshed = await stream_registry.get(record.stream_id)
    assert refreshed is not None
    return StreamResponse(
        stream_id=refreshed.stream_id,
        camera_id=refreshed.camera_id,
        worker_id=refreshed.worker_id,
        state=refreshed.state,
        batch_size=refreshed.batch_size,
        created_at=refreshed.created_at,
    )


@app.get("/streams", response_model=list[StreamResponse])
async def list_streams() -> list[StreamResponse]:
    return await stream_registry.list_streams()


@app.get("/streams/{stream_id}")
async def stream_detail(stream_id: str) -> Dict[str, Any]:
    record = await stream_registry.get(stream_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="stream not found")
    payload: Dict[str, Any] = {
        "stream_id": record.stream_id,
        "camera_id": record.camera_id,
        "state": record.state,
        "worker_id": record.worker_id,
        "batch_size": record.batch_size,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }
    if record.last_result:
        payload["last_result"] = record.last_result.dict()
    return payload


@app.post("/streams/{stream_id}/stop")
async def stop_stream(stream_id: str) -> Dict[str, Any]:
    record = await stream_registry.get(stream_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="stream not found")
    if record.worker_id is None:
        await stream_registry.update_state(stream_id, StreamState.STOPPED)
        return {"status": "stopped"}

    worker = await worker_registry.get(record.worker_id)
    if worker is None:
        await stream_registry.update_state(stream_id, StreamState.STOPPED)
        return {"status": "stopped"}

    await stream_registry.update_state(stream_id, StreamState.STOPPING)
    try:
        await worker_client.stop_stream(worker, stream_id)
    except httpx.HTTPError as exc:
        logger.exception("Failed to stop stream %s", stream_id)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))

    await worker_registry.release_stream(worker.worker_id, stream_id)
    await stream_registry.release_worker(stream_id)
    return {"status": "stopped"}


@app.post("/streams/{stream_id}/results", status_code=status.HTTP_202_ACCEPTED)
async def ingest_results(
    stream_id: str,
    payload: StreamResult,
    secret: Optional[str] = Header(default=None, alias="X-Orchestrator-Secret"),
    settings: OrchestratorSettings = Depends(get_orchestrator_settings),
) -> JSONResponse:
    if payload.stream_id != stream_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="mismatched stream id")
    if settings.orchestrator_callback_secret:
        if secret != settings.orchestrator_callback_secret:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid secret")

    record = await stream_registry.attach_result(payload)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="stream not found")
    return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content={"status": "accepted"})
