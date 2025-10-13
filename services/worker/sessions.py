from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime
from typing import Dict, Optional

import httpx

from .config import WorkerSettings
from .inference import StreamingMuSc
from .rtsp import FrameBatcher
from .schemas import StreamStartRequest, StreamStatus
from .storage import S3StorageClient, StorageError

logger = logging.getLogger(__name__)


class StreamSession:
    def __init__(
        self,
        request: StreamStartRequest,
        storage_client: S3StorageClient,
        http_client: httpx.Client,
        settings: WorkerSettings,
    ) -> None:
        self.request = request
        self.stream_id = request.stream_id
        self.camera_id = request.camera_id
        self.batch_size = request.resolved_batch_size()
        self.fps = request.config.fps
        self.started_at = datetime.utcnow()
        self._engine = StreamingMuSc(request.config, device_override=settings.device)
        self._storage_client = storage_client
        self._http = http_client
        self._settings = settings
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._storage = self._storage_client.resolve(request.storage)
        self._callback = request.callback

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Session already started")
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def to_status(self) -> StreamStatus:
        return StreamStatus(
            stream_id=self.stream_id,
            camera_id=self.camera_id,
            started_at=self.started_at,
            batch_size=self.batch_size,
            fps=self.fps,
        )

    def _run_loop(self) -> None:
        batcher = FrameBatcher(self.request.config.rtsp_url, self.fps, self.batch_size)
        try:
            for frames in batcher.capture_batches(self._stop_event):
                if not frames:
                    continue
                scores, heatmaps = self._engine.process_batch(frames)
                timestamp = datetime.utcnow()
                heatmap_uris = []
                for index, image_bytes in enumerate(heatmaps):
                    key = f"streams/{self.stream_id}/{timestamp.strftime('%Y%m%dT%H%M%S')}_frame{index}.png"
                    uri = self._storage_client.upload_bytes(
                        self._storage, object_name=key, payload=image_bytes, content_type="image/png"
                    )
                    heatmap_uris.append(uri)
                self._dispatch_results(scores, heatmap_uris, timestamp)
                if self._stop_event.is_set():
                    break
        except StorageError as exc:
            logger.error("Storage error for stream %s: %s", self.stream_id, exc)
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Unhandled exception in stream session %s", self.stream_id)
        finally:
            logger.info("Stream session %s terminated", self.stream_id)

    def _dispatch_results(self, scores, heatmap_uris, timestamp: datetime) -> None:
        if not self._callback or not self._callback.url:
            return
        payload = {
            "stream_id": self.stream_id,
            "worker_id": self._settings.worker_id,
            "batch_timestamp": timestamp.isoformat(),
            "anomaly_scores": scores,
            "heatmap_uris": heatmap_uris,
        }
        headers = {}
        if self._callback.secret:
            headers["X-Orchestrator-Secret"] = self._callback.secret
        try:
            response = self._http.post(str(self._callback.url), json=payload, headers=headers)
            response.raise_for_status()
        except httpx.HTTPError as exc:  # pragma: no cover - network failures
            logger.warning("Failed to submit inference results for %s: %s", self.stream_id, exc)


class StreamManager:
    def __init__(
        self,
        storage_client: S3StorageClient,
        http_client: httpx.Client,
        settings: WorkerSettings,
    ) -> None:
        self._storage_client = storage_client
        self._http = http_client
        self._settings = settings
        self._sessions: Dict[str, StreamSession] = {}
        self._lock = asyncio.Lock()

    async def start_stream(self, request: StreamStartRequest) -> StreamStatus:
        async with self._lock:
            if len(self._sessions) >= self._settings.max_streams:
                raise RuntimeError("Worker at capacity")
            if request.stream_id in self._sessions:
                raise RuntimeError("Stream already active")
            session = StreamSession(request, self._storage_client, self._http, self._settings)
            session.start()
            self._sessions[request.stream_id] = session
            return session.to_status()

    async def stop_stream(self, stream_id: str) -> None:
        async with self._lock:
            session = self._sessions.pop(stream_id, None)
        if session:
            session.stop()

    async def active_streams(self) -> Dict[str, StreamStatus]:
        async with self._lock:
            return {stream_id: session.to_status() for stream_id, session in self._sessions.items()}

    async def slots_available(self) -> int:
        async with self._lock:
            return max(0, self._settings.max_streams - len(self._sessions))
