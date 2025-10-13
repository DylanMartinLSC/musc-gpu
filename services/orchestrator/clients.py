from __future__ import annotations

import logging
from typing import Any, Dict
from urllib.parse import urljoin

import httpx

from .config import OrchestratorSettings
from .streams import StreamRecord
from .workers import WorkerInfo

logger = logging.getLogger(__name__)


class WorkerClient:
    def __init__(self, http_client: httpx.AsyncClient, settings: OrchestratorSettings):
        self._http = http_client
        self._settings = settings

    def _callback_url(self, stream_id: str) -> str | None:
        base = self._settings.orchestrator_external_url
        if base is None:
            return None
        return urljoin(str(base), f"streams/{stream_id}/results")

    async def start_stream(self, worker: WorkerInfo, stream: StreamRecord) -> None:
        payload: Dict[str, Any] = {
            "stream_id": stream.stream_id,
            "camera_id": stream.camera_id,
            "config": stream.config.dict(),
            "batch_size": stream.batch_size,
            "callback": {
                "url": self._callback_url(stream.stream_id),
                "secret": self._settings.orchestrator_callback_secret or None,
            },
            "storage": {
                "bucket": self._settings.s3_bucket,
                "region": self._settings.aws_region,
                "prefix": stream.config.bucket_prefix,
            },
        }
        endpoint = urljoin(str(worker.endpoint), "streams/start")
        logger.info("Dispatching stream %s to worker %s", stream.stream_id, worker.worker_id)
        response = await self._http.post(endpoint, json=payload)
        response.raise_for_status()

    async def stop_stream(self, worker: WorkerInfo, stream_id: str) -> None:
        endpoint = urljoin(str(worker.endpoint), "streams/stop")
        payload = {"stream_id": stream_id}
        logger.info("Stopping stream %s on worker %s", stream_id, worker.worker_id)
        response = await self._http.post(endpoint, json=payload)
        response.raise_for_status()
