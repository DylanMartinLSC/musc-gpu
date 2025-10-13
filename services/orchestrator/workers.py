from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional

from pydantic import AnyUrl

from .config import OrchestratorSettings


@dataclass
class WorkerInfo:
    worker_id: str
    endpoint: AnyUrl
    max_streams: int
    gpu_uuid: str | None = None
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    active_streams: set[str] = field(default_factory=set)
    available_stream_slots: int = 0

    @property
    def has_capacity(self) -> bool:
        return self.available_stream_slots > 0 and len(self.active_streams) < self.max_streams

    def record_assignment(self, stream_id: str) -> None:
        self.active_streams.add(stream_id)
        self.available_stream_slots = max(0, self.available_stream_slots - 1)

    def release(self, stream_id: str) -> None:
        self.active_streams.discard(stream_id)
        self.available_stream_slots = min(
            self.max_streams - len(self.active_streams), self.max_streams
        )


class WorkerRegistry:
    def __init__(self, settings: OrchestratorSettings):
        self._settings = settings
        self._workers: Dict[str, WorkerInfo] = {}
        self._lock = asyncio.Lock()

    async def register(
        self,
        worker_id: str,
        endpoint: AnyUrl,
        max_streams: int,
        gpu_uuid: str | None = None,
    ) -> WorkerInfo:
        async with self._lock:
            info = WorkerInfo(
                worker_id=worker_id,
                endpoint=endpoint,
                max_streams=max_streams,
                gpu_uuid=gpu_uuid,
                available_stream_slots=max_streams,
            )
            self._workers[worker_id] = info
            return info

    async def heartbeat(self, worker_id: str, available_slots: int) -> Optional[WorkerInfo]:
        async with self._lock:
            worker = self._workers.get(worker_id)
            if worker is None:
                return None
            worker.last_heartbeat = datetime.utcnow()
            worker.available_stream_slots = min(max(available_slots, 0), worker.max_streams)
            return worker

    async def mark_assignment(self, worker_id: str, stream_id: str) -> None:
        async with self._lock:
            worker = self._workers[worker_id]
            worker.record_assignment(stream_id)

    async def release_stream(self, worker_id: str, stream_id: str) -> None:
        async with self._lock:
            worker = self._workers.get(worker_id)
            if worker:
                worker.release(stream_id)

    async def select_worker(self) -> Optional[WorkerInfo]:
        async with self._lock:
            ttl = timedelta(seconds=self._settings.worker_heartbeat_ttl)
            now = datetime.utcnow()
            candidates = [
                worker
                for worker in self._workers.values()
                if worker.has_capacity and now - worker.last_heartbeat <= ttl
            ]
            if not candidates:
                return None
            candidates.sort(key=lambda w: (len(w.active_streams), -w.available_stream_slots))
            return candidates[0]

    async def cleanup_stale_workers(self) -> None:
        async with self._lock:
            ttl = timedelta(seconds=self._settings.worker_heartbeat_ttl)
            now = datetime.utcnow()
            stale = [wid for wid, info in self._workers.items() if now - info.last_heartbeat > ttl]
            for worker_id in stale:
                del self._workers[worker_id]

    async def get(self, worker_id: str) -> Optional[WorkerInfo]:
        async with self._lock:
            return self._workers.get(worker_id)

    async def all_workers(self) -> Dict[str, WorkerInfo]:
        async with self._lock:
            return dict(self._workers)
