from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from .schemas import StreamConfig, StreamResponse, StreamResult, StreamState


@dataclass
class StreamRecord:
    stream_id: str
    camera_id: str
    config: StreamConfig
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    state: StreamState = StreamState.STARTING
    worker_id: str | None = None
    last_result: StreamResult | None = None

    @property
    def batch_size(self) -> int:
        return self.config.batch_size


class StreamRegistry:
    def __init__(self) -> None:
        self._streams: Dict[str, StreamRecord] = {}
        self._lock = asyncio.Lock()

    async def create_stream(self, camera_id: str, config: StreamConfig) -> StreamRecord:
        async with self._lock:
            stream_id = str(uuid.uuid4())
            record = StreamRecord(stream_id=stream_id, camera_id=camera_id, config=config)
            self._streams[stream_id] = record
            return record

    async def assign_worker(self, stream_id: str, worker_id: str) -> Optional[StreamRecord]:
        async with self._lock:
            record = self._streams.get(stream_id)
            if record is None:
                return None
            record.worker_id = worker_id
            record.state = StreamState.RUNNING
            record.updated_at = datetime.utcnow()
            return record

    async def update_state(self, stream_id: str, state: StreamState) -> Optional[StreamRecord]:
        async with self._lock:
            record = self._streams.get(stream_id)
            if record is None:
                return None
            record.state = state
            record.updated_at = datetime.utcnow()
            return record

    async def attach_result(self, result: StreamResult) -> Optional[StreamRecord]:
        async with self._lock:
            record = self._streams.get(result.stream_id)
            if record is None:
                return None
            record.last_result = result
            record.updated_at = datetime.utcnow()
            return record

    async def list_streams(self) -> List[StreamResponse]:
        async with self._lock:
            responses: List[StreamResponse] = []
            for record in self._streams.values():
                responses.append(
                    StreamResponse(
                        stream_id=record.stream_id,
                        camera_id=record.camera_id,
                        state=record.state,
                        worker_id=record.worker_id,
                        batch_size=record.batch_size,
                        created_at=record.created_at,
                    )
                )
            return responses

    async def get(self, stream_id: str) -> Optional[StreamRecord]:
        async with self._lock:
            return self._streams.get(stream_id)

    async def by_camera(self, camera_id: str) -> List[StreamRecord]:
        async with self._lock:
            return [record for record in self._streams.values() if record.camera_id == camera_id]

    async def release_worker(self, stream_id: str) -> None:
        async with self._lock:
            record = self._streams.get(stream_id)
            if record:
                record.worker_id = None
                record.state = StreamState.STOPPED
                record.updated_at = datetime.utcnow()
