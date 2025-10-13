from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import BaseSettings, HttpUrl


class WorkerSettings(BaseSettings):
    worker_id: str = "worker-1"
    endpoint: HttpUrl
    orchestrator_url: HttpUrl
    max_streams: int = 1
    heartbeat_interval_seconds: int = 30
    gpu_uuid: Optional[str] = None
    device: Optional[str] = None
    default_s3_bucket: Optional[str] = None
    default_s3_region: Optional[str] = None

    class Config:
        env_prefix = "MUSC_WORKER_"
        case_sensitive = False


@lru_cache()
def get_settings() -> WorkerSettings:
    return WorkerSettings()
