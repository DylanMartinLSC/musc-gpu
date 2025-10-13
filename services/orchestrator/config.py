from __future__ import annotations

from functools import lru_cache

from pydantic import BaseSettings, HttpUrl


class OrchestratorSettings(BaseSettings):
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    result_retention_seconds: int = 3600
    worker_heartbeat_ttl: int = 120
    autoscaler_enabled: bool = True
    autoscaler_cooldown_seconds: int = 120
    orchestrator_callback_secret: str = ""
    orchestrator_external_url: HttpUrl | None = None

    aws_region: str | None = None
    s3_bucket: str | None = None

    class Config:
        env_prefix = "MUSC_"
        case_sensitive = False


@lru_cache()
def get_settings() -> OrchestratorSettings:
    return OrchestratorSettings()
