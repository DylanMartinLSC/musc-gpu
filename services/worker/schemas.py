from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import AnyUrl, BaseModel, Field, HttpUrl, validator


class CallbackConfig(BaseModel):
    url: Optional[HttpUrl] = None
    secret: Optional[str] = None


class StorageConfig(BaseModel):
    bucket: Optional[str] = None
    region: Optional[str] = None
    prefix: Optional[str] = None


class StreamConfig(BaseModel):
    rtsp_url: AnyUrl
    backbone_name: str
    image_size: int = Field(..., ge=32)
    fps: float = Field(..., gt=0)
    duration: float = Field(..., gt=0)
    feature_layers: List[int]
    r_list: List[int]
    pretrained: Optional[str] = None
    dataset_name: str = "streaming"
    vis_type: str = "global_norm"
    bucket_prefix: Optional[str] = None

    @validator("feature_layers", "r_list")
    def _ensure_non_empty(cls, value: List[int]) -> List[int]:
        if not value:
            raise ValueError("value must not be empty")
        return value

    @property
    def batch_size(self) -> int:
        return max(1, int(round(self.fps * self.duration)))


class StreamStartRequest(BaseModel):
    stream_id: str
    camera_id: str
    config: StreamConfig
    batch_size: Optional[int] = None
    callback: Optional[CallbackConfig] = None
    storage: Optional[StorageConfig] = None

    def resolved_batch_size(self) -> int:
        return self.batch_size or self.config.batch_size


class StreamStopRequest(BaseModel):
    stream_id: str


class StreamStatus(BaseModel):
    stream_id: str
    camera_id: str
    started_at: datetime
    batch_size: int
    fps: float
