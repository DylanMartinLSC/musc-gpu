from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import AnyUrl, BaseModel, Field, HttpUrl, validator


class StreamState(str, Enum):
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class StreamConfig(BaseModel):
    rtsp_url: AnyUrl = Field(..., description="Network stream to pull frames from")
    backbone_name: str = Field(..., description="Backbone checkpoint to use for MuSc")
    image_size: int = Field(..., ge=32, le=1536, description="Input resolution for inference")
    fps: float = Field(..., gt=0, description="Target frames per second to sample")
    duration: float = Field(..., gt=0, description="Seconds included in a single batch")
    feature_layers: List[int] = Field(..., description="Backbone feature layers to read")
    r_list: List[int] = Field(..., description="Aggregation degrees for LNAMD")
    pretrained: Optional[str] = Field(None, description="Pretraining dataset identifier")
    dataset_name: str = Field("streaming", description="Logical dataset name")
    vis_type: str = Field("global_norm", description="Heatmap normalization policy")
    bucket_prefix: Optional[str] = Field(
        None, description="Optional sub-path for persisted heatmaps"
    )

    @validator("feature_layers", "r_list")
    def _ensure_not_empty(cls, value: List[int]) -> List[int]:
        if not value:
            raise ValueError("value must not be empty")
        return value

    @property
    def batch_size(self) -> int:
        return max(1, int(round(self.fps * self.duration)))


class StreamCreateRequest(BaseModel):
    camera_id: str = Field(..., description="Unique camera identifier")
    config: StreamConfig


class StreamResponse(BaseModel):
    stream_id: str
    camera_id: str
    state: StreamState
    worker_id: Optional[str] = None
    batch_size: int
    created_at: datetime


class WorkerRegistration(BaseModel):
    worker_id: str
    endpoint: HttpUrl
    gpu_uuid: Optional[str] = None
    max_streams: int = Field(1, ge=1)


class WorkerHeartbeat(BaseModel):
    worker_id: str
    available_stream_slots: int = Field(..., ge=0)


class StreamResult(BaseModel):
    stream_id: str
    worker_id: str
    batch_timestamp: datetime
    anomaly_scores: List[float]
    heatmap_uris: List[str]

    @validator("heatmap_uris")
    def _validate_heatmap_uris(cls, value: List[str], values: dict) -> List[str]:
        scores = values.get("anomaly_scores", [])
        if scores and len(scores) != len(value):
            raise ValueError("heatmap_uris must align with anomaly_scores")
        return value
