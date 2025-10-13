from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import boto3

from .schemas import StorageConfig

logger = logging.getLogger(__name__)


class StorageError(RuntimeError):
    pass


@dataclass
class ResolvedStorage:
    bucket: str
    region: Optional[str]
    prefix: Optional[str]

    def build_key(self, object_name: str) -> str:
        if self.prefix:
            return f"{self.prefix.rstrip('/')}/{object_name}"
        return object_name

    @property
    def uri_prefix(self) -> str:
        return f"s3://{self.bucket}/"


class S3StorageClient:
    def __init__(self, default_bucket: Optional[str] = None, default_region: Optional[str] = None) -> None:
        self._default_bucket = default_bucket
        self._default_region = default_region

    def resolve(self, config: Optional[StorageConfig]) -> ResolvedStorage:
        bucket = (config.bucket if config else None) or self._default_bucket
        region = (config.region if config else None) or self._default_region
        prefix = config.prefix if config else None
        if not bucket:
            raise StorageError("No S3 bucket configured for results")
        return ResolvedStorage(bucket=bucket, region=region, prefix=prefix)

    def upload_bytes(self, storage: ResolvedStorage, object_name: str, payload: bytes, content_type: str) -> str:
        key = storage.build_key(object_name)
        logger.debug("Uploading %s to bucket %s", key, storage.bucket)
        client = boto3.client("s3", region_name=storage.region)
        client.put_object(Bucket=storage.bucket, Key=key, Body=payload, ContentType=content_type)
        return f"s3://{storage.bucket}/{key}"
