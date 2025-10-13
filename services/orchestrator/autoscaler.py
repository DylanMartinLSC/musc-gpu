from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta

from .config import OrchestratorSettings
from .workers import WorkerRegistry

logger = logging.getLogger(__name__)


class AutoScaler:
    def __init__(self, settings: OrchestratorSettings) -> None:
        self._settings = settings
        self._lock = asyncio.Lock()
        self._last_scale = datetime.min

    async def ensure_capacity(self, workers: WorkerRegistry) -> None:
        if not self._settings.autoscaler_enabled:
            return
        async with self._lock:
            now = datetime.utcnow()
            cooldown = timedelta(seconds=self._settings.autoscaler_cooldown_seconds)
            if now - self._last_scale < cooldown:
                return
            # Placeholder for infrastructure specific scale out hook.
            logger.warning(
                "AutoScaler triggered but no scale-out backend is configured."
                " Provision additional GPU workers manually if throughput is insufficient."
            )
            self._last_scale = now
