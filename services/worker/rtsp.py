from __future__ import annotations

import logging
import threading
import time
from typing import Generator, List

import cv2

logger = logging.getLogger(__name__)


class FrameBatcher:
    def __init__(self, rtsp_url: str, fps: float, batch_size: int) -> None:
        self._rtsp_url = rtsp_url
        self._fps = fps
        self._batch_size = batch_size

    def capture_batches(self, stop_event: threading.Event) -> Generator[List, None, None]:
        capture = cv2.VideoCapture(self._rtsp_url)
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open RTSP stream: {self._rtsp_url}")
        target_period = self._batch_size / self._fps if self._fps > 0 else 0
        try:
            while not stop_event.is_set():
                frames: List = []
                batch_start = time.time()
                while len(frames) < self._batch_size and not stop_event.is_set():
                    ok, frame = capture.read()
                    if not ok:
                        logger.warning("Failed to read frame from %s", self._rtsp_url)
                        time.sleep(1.0)
                        continue
                    frames.append(frame)
                if not frames:
                    continue
                yield frames
                elapsed = time.time() - batch_start
                sleep_for = max(0.0, target_period - elapsed)
                if sleep_for > 0:
                    stop_event.wait(timeout=sleep_for)
        finally:
            capture.release()
