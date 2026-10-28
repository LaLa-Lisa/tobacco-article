"""Latest-frame synthetic camera for the manual inference benchmarks."""

from dataclasses import dataclass
import math
import threading
import time

from ultralytics.data.loaders import LoadStreams, SourceTypes


@dataclass(frozen=True)
class FrameMetadata:
    frame_id: int
    generated_at: float


class SyntheticStream(LoadStreams):
    def __init__(self, image, fps=30):
        # Do not initialize LoadStreams: that would open a real camera/URL.
        if image is None or not math.isfinite(fps) or fps <= 0:
            raise ValueError("Provide a valid image and finite positive FPS")

        self.image = image
        self.fps = [fps]
        self.bs = 1
        self.mode = "stream"
        self.count = -1
        self.running = True
        self.source_type = SourceTypes(stream=True)
        self.last_frame = None

        self._condition = threading.Condition()
        self._pending = None
        self._error = None
        self._generated = 0
        self._consumed = 0
        self._dropped = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._produce, daemon=True)
        self.thread.start()

    def _produce(self):
        try:
            interval = 1 / self.fps[0]
            next_time = time.perf_counter()
            while not self.stop_event.wait(max(0, next_time - time.perf_counter())):
                generated_at = time.perf_counter()
                frame = self.image.copy()
                with self._condition:
                    if self.stop_event.is_set():
                        break
                    self._generated += 1
                    metadata = FrameMetadata(self._generated, generated_at)
                    if self._pending is not None:
                        self._dropped += 1
                    self._pending = (frame, metadata)
                    self._condition.notify_all()
                # Missed scheduling slots are not generated or dropped frames.
                next_time = max(next_time + interval, time.perf_counter())
        except Exception as exc:
            with self._condition:
                self._error = exc
                self.running = False
                self._condition.notify_all()

    def snapshot(self):
        """Return atomic lifetime counters and their monotonic observation time."""
        with self._condition:
            return {
                "timestamp": time.perf_counter(),
                "generated": self._generated,
                "consumed": self._consumed,
                "dropped": self._dropped,
            }

    def __iter__(self):
        return self

    def __next__(self):
        with self._condition:
            self._condition.wait_for(
                lambda: self._pending is not None
                or self._error is not None
                or self.stop_event.is_set()
            )
            if self._error is not None:
                raise RuntimeError("Synthetic frame producer failed") from self._error
            if self.stop_event.is_set():
                raise StopIteration
            frame, self.last_frame = self._pending
            self._pending = None
            self._consumed += 1
            self.count += 1
        return ["synthetic.jpg"], [frame], [""]

    def __len__(self):
        return self.bs

    def close(self):
        with self._condition:
            self.stop_event.set()
            self.running = False
            self._condition.notify_all()
        self.thread.join()
