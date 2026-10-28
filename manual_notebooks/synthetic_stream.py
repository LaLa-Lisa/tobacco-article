"""Synthetic image streams for the manual inference benchmarks."""

import queue
import threading
import time

from ultralytics.data.loaders import LoadStreams, SourceTypes


class SyntheticStream(LoadStreams):
    def __init__(self, image, fps=30):
        # Do not call super().__init__():
        # that would try to open a real camera/URL.
        if image is None or fps <= 0:
            raise ValueError("Provide a valid image and positive FPS")

        self.image = image
        self.fps = [fps]
        self.bs = 1
        self.mode = "stream"
        self.count = -1
        self.running = True
        self.source_type = SourceTypes(stream=True)

        self.frames = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()

        self.thread = threading.Thread(
            target=self._produce,
            daemon=True
        )
        self.thread.start()

    def _produce(self):
        interval = 1 / self.fps[0]
        next_time = time.perf_counter()

        while not self.stop_event.is_set():
            if self.stop_event.wait(
                max(0, next_time - time.perf_counter())
            ):
                break

            frame = self.image.copy()

            # Keep only the newest frame
            if self.frames.full():
                try:
                    self.frames.get_nowait()
                except queue.Empty:
                    pass

            self.frames.put_nowait(frame)

            next_time += interval

            if next_time < time.perf_counter():
                next_time = time.perf_counter()

    def __iter__(self):
        return self

    def __next__(self):
        if self.stop_event.is_set():
            raise StopIteration

        frame = self.frames.get(timeout=5)
        self.count += 1

        return ["synthetic.jpg"], [frame], [""]

    def __len__(self):
        return self.bs

    def close(self):
        self.stop_event.set()
        self.running = False
        self.thread.join()
