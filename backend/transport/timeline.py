from collections import deque
import threading
import numpy as np
from scipy.signal import resample_poly
from math import gcd


class Timeline:
    """A/V time advances only when PCM is consumed by the audio track."""
    def __init__(self, sample_rate=48000):
        self.sample_rate = sample_rate
        self._lock = threading.RLock()
        self._generation = 0
        self._queue = deque()
        self._chunk = None
        self._cursor = 0
        self._pts = 0
        self._expr = np.zeros(100, np.float32)
        self._jaw = np.zeros(3, np.float32)

    def begin_turn(self):
        self.interrupt()
        return self._generation

    def interrupt(self):
        with self._lock:
            self._generation += 1
            self._queue.clear()
            self._chunk = None
            self._cursor = 0
            self._expr.fill(0)
            self._jaw.fill(0)

    def current(self, generation):
        with self._lock:
            return generation == self._generation

    def append(self, generation, pcm, rate, expressions, jaws):
        pcm = np.asarray(pcm, np.float32).reshape(-1)
        if rate != self.sample_rate:
            divisor = gcd(int(rate), self.sample_rate)
            pcm = resample_poly(pcm, self.sample_rate // divisor, int(rate) // divisor).astype(np.float32)
        with self._lock:
            if generation != self._generation:
                return False
            if len(self._queue) >= 2:
                raise BufferError("待播队列已满")
            self._queue.append((pcm, np.asarray(expressions, np.float32), np.asarray(jaws, np.float32)))
            return True

    def read_audio(self, count):
        with self._lock:
            output = np.zeros(count, np.float32)
            written = 0
            while written < count:
                if self._chunk is None:
                    if not self._queue:
                        self._expr.fill(0)
                        self._jaw.fill(0)
                        break
                    self._chunk = self._queue.popleft()
                    self._cursor = 0
                pcm, expressions, jaws = self._chunk
                n = min(count - written, len(pcm) - self._cursor)
                if n <= 0:
                    self._chunk = None
                    continue
                output[written:written + n] = pcm[self._cursor:self._cursor + n]
                self._cursor += n
                written += n
                index = min(len(expressions) - 1, int(self._cursor / self.sample_rate * 30))
                if index >= 0:
                    self._expr = expressions[index].copy()
                    self._jaw = jaws[index].copy()
                if self._cursor == len(pcm):
                    self._chunk = None
            pts = self._pts
            self._pts += count
            return output, pts

    def expression(self):
        with self._lock:
            return self._expr.copy(), self._jaw.copy()

    @property
    def busy(self):
        with self._lock:
            return self._chunk is not None or bool(self._queue)
