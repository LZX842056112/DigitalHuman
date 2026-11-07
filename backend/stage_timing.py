"""Wall-time stages; overlapping workers are deliberately not summed."""
from contextlib import contextmanager
from backend import clock


class StageTiming:
    def __init__(self):self.values={};self.spans={}

    @contextmanager
    def measure(self,name):
        started=clock.monotonic()
        try:yield
        finally:
            ended=clock.monotonic()
            self.values[name]=self.values.get(name,0.)+(ended-started)*1000
            self.spans[name]=[self.spans.get(name,[started])[0],ended]
