"""One GPU operation across CLI builders and server processes."""
import os
import time
from backend import clock
from contextlib import contextmanager
from backend.errors import NotReady


class GpuLease:
    def __init__(self, root):
        self.path=root/'.runtime/gpu-operation.lock'
        self.stream=None

    def acquire(self):
        if self.stream is not None: return self
        self.path.parent.mkdir(parents=True,exist_ok=True)
        stream=self.path.open('a+b')
        if stream.tell()==0:
            stream.write(b'0');stream.flush()
        stream.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc:
            stream.close()
            raise NotReady('已有构建或交互进程正在使用本项目显存，请先停止该任务') from exc
        self.stream=stream
        return self

    def release(self):
        stream,self.stream=self.stream,None
        if stream is None: return
        try:
            stream.seek(0)
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
        finally:
            stream.close()


@contextmanager
def gpu_operation(root, cancelled=None, timeout=180):
    lease=GpuLease(root);deadline=clock.monotonic()+timeout
    while True:
        if cancelled and cancelled():raise InterruptedError('当前回答已停止')
        try:lease.acquire();break
        except NotReady:
            if clock.monotonic()>=deadline:raise NotReady('等待本项目显卡任务超时')
            time.sleep(.1)
    try:yield
    finally:lease.release()
