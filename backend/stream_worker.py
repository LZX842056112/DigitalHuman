"""Keep synchronous native generators owned after an HTTP consumer leaves."""
import asyncio
from contextlib import asynccontextmanager
import json
import queue
import threading
import os
from pathlib import Path

from fastapi.responses import StreamingResponse


class StreamWorker:
    active=set()
    active_lock=threading.Lock()

    def __init__(self,produce,identity=None,*,job_path=None):
        self.produce=produce;self.identity=identity or {}
        self.job_path=Path(job_path) if job_path is not None else None
        self.queue=queue.Queue(maxsize=2);self.stopped=threading.Event();self.done=threading.Event()
        self.thread=threading.Thread(target=self._run,name='photo stream worker',daemon=False)
        with self.active_lock:self.active.add(self)
        self.thread.start()

    def _put(self,event):
        while not self.stopped.is_set():
            try:self.queue.put({**self.identity,**event},timeout=.1);return
            except queue.Full:pass

    def _run(self):
        from backend.gpu import process_signature
        from backend.storage import atomic_json
        error_text=None
        tid=threading.get_native_id();pid=os.getpid()
        record=dict(pid=pid,signature=process_signature(pid),tid=tid,
                    thread_signature=process_signature(f'{pid}/task/{tid}'),state='running')
        try:
            if self.job_path is not None:atomic_json(self.job_path,record)
            for event in self.produce(self.stopped.is_set):self._put(event)
        except Exception as error:
            error_text=str(error)[:500]
            self._put(dict(type='error',text=error_text,code=getattr(error,'code',None)))
        finally:
            try:
                if self.job_path is not None:atomic_json(self.job_path,{**record,'state':'done','error':error_text})
            finally:
                self.done.set()
                with self.active_lock:self.active.discard(self)

    def _next(self):
        try:return self.queue.get(timeout=.1)
        except queue.Empty:return None

    async def events(self):
        try:
            while not self.done.is_set() or not self.queue.empty():
                event=await asyncio.to_thread(self._next)
                if event is not None:yield event
        finally:self.stopped.set()

    def response(self):
        async def encoded():
            async for event in self.events():yield json.dumps(event,ensure_ascii=False)+'\n'
        return StreamingResponse(encoded(),media_type='application/x-ndjson',headers={'Cache-Control':'no-store'})


@asynccontextmanager
async def worker_lifespan(app):
    try:yield
    finally:
        with StreamWorker.active_lock:workers=list(StreamWorker.active)
        for worker in workers:worker.stopped.set()
        await asyncio.gather(*(asyncio.to_thread(worker.thread.join) for worker in workers))


async def wait_worker_done(path):
    """A closed socket is not proof that a native producer has stopped."""
    from backend.gpu import process_signature
    from backend.storage import read_json
    while await asyncio.to_thread(path.exists):
        record=await asyncio.to_thread(read_json,path)
        if record['state']=='done':return
        if process_signature(record['pid'])!=record['signature']:return
        if process_signature(f'{record["pid"]}/task/{record["tid"]}')!=record['thread_signature']:return
        await asyncio.sleep(.02)
