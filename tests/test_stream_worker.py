import asyncio
import threading


def test_disconnecting_response_retains_worker_until_native_drain_finishes():
    from backend.stream_worker import StreamWorker
    entered=threading.Event();release=threading.Event();drained=threading.Event()
    def produce(cancelled):
        yield {'type':'start'}
        entered.set()
        release.wait(3)
        assert cancelled()
        drained.set()
    worker=StreamWorker(produce)
    async def check():
        iterator=worker.events()
        assert (await anext(iterator))['type']=='start'
        assert await asyncio.to_thread(entered.wait,1)
        await iterator.aclose()
        assert worker in StreamWorker.active and not drained.is_set()
        release.set()
        await asyncio.to_thread(worker.thread.join,2)
        assert drained.is_set() and worker not in StreamWorker.active
    asyncio.run(check())


def test_worker_reports_failure_and_keeps_event_queue_bounded():
    from backend.stream_worker import StreamWorker
    def produce(cancelled):
        yield {'type':'start'}
        raise ValueError('模型错误')
    worker=StreamWorker(produce)
    async def collect():return [event async for event in worker.events()]
    events=asyncio.run(collect())
    assert events[0]['type']=='start'
    assert events[-1]['type']=='error' and '模型错误' in events[-1]['text']
    assert worker.queue.maxsize==2


def test_worker_acknowledges_only_after_producer_cleanup(tmp_path):
    from backend.stream_worker import StreamWorker,wait_worker_done
    from backend.storage import read_json
    release=threading.Event();entered=threading.Event()
    def produce(cancelled):
        entered.set();release.wait(3)
        yield {'type':'complete'}
    path=tmp_path/'voice-job.json'
    worker=StreamWorker(produce,job_path=path)
    async def check():
        assert await asyncio.to_thread(entered.wait,1)
        worker.stopped.set()
        waiting=asyncio.create_task(wait_worker_done(path))
        await asyncio.sleep(.03)
        assert not waiting.done() and read_json(path)['state']=='running'
        release.set();await asyncio.wait_for(waiting,2)
        assert read_json(path)['state']=='done'
    try:asyncio.run(check())
    finally:release.set();worker.thread.join(3)
