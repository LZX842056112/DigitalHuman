"""Instantiate models when a worker starts so the first answer is not cold."""
import os
import threading
import time


def enabled():
    return os.environ.get('DIGITALHUMAN_WARMUP','0')=='1'


def start(load,lock,name):
    if not enabled():return None
    def run():
        started=time.monotonic()
        try:
            with lock:load()
            print(f'{name}: 模型预热完成，用时 {time.monotonic()-started:.1f} 秒',flush=True)
        except BaseException as error:
            print(f'{name}: 模型预热跳过 {type(error).__name__}: {error}',flush=True)
    thread=threading.Thread(target=run,name=name,daemon=True)
    thread.start()
    return thread
