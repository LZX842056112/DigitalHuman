"""One GPU operation across CLI builders and server processes."""
import os
import secrets
import threading
import time
from backend import clock
from contextlib import contextmanager
from backend.errors import NotReady


class GpuLease:
    def __init__(self, root):
        self.path=root/'.runtime/gpu-operation.lock'
        self.stream=None

    def acquire(self,*,shared=False):
        if self.stream is not None: return self
        self.path.parent.mkdir(parents=True,exist_ok=True)
        stream=self.path.open('a+b')
        if stream.tell()==0:
            stream.write(b'0');stream.flush()
        stream.seek(0)
        try:
            if os.name=='nt':
                if shared:raise OSError('共享GPU锁需要WSL')
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),(fcntl.LOCK_SH if shared else fcntl.LOCK_EX)|fcntl.LOCK_NB)
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
def gpu_operation(root, cancelled=None, timeout=180, *, lease=None):
    lease=lease or GpuLease(root);deadline=clock.monotonic()+timeout
    while True:
        if cancelled and cancelled():raise InterruptedError('当前回答已停止')
        try:lease.acquire();break
        except NotReady:
            if clock.monotonic()>=deadline:raise NotReady('等待本项目显卡任务超时')
            time.sleep(.1)
    try:yield
    finally:lease.release()


def process_signature(pid):
    try:
        from pathlib import Path
        value=Path(f'/proc/{pid}/stat').read_text()
        return value[value.rfind(')')+2:].split()[19]
    except (OSError,IndexError):return None


def enough_stream_memory(free_mib,measured_peak_mib):
    return (measured_peak_mib is not None and measured_peak_mib>0
            and free_mib>=measured_peak_mib+1024)


def stream_code_key(root):
    import hashlib
    from pathlib import Path
    paths=('configs/photo-models.json','backend/gpu.py','backend/stream_audio.py',
           'backend/voice/engine.py','backend/portrait/engine.py','backend/portrait/stream.py')
    digest=hashlib.sha256()
    for relative in paths:
        digest.update(relative.encode());digest.update((Path(root)/relative).read_bytes())
    return digest.hexdigest()


def stream_admission(root):
    import subprocess
    from pathlib import Path
    from backend.storage import read_json
    try:
        profile=read_json(Path(root)/'.runtime/stream-admission.json')
        if (profile.get('status')!='passed' or profile.get('code_key')!=stream_code_key(root)
                or profile.get('qualification_version')!=2 or profile.get('overlap_seconds',0)<=0):
            return False,'显存准入记录尚未通过当前版本检查'
        result=subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.free',
            '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,check=True)
        identity,free=result.stdout.strip().splitlines()[0].split(',')
        if identity.strip()!=profile['gpu_uuid']:return False,'显卡与准入记录不一致'
        allowed=enough_stream_memory(float(free),profile['peak_delta_mib'])
        return allowed,None if allowed else '显存不足以保留1 GiB余量'
    except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError):
        return False,'尚无有效的双模型显存准入记录'


class SessionGpuLease(GpuLease):
    def __init__(self,root,session_id,*,concurrent=False):
        super().__init__(root)
        from backend.storage import safe_id
        from backend.stream_jobs import stream_base
        self.session_id=safe_id(session_id);self.token=secrets.token_hex(32)
        self.record=stream_base(root)/'lease.json';self.concurrent=concurrent

    def acquire(self):
        from backend.storage import atomic_json
        super().acquire()
        try:
            signature=process_signature(os.getpid())
            if signature is None:raise NotReady('流式GPU会话需要WSL进程身份检查')
            atomic_json(self.record,dict(session_id=self.session_id,token=self.token,
                pid=os.getpid(),signature=signature,concurrent=self.concurrent))
            import fcntl
            fcntl.flock(self.stream.fileno(),fcntl.LOCK_SH)
        except BaseException:
            super().release();raise
        return self

    def release(self):
        try:self.revoke()
        finally:super().release()

    def revoke(self):
        from backend.storage import read_json
        if self.stream is not None and self.record.exists() and read_json(self.record).get('token')==self.token:
            self.record.unlink()

    def set_concurrent(self,value):
        from backend.storage import atomic_json
        owner=stream_owner(self.path.parent.parent,self.session_id,self.token)
        atomic_json(self.record,{**owner,'concurrent':bool(value)})
        self.concurrent=bool(value)


def stream_owner(root,session_id,token):
    from backend.storage import read_json
    from backend.stream_jobs import stream_base
    try:
        owner=read_json(stream_base(root)/'lease.json')
        return _validate_owner(owner,session_id,token)
    except (OSError,ValueError,KeyError,TypeError):raise NotReady('流式任务不属于当前GPU会话')


def _validate_owner(owner,session_id,token):
    try:
        valid=(owner['session_id']==session_id and isinstance(token,str)
            and secrets.compare_digest(owner['token'],token) and type(owner['pid']) is int
            and isinstance(owner['signature'],str) and bool(owner['signature'])
            and process_signature(owner['pid'])==owner['signature'])
    except (KeyError,TypeError):valid=False
    if not valid:raise NotReady('流式任务不属于当前GPU会话')
    return owner


class StreamCancellation:
    def __init__(self,root,session_id,generation,token,disconnected):
        from backend.model_jobs import session_folder
        from backend.stream_jobs import stream_base
        if session_id is not None and generation is None:raise ValueError('缺少回答编号')
        self.marker=session_folder(root,session_id)/'generation.json' if session_id is not None else None
        self.owner=stream_base(root)/'lease.json' if token is not None else None
        self.session_id=session_id;self.generation=generation;self.token=token
        self.disconnected=disconnected;self.stopped=False;self.checked=-1e99;self.lock=threading.Lock()

    def __call__(self):
        from backend.storage import read_json
        if self.disconnected():self.stopped=True
        if self.stopped:return True
        if self.marker is None:return False
        with self.lock:
            now=clock.monotonic()
            if now-self.checked<.05:return self.stopped
            self.checked=now
            try:
                marker=read_json(self.marker)
                if self.owner is not None:_validate_owner(read_json(self.owner),self.session_id,self.token)
                self.stopped=bool(marker['closed']) or marker['generation']!=self.generation
            except (OSError,ValueError,KeyError,TypeError,NotReady):self.stopped=True
            return self.stopped


def free_vram_mib():
    import torch
    return torch.cuda.mem_get_info()[0]/1024**2


def check_stream_headroom(root,session_id,token):
    from backend.errors import StreamMemoryUnavailable
    owner=stream_owner(root,session_id,token)
    if owner['concurrent'] and free_vram_mib()<1024:
        raise StreamMemoryUnavailable('显存余量不足1 GiB，需要串行生成')


@contextmanager
def session_operation(root,session_id,token,cancelled=None):
    from backend.stream_jobs import stream_base
    owner=stream_owner(root,session_id,token)
    shared=GpuLease(root).acquire(shared=True)
    try:
        stream_owner(root,session_id,token)
        if cancelled and cancelled():raise InterruptedError('当前回答已停止')
        if owner['concurrent']:
            check_stream_headroom(root,session_id,token)
            yield owner
        else:
            lease=GpuLease(root);lease.path=stream_base(root)/'compute.lock'
            with gpu_operation(root,cancelled,lease=lease):
                stream_owner(root,session_id,token)
                yield owner
    finally:shared.release()
