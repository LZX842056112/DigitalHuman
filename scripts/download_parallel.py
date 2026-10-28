"""Resumable HTTP ranges with a frozen prefix and atomic verified publication."""
import contextlib
import http.client
import os
import shutil
import threading
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from backend.storage import atomic_json,read_json,sha256


@contextlib.contextmanager
def download_lock(target):
    stream=target.with_name(target.name+'.download.lock').open('a+b')
    if stream.tell()==0: stream.write(b'0');stream.flush()
    stream.seek(0)
    try:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        stream.close()
        raise RuntimeError(f'Another process is downloading {target.name}')
    try: yield
    finally: stream.close()


def fetch_parallel(url,target,digest,size,workers=4,chunk_size=1024**2):
    target=Path(target).resolve();target.parent.mkdir(parents=True,exist_ok=True)
    with download_lock(target):
        if target.is_file() and target.stat().st_size==size and sha256(target)==digest: return
        prefix=target.with_name(target.name+'.partial')
        base=prefix.stat().st_size if prefix.exists() else 0
        if base>=size:
            if base==size and sha256(prefix)==digest:
                os.replace(prefix,target);return
            os.replace(prefix,prefix.with_name(prefix.name+'.failed-'+uuid.uuid4().hex))
            raise ValueError('SHA-256 or size mismatch in existing prefix')
        parts=target.with_name(target.name+'.parts')
        if parts.is_symlink(): raise ValueError('Download cache cannot be a symlink')
        parts.mkdir(exist_ok=True)
        meta=parts/'meta.json'
        if meta.exists():
            previous=read_json(meta)
            if any(previous.get(k)!=v for k,v in {'sha256':digest,'size':size,'prefix_bytes':base}.items()):
                raise ValueError('Download cache belongs to a different artifact or prefix')
            chunk_size=previous['chunk_size']
        else:
            atomic_json(meta,{'sha256':digest,'size':size,'prefix_bytes':base,'chunk_size':chunk_size})
        count=(size-base+chunk_size-1)//chunk_size
        part_paths=[parts/f'{index:08d}.bin' for index in range(count)]
        progress_lock=threading.Lock()
        completed=base+sum(p.stat().st_size for p in part_paths if p.is_file())
        printed=completed//(32*1024**2)
        print(f'{target.name}: resume {completed}/{size}; {workers} connections',flush=True)
        def transfer(index):
            nonlocal completed,printed
            start=base+index*chunk_size;end=min(start+chunk_size,size)-1
            length=end-start+1;part=part_paths[index]
            if part.exists():
                if part.stat().st_size!=length: raise ValueError('Cached chunk has an invalid size')
                return
            temporary=part.with_suffix('.pending')
            failures=0
            while True:
                offset=temporary.stat().st_size if temporary.exists() else 0
                if offset==length: break
                if offset>length: raise ValueError('Partial chunk exceeds its assigned range')
                try:
                    request=urllib.request.Request(url,headers={'User-Agent':'DigitalHuman/0.1',
                              'Range':f'bytes={start+offset}-{end}','Accept-Encoding':'identity'})
                    with urllib.request.urlopen(request,timeout=15) as response:
                        expected=f'bytes {start+offset}-{end}/{size}'
                        if response.status!=206 or response.headers.get('Content-Range')!=expected:
                            raise ValueError(f'Unexpected HTTP range for {target.name}')
                        with temporary.open('ab') as output:
                            remaining=length-offset
                            while remaining:
                                data=response.read(min(65536,remaining))
                                if not data: break
                                output.write(data);remaining-=len(data)
                    after=temporary.stat().st_size
                    if after<=offset: raise OSError('Empty range response')
                    failures=0
                except (OSError,TimeoutError,http.client.HTTPException) as exc:
                    after=temporary.stat().st_size if temporary.exists() else 0
                    failures=0 if after>offset else failures+1
                    if failures>=8: raise OSError(f'Range {index} repeatedly stalled: {exc}') from exc
                    time.sleep(min(1+failures,5))
            os.replace(temporary,part)
            with progress_lock:
                completed+=length
                if completed//(32*1024**2)>printed or completed==size:
                    printed=completed//(32*1024**2)
                    print(f'{target.name}: {completed}/{size}',flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(transfer,range(count)))
        assembling=target.with_name(target.name+'.assembling')
        with assembling.open('wb') as output:
            if base:
                if prefix.stat().st_size!=base: raise ValueError('Prefix changed during download')
                with prefix.open('rb') as stream: shutil.copyfileobj(stream,output,1024**2)
            for part in part_paths:
                with part.open('rb') as stream: shutil.copyfileobj(stream,output,1024**2)
        if assembling.stat().st_size!=size or sha256(assembling)!=digest:
            stamp=uuid.uuid4().hex
            os.replace(assembling,assembling.with_name(assembling.name+'.failed-'+stamp))
            os.replace(parts,parts.with_name(parts.name+'.failed-'+stamp))
            if prefix.exists(): os.replace(prefix,prefix.with_name(prefix.name+'.failed-'+stamp))
            raise ValueError(f'SHA-256 mismatch: {target.name}; failure evidence retained')
        os.replace(assembling,target)
        # Only remove files created by this transfer. No recursive delete.
        prefix.unlink(missing_ok=True)
        for part in part_paths: part.unlink(missing_ok=True)
        meta.unlink()
        if not any(parts.iterdir()): parts.rmdir()
        print(f'Verified {target.name}: {size} bytes',flush=True)
