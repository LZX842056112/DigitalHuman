import hashlib
import threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import pytest


@pytest.fixture
def endpoint():
    payload=bytes(range(256))*4096
    ranges=[]
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            start,end=map(int,self.headers['Range'].removeprefix('bytes=').split('-'))
            ranges.append((start,end))
            self.send_response(206)
            self.send_header('Content-Range',f'bytes {start}-{end}/{len(payload)}')
            self.send_header('Content-Length',str(end-start+1))
            self.end_headers()
            self.wfile.write(payload[start:end+1])
        def log_message(self,*args): pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    yield f'http://127.0.0.1:{server.server_port}/model',payload,ranges
    server.shutdown();server.server_close();thread.join()


def test_parallel_download_preserves_existing_prefix_and_verifies(endpoint,tmp_path):
    from scripts.download_parallel import fetch_parallel
    url,payload,ranges=endpoint
    target=tmp_path/'model.bin'
    target.with_name('model.bin.partial').write_bytes(payload[:123456])
    fetch_parallel(url,target,hashlib.sha256(payload).hexdigest(),len(payload),workers=4,chunk_size=65536)
    assert target.read_bytes()==payload
    assert min(start for start,_ in ranges)==123456
    assert not target.with_name('model.bin.partial').exists()


def test_parallel_download_never_publishes_bad_hash(endpoint,tmp_path):
    from scripts.download_parallel import fetch_parallel
    url,payload,_=endpoint
    target=tmp_path/'model.bin'
    with pytest.raises(ValueError,match='SHA'):
        fetch_parallel(url,target,'0'*64,len(payload),workers=4,chunk_size=65536)
    assert not target.exists()


def test_parallel_download_reuses_completed_verified_target(endpoint,tmp_path):
    from scripts.download_parallel import fetch_parallel
    url,payload,ranges=endpoint
    target=tmp_path/'model.bin';target.write_bytes(payload)
    fetch_parallel(url,target,hashlib.sha256(payload).hexdigest(),len(payload))
    assert not ranges
