"""Owned subprocesses must remain recoverable when startup registration fails."""
import json
import socket
import subprocess
import sys

import pytest

from scripts import services


def plan(port,name='studio'):
    return {'name':name,'port':port,'cwd':None,'health':'/health',
            'command':[sys.executable,'-u','-c',
                       f'from http.server import HTTPServer,SimpleHTTPRequestHandler; '
                       f'HTTPServer(("127.0.0.1",{port}),SimpleHTTPRequestHandler).serve_forever()']}


def free_port():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0))
        return listener.getsockname()[1]


@pytest.mark.parametrize('failure',[OSError('disk full'),KeyboardInterrupt()])
def test_registration_failure_reaps_actual_new_child_and_preserves_original_error(tmp_path,monkeypatch,failure):
    owned=[];popen=subprocess.Popen
    monkeypatch.setattr(services,'service_plan',lambda *_:[plan(free_port())])
    def launch(*args,**kwargs):
        process=popen(*args,**kwargs);owned.append(process);return process
    def fail(*args):raise failure
    monkeypatch.setattr(services.subprocess,'Popen',launch)
    monkeypatch.setattr(services,'atomic_json',fail)
    try:
        with pytest.raises(type(failure)) as error:services.start(tmp_path)
        assert error.value is failure
        assert len(owned)==1 and owned[0].poll() is not None,'New child was left unmanaged'
    finally:
        for process in owned:
            if process.poll() is None:process.terminate()
            process.wait(timeout=3)


def test_reused_unhealthy_service_fails_without_launching_or_stopping_it(tmp_path,monkeypatch):
    item=plan(free_port())
    folder=tmp_path/'.runtime';folder.mkdir()
    record={'pid':123,'start_ticks':'owned','port':item['port'],'command':item['command']}
    (folder/'services.json').write_text(json.dumps({'studio':record}))
    monkeypatch.setattr(services,'service_plan',lambda *_:[item])
    monkeypatch.setattr(services,'alive',lambda _:True)
    monkeypatch.setattr(services.subprocess,'Popen',lambda *a,**k:pytest.fail('restarted existing service'))
    monkeypatch.setattr(services.os,'killpg',lambda *a:pytest.fail('stopped existing service'))
    with pytest.raises(RuntimeError,match='studio'):
        services.start(tmp_path)
    assert json.loads((folder/'services.json').read_text())=={'studio':record}


def test_local_health_ignores_environment_proxy_and_accepts_cold_model(tmp_path,monkeypatch):
    import threading
    from http.server import BaseHTTPRequestHandler,HTTPServer
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200);self.end_headers()
            self.wfile.write(b'{"status":"ok","loaded":false}')
        def log_message(self,*args):pass
    server=HTTPServer(('127.0.0.1',0),Handler)
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    item=plan(server.server_port)
    folder=tmp_path/'.runtime';folder.mkdir()
    record={'pid':123,'start_ticks':'owned','port':item['port'],'command':item['command']}
    (folder/'services.json').write_text(json.dumps({'studio':record}))
    monkeypatch.setattr(services,'service_plan',lambda *_:[item])
    monkeypatch.setattr(services,'alive',lambda _:True)
    monkeypatch.setattr(services.subprocess,'Popen',lambda *a,**k:pytest.fail('launched a duplicate'))
    monkeypatch.setenv('http_proxy','http://127.0.0.1:1')
    monkeypatch.delenv('no_proxy',raising=False);monkeypatch.delenv('NO_PROXY',raising=False)
    hits=[];original=Handler.do_GET
    def get(self):hits.append(self.path);original(self)
    monkeypatch.setattr(Handler,'do_GET',get)
    try:
        services.start(tmp_path)
        assert hits==['/health'],'Reused service skipped readiness check'
    finally:
        server.shutdown();server.server_close();worker.join(timeout=2)


@pytest.mark.parametrize('retain_existing',[False,True])
def test_partial_registration_failure_rolls_back_only_new_children_then_allows_retry(tmp_path,monkeypatch,retain_existing):
    import time
    items=[plan(free_port(),'voice'),plan(free_port(),'studio')]
    for item in items:item['health']='/'
    popen,write=subprocess.Popen,services.atomic_json
    created=[];retained=None
    folder=tmp_path/'.runtime';folder.mkdir()
    registry=folder/'services.json'
    existing={}
    try:
        if retain_existing:
            retained=popen(items[0]['command'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
            existing['voice']={'pid':retained.pid,'start_ticks':services.identity(retained.pid),
                               'port':items[0]['port'],'command':items[0]['command']}
            deadline=time.monotonic()+5
            while True:
                try:services._check_health(items[0]);break
                except OSError:
                    if retained.poll() is not None or time.monotonic()>=deadline:
                        pytest.fail('fixture server failed to start')
                    time.sleep(.02)
        write(registry,existing)
        writes=0
        fail_at=1 if retain_existing else 2
        def save(path,records):
            nonlocal writes
            writes+=1
            if writes==fail_at:raise OSError('registration failed once')
            return write(path,records)
        def launch(*args,**kwargs):
            process=popen(*args,**kwargs);created.append(process);return process
        monkeypatch.setattr(services,'service_plan',lambda *_:items)
        monkeypatch.setattr(services,'atomic_json',save)
        monkeypatch.setattr(services.subprocess,'Popen',launch)
        with pytest.raises(OSError,match='failed once'):services.start(tmp_path)
        assert all(process.poll() is not None for process in created)
        assert json.loads(registry.read_text())==existing
        if retained is not None:assert retained.poll() is None
        services.start(tmp_path)
        assert set(json.loads(registry.read_text()))=={'voice','studio'}
        services.stop(tmp_path,['studio'] if retain_existing else None)
        assert json.loads(registry.read_text())==existing
    finally:
        for process in [*created,*([retained] if retained is not None else [])]:
            if process.poll() is None:process.terminate()
            process.wait(timeout=3)
