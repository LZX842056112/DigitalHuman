"""Local service supervisor. Stop only processes created by this project."""
from __future__ import annotations
import argparse
import json
import os
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from backend.storage import atomic_json


def identity(pid):
    try:
        stat=Path(f'/proc/{pid}/stat').read_text()
        fields=stat[stat.rindex(')')+2:].split()
        return fields[19] if fields[0] not in {'Z','X'} else None
    except (OSError,ValueError,IndexError):
        return None


def alive(record):
    return identity(record['pid'])==record.get('start_ticks') and record.get('start_ticks') is not None


def _worker_services(root):
    return [dict(name=name,port=port,cwd=root,command=[
        f'/opt/digitalhuman/{environment}/bin/python','-m','uvicorn',module,
        '--host','127.0.0.1','--port',str(port)],health=health)
        for name,environment,module,port,health in [
            ('voice','emotional-voice','backend.voice.service:app',9880,'/health'),
            ('portrait','portrait','backend.portrait.service:app',9881,'/health'),
            ('studio','portrait','backend.api.app:app',7860,'/api/health')]]


def _dialogue_command(model):
    return ['/opt/digitalhuman/llama.cpp/build/bin/llama-server','-m',str(model),
        '--host','127.0.0.1','--port','8081','-ngl','0','-t','6','-c','4096',
        '--alias','qwen2.5-3b-instruct']


def service_plan(root, studio_only=False):
    workers=_worker_services(root)
    if studio_only:
        return [item for item in workers if item['name']=='studio']
    ggufs=list((root/'models/llm').glob('*q4_k_m.gguf'))
    if len(ggufs)!=1: raise FileNotFoundError('需要一个 Qwen2.5-3B Q4_K_M 模型')
    command=_dialogue_command(ggufs[0])
    if not Path(command[0]).is_file(): raise FileNotFoundError('llama-server 尚未编译')
    services=[dict(name='dialogue',port=8081,cwd=root,command=command,health='/health'),*workers]
    for item in services:
        if not Path(item['command'][0]).is_file(): raise FileNotFoundError(item['command'][0])
    return services


def start(root, studio_only=False):
    folder=root/'.runtime'
    folder.mkdir(exist_ok=True)
    registry=folder/'services.json'
    records=json.loads(registry.read_text()) if registry.exists() else {}
    plan=service_plan(root,studio_only)
    expected={item['name']:item['command'] for item in plan}
    if studio_only:
        expected.update({item['name']:item['command'] for item in _worker_services(root)
                         if item['name']!='studio'})
        expected['dialogue']=_dialogue_command(root/'models/llm/qwen2.5-3b-instruct-q4_k_m.gguf')
    for name,record in records.items():
        if alive(record) and (name not in expected or record.get('command')!=expected[name]):
            raise RuntimeError(f'{name} 仍在运行旧配置。请先运行 scripts/stop.ps1，再运行 scripts/start.ps1。')
    for item in plan:
        if alive(records.get(item['name'],{'pid':-1})): continue
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1',item['port']))==0:
                raise RuntimeError(f"端口 {item['port']} 已被其他进程使用。未停止该进程。")
    env={**os.environ,'DIGITAL_HUMAN_ROOT':str(root),'PYTHONPATH':str(root),
         'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HOME':str(root/'.runtime/hf'),
         'TORCH_HOME':str(root/'.runtime/torch'),
         'TORCHDYNAMO_DISABLE':'0','TORCHINDUCTOR_CACHE_DIR':str(root/'.runtime/cache/inductor'),
         'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'4'}
    env['PATH']='/usr/lib/wsl/lib:'+env.get('PATH','')
    launched=[]
    try:
        for item in plan:
            name=item['name']
            if alive(records.get(name,{'pid':-1})): continue
            log=folder/'logs'/f'{name}.log'
            log.parent.mkdir(exist_ok=True)
            with log.open('ab') as output:
                proc=subprocess.Popen(item['command'],cwd=item['cwd'],env=env,stdin=subprocess.DEVNULL,
                        stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
            records[name]={'pid':proc.pid,'start_ticks':identity(proc.pid),'port':item['port'],'command':item['command']}
            atomic_json(registry,records)
            launched.append(name)
            deadline=time.monotonic()+180
            while time.monotonic()<deadline:
                if proc.poll() is not None: raise RuntimeError(f'{name} 启动失败。请查看 {log}')
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{item['port']}{item['health']}",timeout=2) as response:
                        if response.status==200: break
                except OSError:
                    time.sleep(.5)
            else: raise TimeoutError(f'{name} 未在180秒内就绪。请查看 {log}')
            print(f'{name}: ready on 127.0.0.1:{item["port"]}',flush=True)
    except Exception:
        stop(root,launched)
        raise
    print('Open http://localhost:7860',flush=True)


def stop(root, selected=None):
    registry=root/'.runtime/services.json'
    records=json.loads(registry.read_text()) if registry.exists() else {}
    for name,item in list(records.items()):
        if selected is not None and name not in selected: continue
        if alive(item):
            try:
                os.killpg(item['pid'],signal.SIGTERM)
            except ProcessLookupError:
                pass
            print(f'{name}: stop requested',flush=True)
            deadline=time.monotonic()+20
            while True:
                with socket.socket() as sock:
                    listening=sock.connect_ex(('127.0.0.1',item['port']))==0
                if not alive(item) and not listening: break
                if time.monotonic()>=deadline:
                    raise TimeoutError(f'{name} 未在20秒内退出或释放端口。已保留管理记录，请稍后重试。')
                time.sleep(.1)
        records.pop(name)
        atomic_json(registry,records)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['start','stop','status'])
    p.add_argument('--studio-only',action='store_true')
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    if a.action=='start': start(root,a.studio_only)
    elif a.action=='stop': stop(root)
    else:
        path=root/'.runtime/services.json'
        for name,item in (json.loads(path.read_text()) if path.exists() else {}).items():
            print(name,'running' if alive(item) else 'stopped',item['pid'])
