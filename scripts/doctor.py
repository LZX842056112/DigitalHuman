"""Read-only checks for the installed local photo and emotional voice runtime."""
import argparse
import hashlib
import json
import platform
import shutil
import socket
import subprocess
from pathlib import Path
from backend.storage import sha256

ENVIRONMENTS={
    'portrait':['torch','torchvision','torchaudio','onnxruntime','mediapipe','cv2','librosa','aiortc','faster_whisper','fastapi'],
    'emotional-voice':['torch','torchaudio','onnxruntime','cosyvoice.cli.cosyvoice','matcha','wetext','fastapi'],
}


def probe_environment(root,name,catalog=None):
    catalog=catalog or {}
    paths=[str(root/item['destination']) for item in catalog.get('sources',{}).values()]
    # Each environment performs its own imports and CUDA allocation check.
    code=("import sys,json,importlib;sys.path[:0]="+repr([str(root),*paths])+";"
          +"[importlib.import_module(name) for name in "+repr(ENVIRONMENTS[name])+"];"
          +"import torch;assert torch.__version__=='2.3.0+cu121',torch.__version__;"
          +"assert torch.cuda.is_available(),'CUDA不可用';"
          +"x=torch.zeros(1,device='cuda');"
          +"print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda}))")
    try:
        result=subprocess.run([f'/opt/digitalhuman/{name}/bin/python','-c',code],
                              cwd=root,capture_output=True,text=True,timeout=90)
        return {'ok':result.returncode==0,'detail':(result.stdout if result.returncode==0 else result.stderr).strip()}
    except (OSError,subprocess.SubprocessError) as error:
        return {'ok':False,'detail':str(error)}


def port_checks(root):
    from scripts.services import alive,service_plan
    registry=root/'.runtime/services.json'
    try:
        records=json.loads(registry.read_text()) if registry.exists() else {}
        expected={row['name']:row for row in service_plan(root)}
    except (OSError,ValueError) as error:
        return [{'name':'ports','ok':False,'detail':str(error)}]
    checks=[]
    for name,row in expected.items():
        record=records.get(name,{'pid':-1})
        with socket.socket() as sock:
            sock.settimeout(.2);listening=sock.connect_ex(('127.0.0.1',row['port']))==0
        owned=alive(record) and record.get('command')==row['command']
        checks.append({'name':'port_'+str(row['port']),'ok':not listening or owned,
                       'detail':'owned' if owned else ('空闲' if not listening else '端口被占用。请检查 scripts/stop.ps1。')})
    for name,record in records.items():
        if alive(record) and (name not in expected or record.get('command')!=expected[name]['command']):
            checks.append({'name':'service_'+name,'ok':False,
                           'detail':'旧服务仍在运行。请先运行 scripts/stop.ps1，再启动。'})
    return checks


def inspect(root:Path,verify_hashes=True):
    root=Path(root);checks=[]
    def add(name,ok,detail):checks.append({'name':name,'ok':bool(ok),'detail':str(detail)})
    def safe_path(base,name):
        path=(base/name).resolve()
        if base.resolve() not in path.parents:raise ValueError('校验路径越界')
        return path
    add('disk',shutil.disk_usage(root).free>10*1024**3,
        f'{shutil.disk_usage(root).free/1024**3:.1f} GiB available')
    add('platform',platform.system()=='Linux',platform.platform())
    add('ffmpeg',shutil.which('ffmpeg') is not None,shutil.which('ffmpeg') or 'missing')
    try:
        catalog=json.loads((root/'configs/photo-models.json').read_text(encoding='utf-8'))
    except (OSError,ValueError) as error:
        add('catalog',False,error)
        return {'ready':False,'checks':checks}
    for name in ENVIRONMENTS:
        result=probe_environment(root,name,catalog);add('environment_'+name,result['ok'],result['detail'])
    for name,item in catalog['sources'].items():
        invalid=[]
        try:
            folder=safe_path(root,item['destination'])
            marker=json.loads((folder/'.source-lock.json').read_text())
            if marker.get('commit')!=item['commit']:invalid.append('commit')
            for row in item['files']:
                path=safe_path(folder,row['name'])
                if not path.is_file() or path.stat().st_size!=row['size']:
                    invalid.append(row['name']);continue
                if verify_hashes:
                    data=path.read_bytes()
                    if hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()!=row['sha1']:
                        invalid.append(row['name'])
        except (OSError,ValueError,KeyError) as error:invalid.append(str(error))
        add('source_'+name,not invalid,', '.join(invalid) or item['commit'])
    for name,item in catalog['models'].items():
        invalid=[]
        try:
            folder=safe_path(root,item['destination'])
            for row in item['files']:
                path=safe_path(folder,row['name'])
                if (not path.is_file() or path.stat().st_size!=row['size']
                    or verify_hashes and sha256(path)!=row['sha256']):invalid.append(row['name'])
        except (OSError,ValueError,KeyError) as error:invalid.append(str(error))
        add('hash_'+name,not invalid,', '.join(invalid) or item['revision'])
    lock=root/'models/photo-lock.json'
    try:add('model_lock',json.loads(lock.read_text())==catalog,'pinned photo catalog')
    except (OSError,ValueError) as error:add('model_lock',False,error)
    add('llama_server',Path('/opt/digitalhuman/llama.cpp/build/bin/llama-server').is_file(),'llama.cpp b11897 CPU')
    from scripts.check_frontend import check
    try:add('frontend',check(root),'production files match source digest')
    except (OSError,ValueError,KeyError) as error:add('frontend',False,error)
    checks.extend(port_checks(root))
    return {'ready':all(row['ok'] for row in checks),'checks':checks}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--json',action='store_true')
    options=parser.parse_args();report=inspect(options.root.resolve())
    print(json.dumps(report,ensure_ascii=False,indent=2))
    raise SystemExit(0 if report['ready'] else 2)
