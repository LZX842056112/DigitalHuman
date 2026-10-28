"""Fetch only runtime source files from pinned public Git trees, with Git blob hashes.

This avoids large demonstration videos and the private optional submodule.
"""
import argparse
import hashlib
import json
import time
import gzip
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from backend.storage import atomic_json
from scripts.fetch_sources import NAMES

SUFFIXES={'.py','.json','.yaml','.yml','.toml','.txt','.ini','.cfg','.cpp','.cc','.c','.h','.hpp','.cu','.cuh','.sh','.md','.csv','.npy','.npz','.pkl','.pickle','.pt','.pth','.so','.onnx','.tiktoken','.jinja'}


def get(url):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'DigitalHuman-local-setup', 'Accept-Encoding':'gzip'}),timeout=45) as response:
                data=response.read()
                return gzip.decompress(data) if response.headers.get('Content-Encoding')=='gzip' else data
        except Exception:
            if attempt==4: raise
            time.sleep(attempt+1)


def fetch(root, selected=None):
    catalog=json.loads((root/'configs/models.json').read_text(encoding='utf-8'))
    for key, folder in NAMES.items():
        if selected and key not in selected:
            continue
        item=catalog[key]
        destination=root/'third_party'/folder
        marker=destination/'.source-lock.json'
        if marker.exists() and json.loads(marker.read_text())['commit']==item['commit']:
            continue
        tree=json.loads(get(f"https://api.github.com/repos/{item['repo']}/git/trees/{item['commit']}?recursive=1"))
        if tree.get('truncated'):
            raise RuntimeError(f'Git tree is truncated: {key}')
        rows=[r for r in tree['tree'] if r['type']=='blob' and
              (Path(r['path']).suffix.lower() in SUFFIXES or Path(r['path']).name in {'LICENSE','Makefile','CMakeLists.txt'})]
        def one(row):
            target=(destination/row['path']).resolve()
            if destination.resolve() not in target.parents:
                raise ValueError('Source path traversal')
            def valid(data):
                return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==row['sha']
            if target.exists():
                if valid(target.read_bytes()): return
                raise RuntimeError(f'Existing source differs: {target}')
            for attempt in range(4):
                try:
                    data=get(f"https://raw.githubusercontent.com/{item['repo']}/{item['commit']}/{row['path']}")
                    if not valid(data): raise OSError('Source Git blob hash mismatch')
                    target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(data)
                    return
                except OSError:
                    if attempt==3: raise
                    time.sleep(attempt+1)
        print(f"Fetch {key}: {len(rows)} source files @ {item['commit']}",flush=True)
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(one, rows))
        atomic_json(marker,{'repo':item['repo'],'commit':item['commit'],'files':{r['path']:r['sha'] for r in rows}})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--only',nargs='*')
    a=p.parse_args()
    fetch(a.root.resolve(),a.only)
