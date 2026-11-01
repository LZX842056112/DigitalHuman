"""Fetch only pinned photo-animation and emotional-voice inference assets."""
import argparse
import hashlib
import gzip
import os
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest
from pathlib import Path
from urllib.parse import quote
from backend.storage import read_json, atomic_json, sha256
from scripts.fetch_artifact import fetch_file


def get(url):
    for attempt in range(5):
        try:
            request=urllib.request.Request(url,headers={'User-Agent':'DigitalHuman-local-setup','Accept-Encoding':'gzip'})
            with urllib.request.urlopen(request,timeout=45) as response:
                data=response.read()
                return gzip.decompress(data) if response.headers.get('Content-Encoding')=='gzip' else data
        except (OSError,TimeoutError):
            if attempt==4:raise
            time.sleep(attempt+1)


def load_catalog(root):
    return read_json(Path(root)/'configs/photo-models.json')


def check_source_lock(folder, commit):
    marker=folder/'.source-lock.json'
    if not marker.exists():return False
    if read_json(marker).get('commit') != commit:
        raise ValueError(f'源码版本发生变化，未覆盖：{folder}')
    return True


def fetch_sources(root, catalog):
    for item in catalog['sources'].values():
        folder=root/item['destination']
        check_source_lock(folder,item['commit'])
        def fetch(row):
            target=(folder/row['name']).resolve()
            if folder.resolve() not in target.parents:raise ValueError('源码路径越界')
            def valid(data):return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==row['sha1']
            if target.exists():
                if not valid(target.read_bytes()):raise ValueError(f'本机源码已修改，未覆盖：{target}')
                return
            data=get(f"https://raw.githubusercontent.com/{item['repo']}/{item['commit']}/{row['name']}")
            if not valid(data):raise ValueError(f'源码校验失败：{row["name"]}')
            target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_name(f'.{target.name}.{uuid.uuid4().hex}.tmp')
            try:
                temporary.write_bytes(data)
                # Do not overwrite a local edit that appeared during the download.
                if target.exists():
                    if not valid(target.read_bytes()):raise ValueError(f'本机源码已修改，未覆盖：{target}')
                else:os.replace(temporary,target)
            finally:temporary.unlink(missing_ok=True)
        with ThreadPoolExecutor(6) as pool:list(pool.map(fetch,item['files']))
        atomic_json(folder/'.source-lock.json',{'repo':item['repo'],'commit':item['commit']})
        print(f"Verified source {item['repo']} @ {item['commit']}",flush=True)


def fetch_models(root, catalog, selected=None):
    selected=set(catalog['models'] if selected is None else selected)
    unknown=selected-catalog['models'].keys()
    if unknown:raise ValueError('未知照片模型：'+', '.join(sorted(unknown)))
    def fetch(work):
        item,row=work
        path=root/item['destination']/row['name']
        url=row.get('url') or f"{item['host']}/{item['repo']}/resolve/{item['revision']}/{quote(row['name'])}"
        fetch_file(url,path,row['sha256'],row['size'])
    batches=[[(item,row) for row in item['files']] for name,item in catalog['models'].items() if name in selected]
    work=[entry for pair in zip_longest(*batches) for entry in pair if entry is not None]
    with ThreadPoolExecutor(4) as pool:list(pool.map(fetch,work))
    # A partial download must not replace a complete lock with a subset.
    # Publish the complete catalog only when every pinned file is verified.
    for item in catalog['models'].values():
        for row in item['files']:
            path=root/item['destination']/row['name']
            if not path.is_file() or path.stat().st_size!=row['size'] or sha256(path)!=row['sha256']:
                return
    atomic_json(root/'models/photo-lock.json',catalog)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--only',choices=['sources','models','all'],default='all')
    args=parser.parse_args();root=args.root.resolve();catalog=load_catalog(root)
    if args.only in {'sources','all'}:fetch_sources(root,catalog)
    if args.only in {'models','all'}:fetch_models(root,catalog)
