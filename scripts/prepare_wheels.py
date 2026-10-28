"""Cache exact PyPI wheels using publisher sizes and SHA-256, then let pip resolve."""
import argparse
import gzip
import json
import re
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
try:
    from packaging.tags import sys_tags
    from packaging.utils import parse_wheel_filename
except ImportError:
    from pip._vendor.packaging.tags import sys_tags
    from pip._vendor.packaging.utils import parse_wheel_filename
from scripts.fetch_artifact import fetch_file


def prepare(root, requirements):
    supported={tag:index for index,tag in enumerate(sys_tags())}
    folder=root/'.runtime/wheels'
    def one(requirement):
        name,version=requirement.split('==',1)
        with urllib.request.urlopen(urllib.request.Request(f'https://pypi.org/pypi/{name}/{version}/json',headers={'Accept-Encoding':'gzip'}),timeout=45) as response:
            data=response.read()
            data=gzip.decompress(data) if response.headers.get('Content-Encoding')=='gzip' else data
            metadata=json.loads(data)
        candidates=[]
        for item in metadata['urls']:
            if item['packagetype']!='bdist_wheel': continue
            tags=parse_wheel_filename(item['filename'])[3]
            rank=min((supported[t] for t in tags if t in supported),default=10**9)
            if rank<10**9: candidates.append((rank,item))
        if not candidates: raise RuntimeError(f'No compatible wheel: {requirement}')
        item=min(candidates,key=lambda p:p[0])[1]
        fetch_file(item['url'],folder/item['filename'],item['digests']['sha256'],item['size'])
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(one,requirements))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('requirements',nargs='*')
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    selected=a.requirements or re.findall(r'"([A-Za-z0-9_-]+==[A-Za-z0-9_.+]+)"',(root/'pyproject.toml').read_text())
    prepare(root,selected)
