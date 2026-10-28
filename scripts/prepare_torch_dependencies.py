"""Cache Linux CUDA wheels from the pinned Torch release's published metadata."""
import gzip
import json
import urllib.request
from pathlib import Path
from scripts.prepare_wheels import prepare
from backend.storage import atomic_json
try:
    from packaging.requirements import Requirement
except ImportError:
    from pip._vendor.packaging.requirements import Requirement


def main(root):
    request=urllib.request.Request('https://pypi.org/pypi/torch/2.3.0/json',headers={'Accept-Encoding':'gzip'})
    with urllib.request.urlopen(request,timeout=30) as response:
        data=response.read()
        if response.headers.get('Content-Encoding')=='gzip': data=gzip.decompress(data)
    info=json.loads(data)['info']
    pinned=[]
    for text in info['requires_dist']:
        requirement=Requirement(text)
        if not (requirement.name.startswith('nvidia-') or requirement.name=='triton'): continue
        if requirement.marker and not requirement.marker.evaluate(): continue
        specs=list(requirement.specifier)
        if len(specs)!=1 or specs[0].operator!='==': raise ValueError(f'Unpinned CUDA dependency: {requirement.name}')
        pinned.append(f'{requirement.name}=={specs[0].version}')
    if not pinned: raise RuntimeError('Published Linux CUDA dependency list is empty')
    atomic_json(root/'.runtime/torch-cuda-dependencies.json',{'torch':'2.3.0','requirements':pinned,
                'source':'https://pypi.org/pypi/torch/2.3.0/json'})
    prepare(root,pinned)


if __name__=='__main__': main(Path(__file__).resolve().parents[1])
