"""Install the pinned head extension into the existing avatar environment."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from importlib.metadata import version,PackageNotFoundError
from pathlib import Path
from scripts.download_parallel import fetch_parallel
from backend.storage import atomic_json


def git(folder,*args):
    # Windows checkouts may contain CRLF. Compare their canonical Git content
    # in WSL too; real source edits must still stop installation.
    return subprocess.check_output(['git','-c','core.autocrlf=true','-C',str(folder),*args],text=True).strip()


def source(folder,item):
    if folder.exists():
        if not (folder/'.git').exists():raise RuntimeError(f'Existing source has no Git identity: {folder}')
        if git(folder,'rev-parse','HEAD')!=item['commit']:raise RuntimeError(f'Existing source revision differs: {folder}')
        if git(folder,'status','--porcelain','--untracked-files=no'):raise RuntimeError(f'Existing source has edits: {folder}')
        return
    subprocess.run(['git','clone','--filter=blob:none','--no-checkout',item['repo'],str(folder)],check=True)
    subprocess.run(['git','-C',str(folder),'fetch','--depth','1','origin',item['commit']],check=True)
    subprocess.run(['git','-C',str(folder),'checkout','--detach',item['commit']],check=True)


def setup(root):
    import torch
    if not torch.__version__.startswith('2.3.0') or torch.version.cuda!='12.1':
        raise RuntimeError('Use the existing avatar Python 3.10 / Torch 2.3.0 / CUDA 12.1 environment')
    if shutil.disk_usage(root).free<10*1024**3:raise RuntimeError('Keep at least 10 GiB free before installing head assets')
    catalog=json.loads((root/'configs/head_models.json').read_text())
    source(root/'third_party/gag_avatar',catalog['code'])
    raster=root/'third_party/gag_raster32';source(raster,catalog['raster'])
    subprocess.run(['git','-C',str(raster),'submodule','update','--init','--depth','1','third_party/glm'],check=True)
    if git(raster/'third_party/glm','rev-parse','HEAD')!=catalog['raster']['glm_commit']:
        raise RuntimeError('GLM submodule revision differs')
    for name in ('weights','matting_config'):
        item=catalog[name]
        url=f"https://huggingface.co/{item['repo']}/resolve/{item['revision']}/{item['file']}"
        fetch_parallel(url,root/item['target'],item['sha256'],item['size'],workers=8)
    try:
        import diff_gaussian_rasterization_32d
    except ImportError:
        environment=dict(os.environ,MAX_JOBS='2',TORCH_CUDA_ARCH_LIST='8.6',CUDA_HOME='/usr/local/cuda-12.1')
        environment['PATH']=environment['CUDA_HOME']+'/bin:'+environment['PATH']
        subprocess.run([sys.executable,'-m','pip','install','--no-deps','--no-build-isolation',str(raster)],env=environment,check=True)
    try:tomli_installed=version('tomli')=='2.2.1'
    except PackageNotFoundError:tomli_installed=False
    if not tomli_installed:subprocess.run([sys.executable,'-m','pip','install','tomli==2.2.1'],check=True)
    atomic_json(root/'models/gag-head/install.json',{'code_commit':catalog['code']['commit'],
        'raster_commit':catalog['raster']['commit'],'torch':torch.__version__,'cuda':torch.version.cuda,
        'model_sha256':catalog['weights']['sha256']})
    print('Head assets installed. Run scripts/build_head.ps1 after the body build.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    setup(parser.parse_args().root.resolve())
