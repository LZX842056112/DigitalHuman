"""Download the optional outfit models into models/outfit.

Neither model ships with the normal install:

* ``clothes-seg`` (110 MB) is what makes colour changes possible;
* ``inpaint`` (2.6 GB) is what makes style changes possible.

Hugging Face is often unreachable from mainland China, so the download goes
through a mirror by default. Pass ``--mirror https://huggingface.co`` to use the
origin, or ``--only seg`` to fetch just the segmenter.

    python scripts/fetch_outfit_model.py
    python scripts/fetch_outfit_model.py --only inpaint
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    'seg': {
        'repo': 'mattmdjaga/segformer_b2_clothes',
        'folder': 'clothes-seg',
        'patterns': ['*.json', '*.txt', '*.safetensors'],
    },
    'inpaint': {
        'repo': 'runwayml/stable-diffusion-inpainting',
        'folder': 'inpaint',
        'patterns': ['*.json', '**/*.safetensors', 'tokenizer/*', 'scheduler/*',
                     'feature_extractor/*'],
    },
    'adapter': {
        # IP-Adapter reads the uploaded garment with a CLIP ViT-H encoder.
        # hf-mirror serves that 2.5 GB file at ~130 KB/s, while ModelScope
        # serves the same files at several MB/s, so this one has its own path.
        'folder': 'ip-adapter',
        'files': ['models/ip-adapter-plus_sd15.safetensors',
                  'models/image_encoder/config.json',
                  'models/image_encoder/model.safetensors'],
        'modelscope': 'AI-ModelScope/IP-Adapter',
    },
}


def _download(url, target):
    import shutil
    import urllib.request
    target.parent.mkdir(parents=True, exist_ok=True)
    start = target.stat().st_size if target.exists() else 0
    request = urllib.request.Request(url, headers={'Range': f'bytes={start}-'} if start else {})
    with urllib.request.urlopen(request, timeout=60) as response, target.open('ab') as handle:
        shutil.copyfileobj(response, handle, 1024 * 1024)


def fetch_adapter(root, force=False):
    spec = MODELS['adapter']
    base = f"https://modelscope.cn/models/{spec['modelscope']}/resolve/master/"
    for name in spec['files']:
        target = root / 'models/outfit' / spec['folder'] / name
        if target.is_file() and target.stat().st_size > 1024 and not force:
            print(f"  跳过已存在 {name}")
            continue
        print(f"  下载 {name}")
        _download(base + name, target)
    return root / 'models/outfit' / spec['folder']


def fetch(name, mirror=None, force=False):
    from huggingface_hub import snapshot_download
    spec = MODELS[name]
    target = ROOT / 'models/outfit' / spec['folder']
    if 'modelscope' in spec:
        return fetch_adapter(ROOT, force)
    if target.is_dir() and not force and any(target.iterdir()):
        print(f'{name}: 已存在 {target}（--force 可重新下载）')
        return target
    if mirror:
        os.environ['HF_ENDPOINT'] = mirror
    print(f"{name}: {spec['repo']} -> {target}")
    return Path(snapshot_download(spec['repo'], local_dir=str(target),
                                  allow_patterns=spec['patterns']))


def main(argv=None):
    parser = argparse.ArgumentParser(description='下载换装用的可选模型')
    parser.add_argument('--only', choices=sorted(MODELS), action='append',
                        help='只下载其中一个（可重复）')
    parser.add_argument('--mirror', default='https://hf-mirror.com',
                        help='HF 镜像地址；填 https://huggingface.co 走原站')
    parser.add_argument('--force', action='store_true', help='已存在也重新下载')
    options = parser.parse_args(argv)
    names = options.only or list(MODELS)
    for name in names:
        path = fetch(name, options.mirror, options.force)
        size = sum(item.stat().st_size for item in path.rglob('*') if item.is_file())
        print(f'{name}: {size / 1024 ** 2:.0f} MB')
    print('完成。重启 studio 后 /api/outfit/options 会显示可用的换装方式。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
