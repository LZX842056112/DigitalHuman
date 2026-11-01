"""Explicit downloads from the pinned photo-runtime catalog."""
import argparse
from pathlib import Path
from scripts import photo_assets


def download(root: Path, selected=None):
    catalog=photo_assets.load_catalog(root)
    if selected is not None:
        unknown=set(selected)-catalog['models'].keys()
        if unknown:raise ValueError('未知照片模型：'+', '.join(sorted(unknown)))
    photo_assets.fetch_models(root,catalog,selected=selected)
    return {key:value for key,value in catalog['models'].items() if selected is None or key in selected}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--only',nargs='+')
    options=parser.parse_args()
    download(options.root.resolve(),options.only)
