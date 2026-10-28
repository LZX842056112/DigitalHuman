"""Track exact selected crops and preserve source frame identities."""
import argparse
import os
import sys
import hashlib
import json
from pathlib import Path
import numpy as np
from backend.storage import read_json, atomic_json, sha256


def tracking_input_manifest(root,folder,selected):
    """Bind tracking to the exact crops, source IDs and installed model version."""
    return {'version':1,'frames':[{'tracking_frame':item['tracking_frame'],'file':item['file'],
                    'sha256':sha256(folder/'processed/faces'/item['file'])} for item in selected],
            'model_lock':sha256(root/'models/lock.json') if (root/'models/lock.json').is_file() else None,
            'source_config':sha256(root/'configs/models.json') if (root/'configs/models.json').is_file() else None}


def tracking_current(root,folder,selected):
    marker=folder/'processed/tracking/input_manifest.json'
    return (folder/'processed/tracking/flame.json').is_file() and marker.is_file() and read_json(marker)==tracking_input_manifest(root,folder,selected)


def pack_observations(optimized, base, frames):
    serial={}
    for key,value in optimized.items():
        if key not in frames or key not in base:
            raise ValueError('跟踪结果没有对应的源视频帧')
        observed=np.asarray(base[key]['vgg_results']['lmks_2d70'],np.float64)
        if observed.shape!=(70,2) or not np.isfinite(observed).all():
            raise ValueError('真实面部关键点数量或数值无效')
        source=frames[key]
        serial[str(source['tracking_frame'])]={
            **{name:np.asarray(value[name]).tolist() for name in ('expcode','posecode','eyecode','shapecode')},
            'file':source['file'],'observed_lmks68':observed[:68].tolist(),
            'crop_size':[512,512],'landmark_source':'GAGAvatar HRNet image detector'}
    return serial


def track(root, folder):
    import torch
    import cv2
    repo=root/'third_party/gaga_track'
    sys.path.insert(0,str(repo))
    os.chdir(repo)
    from engines import CoreEngine
    selected=read_json(folder/'processed/faces/frames.json')['selected']
    manifest=tracking_input_manifest(root,folder,selected)
    signature=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()[:20]
    output=folder/'processed/tracking/gaga'/signature
    output.mkdir(parents=True,exist_ok=True)
    frames={f"face_{item['tracking_frame']:05d}":item for item in selected}
    images={}
    for key,item in frames.items():
        image=cv2.cvtColor(cv2.imread(str(folder/'processed/faces'/item['file'])),cv2.COLOR_BGR2RGB)
        image=cv2.resize(image,(512,512))
        images[key]=torch.from_numpy(image).permute(2,0,1).float()
    tracker=CoreEngine(focal_length=12.,device='cuda')
    base=tracker.track_base(images,str(output))
    optimized=tracker.track_optim(base,str(output),images,share_id=True)
    serial=pack_observations(optimized,base,frames)
    if len(serial)<4:
        raise RuntimeError('少于四帧通过真实面部跟踪')
    atomic_json(folder/'processed/tracking/flame.json',serial)
    atomic_json(folder/'processed/tracking/input_manifest.json',manifest)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--avatar",type=Path,required=True)
    args=parser.parse_args()
    track(args.root.resolve(),args.avatar.resolve())
