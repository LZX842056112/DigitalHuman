"""Download auxiliary learned assets during setup, then fingerprint actual files."""
import os
from pathlib import Path
from backend.storage import atomic_json, sha256


def warmup(root):
    os.environ['TORCH_HOME']=str(root/'.runtime/torch')
    import torch
    import face_alignment
    import lpips
    torch.set_num_threads(4)
    detector=face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,device='cpu',flip_input=False)
    del detector
    metric=lpips.LPIPS(net='alex').cpu().eval()
    del metric
    cache=root/'.runtime/torch'
    files={str(p.relative_to(root)).replace('\\','/'):sha256(p) for p in cache.rglob('*')
           if p.is_file() and p.suffix in {'.pt','.pth','.zip'}}
    if not files: raise RuntimeError('辅助学习模型缓存为空')
    atomic_json(root/'models/runtime-assets-lock.json',{'files':files,'device':'cpu','validated_by':'actual constructor load'})


if __name__=='__main__':
    warmup(Path(__file__).resolve().parents[1])
