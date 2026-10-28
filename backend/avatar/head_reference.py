"""Local video reference tracking and a garment-excluding head matte."""
import os
import sys
from pathlib import Path
import numpy as np
import torch


def source_head_mask(tracked,flame):
    import cv2
    from backend.avatar.head_refiner import project
    shape=tracked['shapecode'][None].cuda();eye=torch.zeros(1,6,device='cuda')
    _,lmks=flame(shape,tracked['expcode'][None].cuda(),tracked['posecode'][None].cuda(),eye)
    transform=tracked['transform_matrix'].cuda()
    flip=torch.diag(torch.tensor([-1.,-1.,1.],device='cuda'))
    view=torch.eye(4,device='cuda');view[:3,:3]=flip@transform[:3,:3].T;view[:3,3]=flip@transform[:3,3]
    intr=torch.tensor([[3072.,0,256],[0,3072.,256],[0,0,1.]],device='cuda')
    jaw=project(lmks[0,:17],view,intr).cpu().numpy()
    polygon=np.concatenate([[[0,0],[511,0],[511,jaw[-1,1]]],jaw[::-1],[[0,jaw[0,1]]]])
    upper=np.zeros((512,512),np.float32);cv2.fillPoly(upper,[polygon.astype(np.int32)],1)
    rgb=tracked['image'].permute(1,2,0).numpy().astype(np.float32)
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);r,g,b=rgb.transpose(2,0,1)
    skin=(r-g>.005)&(g-b>-.005)&(r-b>.025)&(hsv[...,0]<50)&(hsv[...,1]>.08)&(hsv[...,2]>.07)
    garment=((hsv[...,0]>300)|((hsv[...,0]>45)&(hsv[...,0]<150)))&(hsv[...,1]>.35)&(upper<.5)
    hair=(hsv[...,2]<.22)&(~garment)
    mask=cv2.GaussianBlur((rgb.max(-1)>.008)*((upper>.5)|skin|hair).astype(np.float32),(7,7),1.5)
    return mask,view,intr


def track_reference(root:Path,image_path:Path):
    from PIL import Image
    repo=root/'third_party/gaga_track'
    if str(repo) not in sys.path:sys.path.insert(0,str(repo))
    old_cwd=Path.cwd();os.chdir(repo)
    from transformers import Mask2FormerForUniversalSegmentation,Mask2FormerConfig
    original=Mask2FormerForUniversalSegmentation.from_pretrained
    def local_pixel_decoder(name,*args,**kwargs):
        if name!='facebook/mask2former-swin-tiny-coco-instance':raise ValueError('分割模型类型不匹配')
        config=Mask2FormerConfig.from_json_file(root/'models/gag-head/matting-config/config.json')
        return Mask2FormerForUniversalSegmentation(config)
    engine=None
    try:
        from engines import CoreEngine
        Mask2FormerForUniversalSegmentation.from_pretrained=staticmethod(local_pixel_decoder)
        engine=CoreEngine(focal_length=12.,device='cuda')
        image=torch.from_numpy(np.array(Image.open(image_path).convert('RGB'))).permute(2,0,1).float()
        result=engine.track_image([image],['head_source'])
        if not result:raise ValueError('视频面部跟踪失败，请选择清晰的正面帧')
        tracked={key:torch.as_tensor(np.array(value)).float().cpu() for key,value in result['head_source'].items()}
        from backend.avatar.state import require_finite_tensor
        for value in tracked.values():require_finite_tensor(value)
        return tracked
    finally:
        Mask2FormerForUniversalSegmentation.from_pretrained=original
        os.chdir(old_cwd)
        del engine
        import gc
        gc.collect();torch.cuda.empty_cache()
