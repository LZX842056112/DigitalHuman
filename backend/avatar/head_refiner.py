"""Drive a cached 3D Gaussian head and composite it on the original 3D body."""
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from backend.avatar.head_compositing import crop_intrinsics,fit_similarity,composite_head,front_blend_weight
from backend.storage import sha256


def project(points,view,intrinsic):
    camera=points@view[:3,:3].T+view[:3,3]
    pixels=camera@intrinsic.T
    return pixels[:,:2]/pixels[:,2:].clamp(min=1e-6)


def raster(xyz,quaternions,scales,opacity,colors,view,intrinsic,size):
    from gsplat import rasterization
    return rasterization(xyz.float(),quaternions.float(),scales.float(),opacity.reshape(-1).float(),colors.float(),
        view[None].float(),intrinsic[None].float(),size,size,tile_size=8,channel_chunk=32)


class HeadRefiner:
    def __init__(self,root:Path,tensors:dict,metadata:dict):
        from backend.avatar.head_model import create_decoder,configure
        configure(root)
        from gagavatar.libs.flame_model import FLAMEModel
        flame_path=root/'third_party/gaga_track/assets/flame/FLAME_with_eye.pt'
        if sha256(flame_path)!=metadata['flame_sha256']:raise ValueError('头部表情模型校验失败')
        self.decoder=self.flame=None;self.values={};self.metadata=metadata;self._closed=False
        try:
            self.decoder=create_decoder(root,{k[8:]:v for k,v in tensors.items() if k.startswith('decoder.')})
            self.decoder.cuda()
            self.flame=FLAMEModel(300,100,scale=5.,no_lmks=True,model_path=flame_path).eval()
            self.flame.cuda()
            self.values={key:value.cuda() for key,value in tensors.items() if not key.startswith('decoder.')}
            count=self.values['xyz'].shape[0]
            for key,width in [('xyz',3),('colors',32),('opacities',1),('head_opacity',1),('scales',3),('rotations',4)]:
                if self.values[key].shape!=(count,width):raise ValueError('头部高斯参数形状无效')
            self.last_status='loaded'
        except BaseException:
            self.close();raise

    @classmethod
    def load(cls,root,folder):
        from backend.avatar.head_assets import load_head_assets
        tensors,metadata=load_head_assets(folder)
        return cls(root,tensors,metadata)

    def close(self):
        for model in (self.decoder,self.flame):
            if model is not None:model.cpu()
        self.decoder=self.flame=None;self.values.clear();self._closed=True

    def apply(self,args,body,gaussian):
        if self._closed:raise RuntimeError('高清头部已卸载')
        with torch.autocast('cuda',enabled=False):return self._apply(args,body.float(),gaussian)

    def _apply(self,args,body,gaussian):
        import cv2
        from pytorch3d.transforms import matrix_to_quaternion,quaternion_multiply
        values=self.values;size=body.shape[0]
        if gaussian is None:raise RuntimeError('头部渲染缺少当前身体姿态')
        scale_delta,rotation_delta,translation_delta=fit_similarity(values['anchors'],gaussian.xyz[values['anchor_indices']].float())
        rotation=values['registration_rotation']@rotation_delta
        scale=values['registration_scale']*scale_delta
        translation=values['registration_translation']@rotation_delta*scale_delta+translation_delta
        camera=args['render_c2ws'][0,0].float()
        center=values['neutral'][0].mean(0)@rotation*scale+translation
        direction=F.normalize(camera[:3,3]-center,dim=0)
        forward=torch.tensor([0.,0.,1.],device=body.device)@rotation_delta
        weight=front_blend_weight(float(direction@forward))
        if weight<=0:
            self.last_status='estimated_side_or_back';return body
        params=args['smplx_params'];pose=torch.zeros(1,6,device=body.device)
        pose[:,3:]=params['jaw_pose'].reshape(1,3).float()
        eye=torch.cat([params['leye_pose'].reshape(1,3),params['reye_pose'].reshape(1,3)],dim=1).float()
        animated=self.flame(values['shape'],params['expr'].reshape(1,100).float(),pose,eye)
        xyz=values['xyz'].clone();xyz[:5023]=animated[0]
        xyz=xyz@rotation*scale+translation
        quaternion=matrix_to_quaternion(rotation.T)
        quaternions=quaternion_multiply(quaternion[None],values['rotations'])
        scales=values['scales']*scale
        view=torch.linalg.inv(camera);intrinsic=args['render_intrs'][0,0,:3,:3].float()
        pixels=project(xyz[values['head_opacity'][:,0]>.02],view,intrinsic)
        low,high=pixels.min(0).values,pixels.max(0).values
        span=math.ceil(float((high-low).max())*1.10);middle=(low+high)/2
        if span<1 or span>512:
            # Never upscale the 512px head decoder. API camera limits normally
            # prevent this case; direct callers retain a valid body render.
            self.last_status='outside_native_detail_range';return body
        left=math.floor(float(middle[0])-span/2);top=math.floor(float(middle[1])-span/2)
        x0,y0=max(0,left),max(0,top);x1,y1=min(size,left+span),min(size,top+span)
        if x1<=x0 or y1<=y0:self.last_status='outside_view';return body
        cropped=crop_intrinsics(intrinsic,left,top,span)
        features,full_alpha,_=raster(xyz,quaternions,scales,values['opacities'],values['colors'],view,cropped,512)
        refined=self.decoder(features.permute(0,3,1,2),randomize_noise=False)[0]
        _,coverage,_=raster(xyz,quaternions,scales,values['head_opacity'],torch.ones_like(values['head_opacity']),view,cropped,512)
        mesh_pixels=project(xyz[:5023],view,cropped).cpu().numpy().astype(np.int32)
        envelope=np.zeros((512,512),np.float32)
        cv2.fillConvexPoly(envelope,cv2.convexHull(mesh_pixels),1)
        envelope=cv2.GaussianBlur(cv2.dilate(envelope,np.ones((31,31),np.uint8)),(11,11),3)
        coverage=coverage*torch.from_numpy(envelope).to(body.device)[None,:,:,None]
        refined=refined*(coverage[0,...,0]/full_alpha[0,...,0].clamp(min=1e-5)).clamp(0,1)[None]
        refined=F.interpolate(refined[None],size=(span,span),mode='area')[0].permute(1,2,0)
        coverage=F.interpolate(coverage.permute(0,3,1,2),size=(span,span),mode='area')[0,0]
        head_rgb=torch.zeros_like(body);head_alpha=torch.zeros_like(body[...,0])
        head_rgb[y0:y1,x0:x1]=refined[y0-top:y1-top,x0-left:x1-left]
        head_alpha[y0:y1,x0:x1]=coverage[y0-top:y1-top,x0-left:x1-left]
        fields,body_alpha,_=raster(gaussian.xyz,gaussian.rotation,gaussian.scaling,gaussian.opacity,
                                  values['body_head_mask'][:,None],view,intrinsic,size)
        old_head=fields[0,...,0];body_alpha=body_alpha[0,...,0]
        occluder=(body_alpha-old_head).clamp(0,1)
        # The reference body has a grey/blue hoodie. Preserve its collar when
        # the video head contains a red coat and a green scarf below the jaw.
        cloth=((body[...,2]-body[...,0]-.005)/.025).clamp(0,1)*(body_alpha>.05)
        occluder=torch.maximum(occluder,cloth)
        old_head=F.max_pool2d((old_head>.002).float()[None,None],19,1,9)
        old_head=F.avg_pool2d(old_head,5,1,2)[0,0]
        skin_or_hair=((body[...,0]>body[...,2]+.015)|(body.max(-1).values<.35)).float()
        result=composite_head(body,head_rgb,head_alpha,old_head*skin_or_hair,occluder,torch.ones_like(old_head))
        self.last_status='refined';return body.lerp(result,weight)
