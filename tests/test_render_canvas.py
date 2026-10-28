from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import pytest
from einops import rearrange
from backend.avatar.lhmpp import LHMRenderer


class Decoder(torch.nn.Module):
    enc_patch_size=7
    def forward(self,images,features,motion,height,width,**kwargs):
        # Exercise the published decoder's real seven-pixel tokenization contract.
        tokens=rearrange(features,'b v c (h ph) (w pw) -> b v h w ph pw c',ph=7,pw=7)
        rgb=rearrange(tokens,'b v h w ph pw c -> b v (h ph) (w pw) c')
        return rgb,torch.ones_like(rgb[...,:1])


class Rasterizer:
    def get_single_view_smpl_data(self,params,index): return params
    def forward_animate_gs(self,models,query,params,c2w,intr,height,width,bg,**kwargs):
        yy,xx=torch.meshgrid(torch.arange(height),torch.arange(width),indexing='ij')
        # A ray-coded plane makes a wrong crop or camera shift observable.
        rgb=torch.stack(((xx-intr[0,0,0,2])/512+.5,(yy-intr[0,0,1,2])/512+.5,
                         torch.ones_like(xx)*kwargs['features'][0]))
        return {'comp_features':rgb[None,None]}


def renderer_and_args():
    model=SimpleNamespace(neural_renderer=Decoder(),renderer=Rasterizer(),
                          neural_renderer_patch_size=1,neural_renderer_input='feats')
    def animation(**args):
        intr=args['render_intrs']
        height,width=int(intr[0,0,1,2]*2),int(intr[0,0,0,2]*2)
        result=model.renderer.forward_animate_gs(None,None,None,None,intr,height,width,None,
                                               features=args['gs_hidden_features'])
        rgb,mask=model.neural_renderer(None,result['comp_features'],None,height,width)
        return rgb[0],mask[0]
    model.animation_infer=animation
    renderer=LHMRenderer(Path('.'),model,{})
    intr=torch.eye(4)[None,None]
    intr[...,0,0]=intr[...,1,1]=812
    intr[...,0,2]=intr[...,1,2]=256
    feature=torch.tensor([.5],requires_grad=True)
    args=dict(render_intrs=intr,render_c2ws=None,render_bg_colors=None,gs_model_list=None,
              query_points=None,smplx_params={},gs_hidden_features=feature,image_latents=None,
              motion_emb=None,pos_emb=None)
    renderer.arguments=lambda *a,**kw:args
    return renderer,args,feature


def test_standard_decoder_renders_512_without_changing_camera_rays():
    renderer,args,_=renderer_and_args()
    actual=renderer.render('idle',0,{},np.zeros(100),np.zeros(3))
    assert actual.shape==(512,512,3)
    assert tuple(actual[256,256])==(127,127,127)
    assert abs(int(actual[128,384,0])-191)<=1
    assert abs(int(actual[128,384,1])-63)<=1
    assert args['render_intrs'][0,0,0,2]==256


def test_calibration_uses_the_same_512_canvas_and_preserves_gradients():
    renderer,args,feature=renderer_and_args()
    rgb,mask=renderer.render_differentiable(args)
    assert rgb.shape==(1,1,512,512,3)
    assert mask.shape[:4]==(1,1,512,512)
    torch.testing.assert_close(rgb[0,0,256,256],torch.full((3,),.5))
    rgb[...,2].mean().backward()
    torch.testing.assert_close(feature.grad,torch.ones(1))


@pytest.mark.parametrize('size',[768,1024])
def test_detail_render_uses_native_pixels_and_matching_camera_rays(size):
    renderer,args,_=renderer_and_args()
    args['render_size']=size
    args['render_intrs'][...,0,2]=args['render_intrs'][...,1,2]=size/2
    rgb,_=renderer.render_differentiable(args)
    assert rgb.shape==(1,1,size,size,3)
    # The rasterizer generates one unique value per native pixel. Resizing a
    # 512 image would alter this spacing and crop the camera rays incorrectly.
    torch.testing.assert_close(rgb[0,0,size//2,size//2],torch.full((3,),.5))
    assert abs(float(rgb[0,0,size//2,size//2+1,0]-rgb[0,0,size//2,size//2,0])-1/512)<1e-6
    assert args['render_intrs'][0,0,0,2]==size/2
