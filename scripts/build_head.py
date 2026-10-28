"""Bake and reload the local video head without rebuilding the saved body."""
import argparse
import gc
import json
import time
from contextlib import ExitStack
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from backend.avatar.lease import GpuLease
from backend.avatar.head_assets import save_head_assets
from backend.avatar.head_compositing import fit_similarity
from backend.avatar.head_model import load_published_head
from backend.avatar.head_refiner import HeadRefiner,project
from backend.avatar.head_reference import source_head_mask,track_reference
from backend.avatar.lhmpp import LHMRenderer
from backend.avatar.head_region import head_mask
from backend.storage import atomic_json,sha256,safe_id


def build(root:Path,avatar_id:str,reference:Path|None=None):
    folder=root/'data/avatars'/safe_id(avatar_id)
    image_path=folder/'processed/tracking/imgs_png/00003.png'
    if not image_path.is_file():raise FileNotFoundError('缺少视频第4秒的原始清晰帧，请先执行素材解析和面部跟踪')
    output=root/'outputs'/('head-detail-'+time.strftime('%Y%m%d-%H%M%S'));output.mkdir(parents=True)
    descriptor=folder/'head.json';previous=descriptor.read_bytes() if descriptor.exists() else None
    body=None;model=None;runtime=None;flame=None
    report={'kind':'head_detail_build_and_reload','complete':False,'visual_accepted':False,
            'body_state_sha256':sha256(folder/'state.json'),'source_frame_sha256':sha256(image_path)}
    with ExitStack() as resources,torch.no_grad():
        resources.callback(GpuLease(root).acquire().release)
        try:
            if reference is not None:
                tracked=torch.load(reference,map_location='cpu',weights_only=True)
            else:
                from safetensors.torch import load_file,save_file
                cache=folder/'processed/head_reference.safetensors'
                cache_meta=folder/'processed/head_reference.json'
                if cache.exists() and cache_meta.exists() and json.loads(cache_meta.read_text())=={
                        'source_sha256':sha256(image_path),'cache_sha256':sha256(cache)}:
                    tracked=load_file(str(cache))
                else:
                    tracked=track_reference(root,image_path)
                    save_file({k:v.contiguous() for k,v in tracked.items()},str(cache))
                    atomic_json(cache_meta,{'source_sha256':sha256(image_path),'cache_sha256':sha256(cache)})
            model,loading=load_published_head(root);model.cuda()
            from gagavatar.runtime import GAGAvatarRuntime,GAGAvatarRuntimeConfig
            from gagavatar.assets import GAGAvatarAssets
            from gagavatar.libs.flame_model import FLAMEModel
            flame_path=root/'third_party/gaga_track/assets/flame/FLAME_with_eye.pt'
            assets=GAGAvatarAssets(root=root/'models/gag-head',model_path=root/'models/gag-head/GAGAvatar.pt',flame_model_path=flame_path)
            runtime=GAGAvatarRuntime.__new__(GAGAvatarRuntime)
            runtime.config=GAGAvatarRuntimeConfig(assets=assets,device='cuda');runtime.assets=assets
            runtime.device=torch.device('cuda');runtime.model=model
            runtime.flame_model=FLAMEModel(300,100,scale=5.,no_lmks=True,model_path=flame_path).cuda()
            runtime.tracked_avatars={};runtime.set_tracked_avatar(tracked,'video_reference')
            motion=torch.cat([tracked['expcode'],torch.zeros(3),tracked['posecode'][3:]])[None]
            params={k:v[0].detach() for k,v in runtime.forward_gaussians_for_batch(runtime.build_forward_batch(motion)).items()}
            flame=FLAMEModel(300,100,scale=5.,model_path=flame_path).cuda()
            shape=tracked['shapecode'][None].cuda();zero=torch.zeros(1,100,device='cuda');pose=torch.zeros(1,6,device='cuda')
            neutral,_=flame(shape,zero,pose,pose)
            mask,view,intrinsic=source_head_mask(tracked,flame)
            grid=(project(params['xyz'],view,intrinsic)/511*2-1).reshape(1,-1,1,2)
            opacity_mask=F.grid_sample(torch.from_numpy(mask).cuda()[None,None],grid,align_corners=True)[0,0,:,0]
            head_opacity=params['opacities']*opacity_mask[:,None]
            decoder={f'decoder.{key}':value.detach().cpu().clone() for key,value in model.upsampler.state_dict().items()}
            # Encoding is a build step. Offload all large networks before
            # loading the body; the small decoder is reloaded from the sidecar.
            model.cpu();runtime.flame_model.cpu();runtime=None;model=None
            gc.collect();torch.cuda.empty_cache()
            body=LHMRenderer.load(root,folder,768,use_head=False)
            args=body.arguments('idle',0,body.default_camera,np.zeros(100),np.zeros(3));sp=args['smplx_params']
            layer=body.model.renderer.smplx_model.smplx_layer
            mapping=np.load(json.loads((root/'models/assets-index.json').read_text())['flame_map'])
            target=layer(betas=sp['betas'],expression=torch.zeros(1,100,device='cuda'),body_pose=sp['body_pose'][:,0].reshape(1,63),
                global_orient=sp['root_pose'][:,0],jaw_pose=torch.zeros(1,3,device='cuda'),
                left_hand_pose=sp['lhand_pose'][:,0].reshape(1,45),right_hand_pose=sp['rhand_pose'][:,0].reshape(1,45),
                leye_pose=sp['leye_pose'][:,0],reye_pose=sp['reye_pose'][:,0],transl=sp['trans'][:,0]).vertices[0,mapping]
            scale,rotation,translation=fit_similarity(neutral[0],target)
            points=body.runtime_cache.last_gaussian.xyz.float()
            semantic=head_mask(body.state);face_points=points[semantic]
            low,high=face_points.min(0).values,face_points.max(0).values
            geometric=(points[:,1]>low[1]+(high[1]-low[1])*.10)&((points[:,0]-(low[0]+high[0])/2).abs()<(high[0]-low[0])*.8)
            anchors=torch.where(semantic&(points[:,1]>torch.quantile(face_points[:,1],.7)))[0]
            anchors=anchors[torch.linspace(0,len(anchors)-1,min(64,len(anchors)),device='cuda').long()]
            if len(anchors)<8:raise ValueError('头部姿态锚点不足')
            payload={**params,**decoder,'head_opacity':head_opacity,'shape':shape,'neutral':neutral,
                'registration_scale':scale,'registration_rotation':rotation,'registration_translation':translation,
                'anchors':points[anchors].clone(),'anchor_indices':anchors,'body_head_mask':(semantic|geometric).float()}
            metadata={**loading,'flame_sha256':sha256(flame_path),'source_frame':120,'source_seconds':4.,
                'source_frame_sha256':sha256(image_path),'native_head_size':512,'head_gaussians':len(params['xyz']),
                'registration':'FLAME_to_SMPLX_similarity_then_posed_scalp_anchors',
                'source_clothes':'red coat and green scarf excluded; grey-blue hoodie preserved',
                'estimated_regions':['head angles beyond 55 degrees blend to the original estimated 3D body head','back and hidden body areas'],
                'code_commit':'7abb72be5f311e7fb130c7f2e04a2d11a31420ba'}
            Image.fromarray((mask*255).astype('uint8')).save(output/'source-head-mask.png')
            Image.fromarray(tracked['image'].permute(1,2,0).mul(255).byte().numpy()).save(output/'source-reference.png')
            before=body.render('idle',0,body.camera_presets()['portrait'],np.zeros(100),np.zeros(3))
            Image.fromarray(before).save(output/'portrait-before.png')
            if sha256(folder/'state.json')!=report['body_state_sha256']:raise ValueError('构建期间身体状态发生变化')
            save_head_assets(folder,payload,metadata)
            body.close();body=None;flame.cpu();flame=None
            del payload,params,decoder,points,target
            gc.collect();torch.cuda.empty_cache()
            # Validate the persisted package through the normal runtime loader.
            body=LHMRenderer.load(root,folder,768)
            views=body.camera_presets();cases=[]
            for name,framing,yaw,jaw,action,seconds,position in [
                ('full','full',0,0,'idle',0,None),('portrait','portrait',0,0,'idle',0,None),
                ('side','portrait',30,0,'idle',0,None),('jaw','portrait',0,.25,'idle',0,None),
                ('back-estimated','portrait',180,0,'idle',0,None),('wave','full',0,0,'wave',.7,None),
                ('walk','full',0,0,'walk',.3,None),('moved','full',0,0,'idle',0,[.25,0,0])]:
                started=time.perf_counter()
                image=body.render(action,seconds,dict(views[framing],yaw=yaw),np.zeros(100),np.array([jaw,0,0]),position)
                Image.fromarray(image).save(output/f'{name}.png')
                cases.append({'name':name,'seconds':time.perf_counter()-started,'head_status':body.head_refiner.last_status})
            report.update(complete=True,loading=loading,cases=cases,head_package_sha256=sha256(descriptor),
                peak_allocated_mib=torch.cuda.max_memory_allocated()/1048576,
                body_unchanged=sha256(folder/'state.json')==report['body_state_sha256'])
            from backend.storage import Store,read_json
            store=Store(root);manifest=store.manifest(avatar_id)
            schema=read_json(descriptor)
            hashes=manifest.setdefault('artifact_hashes',{})
            hashes.update({'head.json':sha256(descriptor),schema['tensor_file']:schema['tensor_sha256']})
            manifest['head_detail']={'method':'GAGAvatar 3D head','native_size':512,
                'source_frame_sha256':metadata['source_frame_sha256'],'visual_accepted':False}
            store.write_manifest(avatar_id,manifest)
            print(json.dumps({'output':str(output),'complete':True,'cases':cases},ensure_ascii=False),flush=True)
        except BaseException as error:
            report['error']=f'{type(error).__name__}: {error}'
            if previous is None:descriptor.unlink(missing_ok=True)
            else:atomic_json(descriptor,json.loads(previous))
            raise
        finally:
            if body is not None:body.close()
            if model is not None:model.cpu()
            if flame is not None:flame.cpu()
            if runtime is not None:runtime.flame_model.cpu()
            atomic_json(output/'report.json',report)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--avatar-id',default='');parser.add_argument('--tracked-reference',type=Path)
    args=parser.parse_args();root=args.root.resolve()
    avatar_id=args.avatar_id or (root/'.runtime/default_avatar.txt').read_text().strip()
    build(root,avatar_id,args.tracked_reference)
