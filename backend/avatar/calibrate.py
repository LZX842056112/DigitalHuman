"""Subject-only head refinement. Body and network weights remain frozen."""
from pathlib import Path
import subprocess
import sys
import math
import time
import cv2
import numpy as np

from backend.storage import read_json, atomic_json
from backend.errors import PipelineError
from backend.avatar.state import save_state


class ValidationCheckpoint:
    def __init__(self,parameters,baseline):
        if not math.isfinite(baseline):
            raise ValueError('面部初始验证误差出现无效数值')
        self.parameters=list(parameters)
        self.loss=float(baseline)
        self.step=0
        self.values=[p.detach().clone() for p in self.parameters]
        self.curve=[{'step':0,'loss':self.loss}]

    def consider(self,loss,step):
        if not math.isfinite(loss):
            raise ValueError('面部验证误差出现无效数值')
        self.curve.append({'step':step,'loss':float(loss)})
        if loss<self.loss:
            self.loss,self.step=float(loss),step
            self.values=[p.detach().clone() for p in self.parameters]

    def restore(self):
        import torch
        with torch.no_grad():
            for parameter,value in zip(self.parameters,self.values):
                parameter.copy_(value)
        return self.step>0


def fit_camera(points, observations, intrinsic):
    points = np.asarray(points,np.float64)
    observations = np.asarray(observations,np.float64)
    intrinsic = np.asarray(intrinsic,np.float64)
    if len(points) < 6 or observations.shape != (len(points),2) or not np.isfinite(observations).all():
        raise ValueError("需要至少六个真实面部关键点")
    ok,rvec,tvec = cv2.solvePnP(points,observations,intrinsic,None,flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        raise PipelineError("face_camera","面部相机拟合失败")
    rvec,tvec = cv2.solvePnPRefineLM(points,observations,intrinsic,None,rvec,tvec)
    projected,_ = cv2.projectPoints(points,rvec,tvec,intrinsic,None)
    error = float(np.median(np.linalg.norm(projected.reshape(-1,2)-observations,axis=1)))
    rotation,_ = cv2.Rodrigues(rvec)
    w2c = np.eye(4,dtype=np.float32)
    w2c[:3,:3],w2c[:3,3] = rotation,tvec.reshape(3)
    if tvec[2,0] <= 0:
        raise PipelineError("face_camera","面部相机位于错误的投影方向")
    return np.linalg.inv(w2c),error


def head_deltas(features, offsets, indices):
    import torch
    return (torch.nn.Parameter(torch.zeros_like(features[:,indices],dtype=torch.float32)),
            torch.nn.Parameter(torch.zeros_like(offsets[indices],dtype=torch.float32)))


def identity_embedding(network, image):
    import torch.nn.functional as F
    # The pinned ArcFace fc5 consumes 512*8*8 features from 128-pixel inputs.
    image = F.interpolate(image,size=(128,128),mode='bilinear',align_corners=False)*2-1
    channel_count = next(p for p in network.parameters() if p.ndim == 4).shape[1]
    if channel_count == 1:
        image = image.mean(dim=1,keepdim=True)
    return F.normalize(network(image).flatten(1),dim=1)


def calibrate_head(root: Path, folder: Path, renderer, steps=120,landmark_weight=0.,balance_photos=False,progress=None,preservation_weight=0.):
    cache=getattr(renderer,'runtime_cache',None)
    if cache is not None and not len(cache.skinning.dynamic):
        from backend.avatar.head_region import head_mask
        with cache.skinning.fixed_head_calibration(head_mask(renderer.state)):
            return _calibrate_head(root,folder,renderer,steps,landmark_weight,balance_photos,progress,preservation_weight)
    return _calibrate_head(root,folder,renderer,steps,landmark_weight,balance_photos,progress,preservation_weight)


def _calibrate_head(root: Path, folder: Path, renderer, steps=120,landmark_weight=0.,balance_photos=False,progress=None,preservation_weight=0.):
    import torch
    import torch.nn.functional as F
    import lpips
    from backend.avatar.head_alignment import make_landmark_anchors,landmark_projection_loss,training_case,outside_appearance_loss,outside_appearance_mask
    from scripts.track_faces import tracking_current
    tracking = folder / "processed/tracking/flame.json"
    frames = read_json(folder/"processed/faces/frames.json")["selected"]
    if not tracking_current(root,folder,frames):
        subprocess.run([sys.executable,str(root/"scripts/track_faces.py"),"--root",str(root),"--avatar",str(folder)],check=True,cwd=root)
    tracked = read_json(tracking)
    state = renderer.state
    layer = renderer.model.renderer.smplx_model.smplx_layer
    cases = []
    alignments={}
    def posed_xyz(args):
        raster=renderer.model.renderer
        single=raster.get_single_view_smpl_data(args['smplx_params'],0)
        models,_=raster.animate_gs_model(args['gs_model_list'][0],args['query_points']['neutral_coords'][0],
            raster._get_single_batch_data(single,0),mesh_meta=args['query_points']['mesh_meta'])
        return models[0].xyz
    for item in frames:
        result = tracked.get(str(item["tracking_frame"]))
        if not result:
            continue
        image = cv2.cvtColor(cv2.imread(str(folder/"processed/faces"/item["file"])),cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(512,512))
        if result.get('file') != item['file'] or result.get('crop_size') != [512,512]:
            raise PipelineError('face_identity','跟踪关键点与源视频帧不一致，请重新跟踪')
        observed = np.asarray(result['observed_lmks68'],np.float64)
        expression = np.asarray(result["expcode"],np.float32).reshape(-1)
        if expression.size != 100:
            raise PipelineError("face_expression","跟踪器未输出100维FLAME表情")
        jaw = np.asarray(result["posecode"],np.float32).reshape(-1)[3:6]
        args = renderer.arguments("idle",0,renderer.default_camera,expression,jaw)
        args['render_size']=512
        sp = args["smplx_params"]
        with torch.no_grad():
            output = layer(betas=sp["betas"],expression=sp["expr"][:,0],body_pose=sp["body_pose"][:,0].reshape(1,63),
                           global_orient=sp["root_pose"][:,0],jaw_pose=sp["jaw_pose"][:,0],
                           left_hand_pose=sp["lhand_pose"][:,0].reshape(1,45),right_hand_pose=sp["rhand_pose"][:,0].reshape(1,45),
                           leye_pose=sp["leye_pose"][:,0],reye_pose=sp["reye_pose"][:,0],transl=sp["trans"][:,0])
        if output.joints.shape[1] < 68:
            raise PipelineError("face_landmarks","SMPL-X未提供完整面部关键点")
        # SMPL-X gives 51 inner landmarks followed by the 17-point contour.
        landmarks = output.joints[0,-68:].cpu().numpy()
        landmarks = np.concatenate([landmarks[-17:],landmarks[:51]])
        intr = np.array([[650,0,256],[0,650,256],[0,0,1]],np.float64)
        c2w,error = fit_camera(landmarks,observed,intr)
        if error > 8:
            continue
        intr4 = np.eye(4,dtype=np.float32)
        intr4[:3,:3] = intr
        args["render_c2ws"] = torch.as_tensor(c2w,device="cuda")[None,None]
        args["render_intrs"] = torch.as_tensor(intr4,device="cuda")[None,None]
        mask = np.zeros((512,512),np.float32)
        cv2.fillConvexPoly(mask,cv2.convexHull(observed.astype(np.int32)),1)
        mask = cv2.GaussianBlur(cv2.dilate(mask,np.ones((15,15),np.uint8)),(15,15),3)
        cases.append((args,torch.from_numpy(image/255).float().cuda()[None].permute(0,3,1,2),
                      torch.from_numpy(mask).cuda()[None,None],error,item["file"]))
        if landmark_weight>0:
            from backend.avatar.head_region import head_mask
            with torch.no_grad():
                ids,offset=make_landmark_anchors(posed_xyz(args),torch.as_tensor(landmarks,device='cuda'),head_mask(state))
                alignments[item['file']]=(ids,offset,torch.as_tensor(np.linalg.inv(c2w),device='cuda'),
                    torch.as_tensor(intr,device='cuda'),torch.as_tensor(observed,device='cuda'))
    if len(cases) < 4:
        raise PipelineError("face_alignment","少于四帧通过真实关键点对齐，不能完成面部校准")
    original_features = state["gs_hidden_features"].detach().float()
    coords = state["query_points"]["neutral_coords"]
    if coords.ndim == 3:
        coords = coords[0]
    from backend.avatar.head_region import head_mask
    mask = head_mask(state)
    if original_features.shape[-2] != len(mask):
        raise PipelineError("face_features","模型特征与高斯点数量不一致，不能安全限定头部优化")
    indices = torch.where(mask)[0]
    gs = state["gs_model_list"][0]
    original_offset = gs.offset_xyz.detach().float()
    feature_delta,offset_delta = head_deltas(original_features,original_offset,indices)
    optimizer = torch.optim.Adam([feature_delta,offset_delta],lr=.001)
    scaler = torch.cuda.amp.GradScaler()
    perceptual = lpips.LPIPS(net="alex").cuda().eval().requires_grad_(False)
    face_net = renderer.model.id_face_net.cuda().eval().requires_grad_(False)
    def identity(image):
        return identity_embedding(face_net,image)
    def apply_delta():
        features = original_features.clone()
        features[:,indices] = features[:,indices]+feature_delta
        offsets = original_offset.clone()
        offsets[indices] = offsets[indices]+offset_delta.clamp(-.01,.01)
        gs.offset_xyz = offsets
        return features
    preserved_frames={};preservation_masks={}
    if preservation_weight>0:
        with torch.no_grad():
            for case in cases:
                args=dict(case[0]);args['gs_hidden_features']=apply_delta()
                image,_=renderer.render_differentiable(args)
                preserved_frames[case[4]]=image[0,0].permute(2,0,1)[None].float().detach()
                preservation_masks[case[4]]=outside_appearance_mask(case[2]).detach()
    def loss_for(case,components=False):
        args,target,region,_,_ = case
        args = dict(args)
        args["gs_hidden_features"] = apply_delta()
        predicted,_ = renderer.render_differentiable(args)
        predicted = predicted[0,0].permute(2,0,1)[None].float()
        rgb = ((predicted-target).abs()*region).sum()/(region.sum()*3+1e-6)
        lp = perceptual(predicted*region*2-1,target*region*2-1).mean()
        cosine = 1-(identity(predicted*region)*identity(target*region)).sum()
        landmark=rgb.new_zeros(())
        if landmark_weight>0:
            landmark=landmark_projection_loss(posed_xyz(args),*alignments[case[4]])
        preservation=outside_appearance_loss(predicted,preserved_frames[case[4]],region,preservation_masks[case[4]]) if preservation_weight>0 else rgb.new_zeros(())
        total=rgb+.1*lp+.05*cosine+landmark_weight*landmark+preservation_weight*preservation+.01*feature_delta.square().mean()+offset_delta.square().mean()
        if components:
            return total,{name:float(value) for name,value in {'rgb':rgb,'lpips':lp,'identity':cosine,'landmark':landmark,'outside':preservation}.items()}
        return total
    training,held = cases[:-1],cases[-1]
    with torch.no_grad():
        initial,before_components=loss_for(held,True)
        before=float(initial)
    def save_comparison(label):
        if not balance_photos: return
        from PIL import Image
        destination=folder/'processed/calibration_previews'
        destination.mkdir(parents=True,exist_ok=True)
        for case in [c for c in training if c[4].startswith('photo_')]+[held]:
            args=dict(case[0]);args['gs_hidden_features']=apply_delta()
            with torch.no_grad(): image,_=renderer.render_differentiable(args)
            image=(image[0,0].clamp(0,1).detach().cpu().numpy()*255).astype(np.uint8)
            Image.fromarray(image).save(destination/f'{Path(case[4]).stem}_{label}.png')
    save_comparison('before')
    checkpoint=ValidationCheckpoint([feature_delta,offset_delta],before)
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for(training_case(training,step,balance_photos))
        if not torch.isfinite(loss):
            raise PipelineError("face_calibration","面部校准损失无效")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([feature_delta,offset_delta],1.)
        scaler.step(optimizer)
        scaler.update()
        if not torch.isfinite(feature_delta).all() or not torch.isfinite(offset_delta).all():
            raise PipelineError('face_calibration','优化增量出现无效数值，人物包未更新')
        if step<10 or (step+1)%10==0 or step+1==steps:
            with torch.no_grad():
                checkpoint.consider(float(loss_for(held)),step+1)
            print(f'Head calibration {step+1}/{steps}: validation={checkpoint.curve[-1]["loss"]:.6f}, best={checkpoint.loss:.6f}',flush=True)
            if progress is not None: progress(step+1,steps)
    with torch.no_grad():
        improved=checkpoint.restore()
        final,after_components=loss_for(held,True)
        after=float(final)
        save_comparison('candidate')
        image_not_worse=landmark_weight<=0 or after_components['rgb']<=before_components['rgb']
        preserved=preservation_weight<=0 or after_components['outside']<=.01
        report={"frames":len(cases),"steps":steps,"selected_step":checkpoint.step,
                "heldout_before":before,"heldout_after":after,'validation_curve':checkpoint.curve,
                'validation_frame':held[4],'training_frames':[c[4] for c in training],
                'selection':'minimum validation loss; no validation gradients',
                "landmark_median_px":[c[3] for c in cases],"body_frozen":True,
                "network_frozen":True,"position_limit_m":.01,'accepted':improved and after<=before and image_not_worse and preserved,
                'landmark_weight':landmark_weight,'balanced_photo_updates':balance_photos,
                'preservation_weight':preservation_weight,'outside_mean_limit':.01 if preservation_weight>0 else None,
                'validation_components_before':before_components,'validation_components_after':after_components,
                'landmark_proxy':'SMPL-X landmarks anchored to nearest head Gaussians' if landmark_weight>0 else 'not_used',
                'visual_approval':'pending'}
        atomic_json(folder/'processed/calibration_attempts'/f'{time.time_ns()}.json',report)
        if not improved or after>before or not image_not_worse or not preserved:
            gs.offset_xyz = original_offset
            raise PipelineError("face_calibration",f"验证总误差或图像误差未通过，校准结果未采用：{before:.6f} → {after:.6f}")
        state["gs_hidden_features"] = apply_delta().detach()
        gs.offset_xyz = gs.offset_xyz.detach()
    save_state(folder,state)
    atomic_json(folder/"face_calibration.json",report)
    del perceptual
    face_net.cpu()
    torch.cuda.empty_cache()
