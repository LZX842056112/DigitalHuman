"""Adapter for the pinned, published LHMPP-700M neural renderer.

No image billboard, video replay or placeholder avatar is used in this adapter.
"""
from __future__ import annotations

import gc
import os
import sys
import threading
from pathlib import Path
import numpy as np
from PIL import Image

from backend.errors import PipelineError, NotReady
from backend.avatar.camera import orbit_camera
from backend.avatar.motion import sample_motion
from backend.avatar.state import save_state, load_state


def configure_upstream(root: Path):
    repo = root / "third_party/lhmpp"
    if not (repo / "core/models/modeling_humana4o_lrm.py").exists():
        raise NotReady("LHM++ 源码尚未安装，请运行 setup.ps1")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "HF_HOME": str(root / ".runtime/hf"), "TORCH_HOME": str(root / ".runtime/torch")})
    os.environ.setdefault('MAX_JOBS','4')
    # Upstream prior assets are referenced relative to the repository root.
    os.chdir(repo)
    return repo


def load_published_checkpoint(model, path: Path, renderer_type: str):
    import torch
    from safetensors.torch import load_model as load_safetensors
    # The published DPT-only decoder never executes these inherited attention
    # blocks. Its checkpoint omits them; keep the active DPT head unchanged.
    if renderer_type == 'patch_4dptonly':
        model.neural_renderer.transformer_block = torch.nn.ModuleList()
    # ArcFace has already loaded its separate published checkpoint strictly in
    # its constructor. It is not part of the LHMPP reconstruction checkpoint.
    identity = model._modules.pop('id_face_net', None)
    try:
        load_safetensors(model, str(path), strict=True)
    finally:
        if identity is not None:
            model.add_module('id_face_net', identity)


def load_model(root: Path):
    import torch
    configure_upstream(root)
    if not (root / "models/lhmpp/model.safetensors").is_file():
        raise NotReady("LHMPP-700M 权重尚未下载")
    if not torch.cuda.is_available():
        raise PipelineError("cuda", "CUDA 不可用，不能启动真实三维重建")
    from accelerate import PartialState
    PartialState()
    from core.models import model_dict
    from core.utils.hf_hub import wrap_model_hub
    from core.models.encoders.dinov2_wrapper import Dinov2Wrapper
    from backend.storage import read_json
    model_type = wrap_model_hub(model_dict["human_lrm_a4o"])
    # The full published checkpoint includes all 343 encoder tensors. Construct
    # that network without a second network download; strict loading forbids
    # silently using random or missing encoder parameters.
    original = Dinov2Wrapper._build_dinov2
    Dinov2Wrapper._build_dinov2 = staticmethod(lambda name,modulation_dim=None,pretrained=True:
                    original(name,modulation_dim=modulation_dim,pretrained=False))
    try:
        config = read_json(root / 'models/lhmpp/config.json')
        model = model_type(config=config)
        load_published_checkpoint(model, root / 'models/lhmpp/model.safetensors',
                                  config['neural_renderer']['type'])
    finally:
        Dinov2Wrapper._build_dinov2 = staticmethod(original)
    model.eval()
    model.requires_grad_(False)
    return model


def neutral_parameters(betas, width=512, height=512, device="cuda"):
    import torch
    z = lambda *shape: torch.zeros(*shape, device=device, dtype=torch.float32)
    return {
        "betas": torch.as_tensor(betas, dtype=torch.float32, device=device).reshape(1, -1),
        "root_pose": z(1,1,3), "body_pose": z(1,1,21,3), "jaw_pose": z(1,1,3),
        "leye_pose": z(1,1,3), "reye_pose": z(1,1,3), "lhand_pose": z(1,1,15,3),
        "rhand_pose": z(1,1,15,3), "trans": z(1,1,3), "expr": z(1,1,100),
        "focal": torch.tensor([[[height, height]]], device=device, dtype=torch.float32),
        "princpt": torch.tensor([[[width/2, height/2]]], device=device, dtype=torch.float32),
        "img_size_wh": torch.tensor([[[width, height]]], device=device, dtype=torch.float32),
    }


def fit_body_shape(estimator, image):
    import torch
    # Raw prediction mode avoids upstream's unused translation .view() on a
    # noncontiguous location tensor. Reconstruction consumes only these betas.
    prediction=estimator(image,is_shape=False)
    shapes=prediction.get('shape') if isinstance(prediction,dict) else None
    if not isinstance(shapes,torch.Tensor) or shapes.ndim!=2 or shapes.shape[0]<1 or not torch.isfinite(shapes).all():
        raise PipelineError('shape','身体照片未能拟合有效体型')
    return shapes[0].detach().float().cpu().numpy().copy()


def reconstruct(root: Path, folder: Path, manifest):
    import torch
    configure_upstream(root)
    free, total = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
    if free < 7000 * 1024**2:
        raise PipelineError("gpu_memory", f"重建需要约7GB可用显存。当前可用 {free // 1024**2} MiB。请释放其他程序的显存后重试。")
    from engine.pose_estimation import pose_estimator as pose_module
    body_path = folder / "source" / manifest["body_photos"][0]
    body = np.asarray(Image.open(body_path).convert("RGB"))
    # Multi-HMR's complete checkpoint supplies its backbone. Do not download
    # an additional DINO initialization while loading an offline checkpoint.
    original_factory=pose_module.Model
    def offline_factory(**kwargs):
        kwargs['pretrained_backbone']=False
        return original_factory(**kwargs)
    pose_module.Model=offline_factory
    try:
        estimator = pose_module.PoseEstimator("./pretrained_models/human_model_files", device="cpu")
    finally:
        pose_module.Model=original_factory
    estimator.to("cuda")
    betas = fit_body_shape(estimator,body)
    del estimator
    gc.collect()
    torch.cuda.empty_cache()

    from engine.BiRefNet.models.birefnet import BiRefNet
    segmenter = BiRefNet(bb_pretrained=False).eval()
    weights = root / "models/assets-index.json"
    from backend.storage import read_json
    index = read_json(weights)
    checkpoint = torch.load(index["birefnet"], map_location="cpu", weights_only=True)
    segmenter.load_state_dict({k.removeprefix("module."): v for k,v in checkpoint.items()}, strict=True)
    from torchvision import transforms
    image = Image.fromarray(body)
    tensor = transforms.Compose([transforms.Resize((1024,1024)), transforms.ToTensor(),
                                 transforms.Normalize([.485,.456,.406], [.229,.224,.225])])(image).unsqueeze(0)
    with torch.no_grad():
        mask = segmenter(tensor)[-1].sigmoid()
        mask = torch.nn.functional.interpolate(mask, size=body.shape[:2], mode="bilinear", align_corners=False)[0,0].numpy()
    del segmenter, checkpoint
    gc.collect()
    Image.fromarray((mask * 255).astype(np.uint8)).save(folder / "processed/body_mask.png")
    masked = body / 255. * mask[...,None] + (1-mask[...,None])
    from core.datasets.data_utils import SrcImagePipeline, src_center_crop_according_to_mask
    cropped, _, _, _, _ = src_center_crop_according_to_mask(masked, mask>.5, 5/3, [1.,1.])
    transform = SrcImagePipeline(dict(name="PadRatioWithScale", target_ratio=5/3, tgt_max_size_list=[840], val=True))
    processed = transform([cropped])[0]
    refs = torch.from_numpy(np.asarray(processed)).permute(2,0,1).float()[None,None].cuda()
    model = load_model(root).cuda()
    parameters = neutral_parameters(betas)
    c2w, intr = orbit_camera(0, 0, 3.2, [0,.8,0], 512,512)
    intrinsic = np.eye(4, dtype=np.float32)
    intrinsic[:3,:3] = intr
    # FP16 produced nonfinite reconstruction features with this checkpoint.
    # BF16 was verified on the real source material.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model.infer_single_view(refs, None, None,
                    render_c2ws=torch.as_tensor(c2w,device="cuda")[None,None],
                    render_intrs=torch.as_tensor(intrinsic,device="cuda")[None,None],
                    render_bg_colors=torch.ones(1,1,3,device="cuda"),
                    smplx_params=parameters, ref_imgs_bool=torch.ones(1,1,device="cuda",dtype=torch.bool))
    names = ["gs_model_list", "query_points", "transform_mat_neutral_pose", "gs_hidden_features", "image_latents", "motion_emb", "pos_emb"]
    state = dict(zip(names, output[:7]))
    state['reconstruction_precision'] = 'bfloat16'
    state["parameters"] = parameters
    state["parameters"]["transform_mat_neutral_pose"] = state["transform_mat_neutral_pose"]
    coords = state["query_points"]["neutral_coords"].reshape(-1,3)
    low,high = coords.min(dim=0).values,coords.max(dim=0).values
    center = ((low+high)/2).detach().cpu().tolist()
    state["camera_defaults"] = {"yaw":0.,"pitch":0.,"target":center,
                                "distance":float((high[1]-low[1]).detach().cpu())*.6/np.tan(np.deg2rad(17.5))}
    renderer = LHMRenderer(root, model, state)
    renderer.release_build_modules()
    renderer.fit_default_camera()
    save_state(folder, state)
    renderer.save_preview(folder)
    return renderer


class LHMRenderer:
    def __init__(self, root, model, state):
        self.root, self.model, self.state = root, model, state
        from backend.config import load_config
        render_config=load_config(root).get('render',{})
        width=render_config.get('width',512);height=render_config.get('height',512)
        if width!=height: raise ValueError('当前人物渲染需要正方形尺寸')
        self.configure_render_size(width)
        self.lock = threading.RLock()
        self.default_camera = state.get("camera_defaults",{"yaw":0.,"pitch":0.,"distance":3.2,"target":[0,0,0]})
        self._runtime_render=False
        self._closed=False
        self.head_refiner=None
        from backend.avatar.runtime_cache import install
        self.runtime_cache=install(self)

    @classmethod
    def load(cls, root, folder, render_size=None, *, use_head=True):
        import torch
        model = load_model(root)
        renderer=None;state=None
        try:
            # Move only the runtime networks, then warm up before connecting.
            model.renderer.cuda()
            model.neural_renderer.cuda()
            state = load_state(folder, "cuda")
            renderer=cls(root, model, state)
            if render_size is not None: renderer.configure_render_size(render_size)
            if renderer.runtime_cache is not None:
                renderer.render('idle',0,renderer.default_camera,np.zeros(100),np.zeros(3))
            if use_head and (folder/'head.json').is_file():
                from backend.avatar.head_refiner import HeadRefiner
                renderer.head_refiner=HeadRefiner.load(root,folder)
                renderer.render('idle',0,renderer.default_camera,np.zeros(100),np.zeros(3))
            return renderer
        except BaseException:
            if renderer is not None:
                renderer.close()
            else:
                model.renderer.cpu();model.neural_renderer.cpu()
                if state is not None: state.clear()
            raise

    def configure_render_size(self,size):
        if isinstance(size,bool) or size not in {512,768,1024}:
            raise ValueError('渲染尺寸必须为512、768或1024')
        self.render_size=int(size)

    def release_build_modules(self):
        import torch
        for name, module in self.model.named_children():
            if name not in {"renderer", "neural_renderer"}:
                module.cpu()
        gc.collect()
        torch.cuda.empty_cache()

    def close(self):
        with self.lock:
            if getattr(self,'_closed',False): return
            if getattr(self,'head_refiner',None) is not None:
                self.head_refiner.close()
                self.head_refiner=None
            if self.runtime_cache is not None:
                self.runtime_cache.close()
                self.runtime_cache=None
            # Loader/import frames can retain Python aliases. Explicitly move
            # their shared network objects off CUDA before releasing ownership.
            self.model.renderer.cpu()
            self.model.neural_renderer.cpu()
            self.state.clear()
            self.model=None
            self._closed=True

    def fit_default_camera(self):
        import torch
        from backend.avatar.camera import fit_bounds
        from backend.avatar.state import require_finite_tensor
        raster=self.model.renderer
        args=self.arguments('idle',0,self.default_camera,np.zeros(100),np.zeros(3))
        params=raster.get_single_view_smpl_data(args['smplx_params'],0)
        with torch.no_grad(), torch.autocast('cuda',dtype=torch.float16):
            models,_=raster.animate_gs_model(self.state['gs_model_list'][0],
                self.state['query_points']['neutral_coords'][0],raster._get_single_batch_data(params,0),
                mesh_meta=self.state['query_points']['mesh_meta'])
            coords=models[0].xyz.float()
            require_finite_tensor(coords)
            self.default_camera=fit_bounds(coords.min(0).values.cpu().numpy(),coords.max(0).values.cpu().numpy())
        self.state['camera_defaults']=self.default_camera

    def camera_presets(self):
        import torch
        from backend.avatar.camera import fit_bounds
        from backend.avatar.head_region import head_mask
        with self.lock,torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            raster=self.model.renderer
            args=self.arguments('idle',0,self.default_camera,np.zeros(100),np.zeros(3))
            params=raster.get_single_view_smpl_data(args['smplx_params'],0)
            models,_=raster.animate_gs_model(args['gs_model_list'][0],args['query_points']['neutral_coords'][0],
                raster._get_single_batch_data(params,0),mesh_meta=args['query_points']['mesh_meta'])
            head=models[0].xyz[head_mask(self.state)].float()
            portrait=fit_bounds(head.min(0).values.cpu().numpy(),head.max(0).values.cpu().numpy(),margin=.15)
        portrait['distance']=max(portrait['distance'],self.camera_limits['min_distance'])
        return {'full':dict(self.default_camera),'portrait':portrait}

    @property
    def camera_limits(self):
        return {'min_distance':1.2*self.render_size/768 if getattr(self,'head_refiner',None) is not None else .3}

    def arguments(self, action, seconds, camera, expression, jaw, position=None):
        import torch
        pose = sample_motion(action, seconds)
        params = {k: v.clone() if isinstance(v, torch.Tensor) else v for k,v in self.state["parameters"].items()}
        for name in ("body_pose", "root_pose", "trans"):
            params[name] = torch.as_tensor(pose[name], device="cuda").reshape_as(params[name])
        if position is not None:
            params["trans"] += torch.as_tensor(position, device="cuda").reshape(1,1,3)
        params["expr"] = torch.as_tensor(expression, device="cuda", dtype=torch.float32).reshape(1,1,100)
        params["jaw_pose"] = torch.as_tensor(jaw, device="cuda", dtype=torch.float32).reshape(1,1,3)
        size=getattr(self,'render_size',512)
        c2w, intr = orbit_camera(camera["yaw"], camera["pitch"], camera["distance"], camera["target"], size,size)
        intrinsic = np.eye(4, dtype=np.float32)
        intrinsic[:3,:3] = intr
        args = {name: self.state[name] for name in ("gs_model_list", "query_points", "gs_hidden_features", "image_latents", "motion_emb", "pos_emb")}
        args.update(smplx_params=params, render_c2ws=torch.as_tensor(c2w,device="cuda")[None,None],
                    render_intrs=torch.as_tensor(intrinsic,device="cuda")[None,None],
                    render_bg_colors=torch.ones(1,1,3,device="cuda"), infer_output_renderer="neural",render_size=size)
        return args

    def render(self, action, seconds, camera, expression, jaw, position=None):
        import torch
        with self.lock:
            if getattr(self,'_closed',False): raise RuntimeError('人物已卸载，请重新加载人物包')
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                self._runtime_render=True
                try:
                    args = self.arguments(action, seconds, camera, expression, jaw, position)
                    rgb, _ = self._render_differentiable(args)
                    if getattr(self,'head_refiner',None) is not None:
                        refined=self.head_refiner.apply(args,rgb[0,0],self.runtime_cache.last_gaussian)
                        rgb=refined[None,None]
                finally:
                    self._runtime_render=False
            rgb = rgb[0,0]
            return rgb.clamp(0,1).mul(255).to(torch.uint8).cpu().numpy()

    def render_canvas(self, args):
        patch = int(getattr(self.model.neural_renderer,'enc_patch_size',1))
        # Pad for the published decoder's seven-pixel patches. Crop native
        # pixels afterward; never resize a smaller rendered image.
        multiple = patch if patch % 2 == 0 else patch * 2
        requested=int(args.get('render_size',512))
        size = (requested + multiple - 1) // multiple * multiple
        border = (size - requested) // 2
        if border:
            args = dict(args)
            intrinsic = args['render_intrs'].clone()
            intrinsic[...,0,2] += border
            intrinsic[...,1,2] += border
            args['render_intrs'] = intrinsic
        return args, size, border

    def render_differentiable(self, args):
        import torch
        with torch.autocast('cuda',dtype=torch.float16):
            return self._render_differentiable(args)

    def _render_differentiable(self, args):
        model = self.model
        requested=int(args.get('render_size',512))
        args, size, border = self.render_canvas(args)
        params = model.renderer.get_single_view_smpl_data(args["smplx_params"], 0)
        result = model.renderer.forward_animate_gs(args["gs_model_list"], args["query_points"], params,
                    args["render_c2ws"], args["render_intrs"], size,size, args["render_bg_colors"],
                    patch_size=model.neural_renderer_patch_size, features=args["gs_hidden_features"])
        features = result["comp_features"]
        if model.neural_renderer_input == "rgb+feats":
            import torch
            features = torch.cat([features, result["comp_rgb"]], dim=2)
        cache=getattr(self,'runtime_cache',None)
        neural=cache.neural if cache is not None and cache.active() else model.neural_renderer
        rgb, mask = neural(args["image_latents"], features, args["motion_emb"],
                                         size,size, pos_emb_list=args["pos_emb"])
        rgb = rgb[:,:,border:border+requested,border:border+requested]
        if mask is not None:
            mask = mask[:,:,border:border+requested,border:border+requested]
        return rgb, mask

    def save_preview(self, folder):
        for label, yaw in (("front",0),("side",45),("back_estimated",180)):
            camera = dict(self.default_camera,yaw=yaw)
            image = self.render("idle", 0, camera, np.zeros(100), np.zeros(3))
            Image.fromarray(image).save(folder / f"{label}.png")
