"""Exact per-avatar skinning caches and inference-only GPU graph reuse."""
import importlib
import time
from contextlib import contextmanager
import torch


def token(tensor):
    return (tensor.data_ptr(),tensor._version,tuple(tensor.shape),str(tensor.device),tensor.dtype)


class PositionCache:
    """Reuse only the decoder's constant image-coordinate embedding."""
    def __init__(self,head):
        self.original=head._apply_pos_embed
        self.entries={}

    def __call__(self,features,width,height,ratio=.1):
        key=(tuple(features.shape[1:]),width,height,ratio,features.dtype,features.device)
        if key not in self.entries:
            with torch.no_grad():
                self.entries[key]=self.original(torch.zeros_like(features[:1]),width,height,ratio)
            if len(self.entries)>8:
                self.entries.pop(next(iter(self.entries)))
        return features+self.entries[key]


def pose_requested_views(raster,attributes,points,data,mesh_meta,canonical):
    transformed=raster._transform_points(data,points,attributes.offset_xyz,points.device,mesh_meta)
    rotations=raster._compute_rotations(transformed['transform_mat_posed'],attributes.rotation,
                                         points.device,transformed['mesh_meta'])
    factory=type(canonical[0])
    posed=[factory(xyz=transformed['posed_coords'][index],opacity=attributes.opacity,
                   rotation=rotations[index],scaling=attributes.scaling,shs=attributes.shs,
                   use_rgb=raster.gs_net.use_rgb) for index in range(data['body_pose'].shape[0])]
    return posed,canonical


class SkinningCache:
    def __init__(self,skin,signature):
        self.skin,self.signature=skin,signature
        self.original_matrices=skin.get_transform_mat_vertex
        self.original_joints=skin.get_zero_pose_human
        fixed=(skin.is_rhand+skin.is_lhand+skin.is_face)>0
        self.fixed=fixed
        self.dynamic=torch.where((skin.expr_dirs.abs().sum((1,2))>0)&~fixed)[0]
        self.weight_key=None;self.weights=None;self.shape=None;self.joints=None
        self.head_context=None;self.head_weights={}

    @contextmanager
    def fixed_head_calibration(self,trainable_mask):
        mask=trainable_mask.to(self.fixed.device).reshape(-1).bool()
        if mask.shape!=self.fixed.shape or bool((mask&~self.fixed).any()) or len(self.dynamic):
            raise ValueError('头部缓存仅支持固定蒙皮区域，躯干坐标和表情权重必须固定')
        if self.head_context is not None: raise RuntimeError('头部缓存上下文已经启用')
        self.head_context=object();self.head_weights={}
        try:
            yield
        finally:
            self.head_context=None;self.head_weights={}

    def matrices(self,transform,points,mask):
        if self.head_context is not None:
            batch=transform.shape[0];key=(batch,points.dtype,str(points.device))
            if key not in self.head_weights:
                with torch.no_grad():
                    weights=self.skin.query_voxel_skinning_weights(points.detach().reshape(1,-1,3)).reshape(batch,-1,self.skin.base_skinning.joint_num)
                    original=self.skin.skinning_weight[None].expand(batch,-1,-1)
                    weights[mask]=original[mask]
                    self.head_weights[key]=weights.detach()
            return (self.head_weights[key]@transform.reshape(batch,self.skin.base_skinning.joint_num,16)).reshape(batch,-1,4,4)
        if torch.is_grad_enabled(): return self.original_matrices(transform,points,mask)
        batch=transform.shape[0]
        key=(self.signature(),batch,points.dtype,str(points.device))
        if key!=self.weight_key:
            weights=self.skin.query_voxel_skinning_weights(points.reshape(1,-1,3)).reshape(batch,-1,self.skin.base_skinning.joint_num)
            original=self.skin.skinning_weight[None].expand(batch,-1,-1)
            weights[mask]=original[mask]
            self.weights,self.weight_key=weights,key
        weights=self.weights
        if len(self.dynamic):
            weights=weights.clone()
            weights[:,self.dynamic]=self.skin.query_voxel_skinning_weights(points[:,self.dynamic].reshape(1,-1,3)).reshape(batch,len(self.dynamic),-1)
        return (weights@transform.reshape(batch,self.skin.base_skinning.joint_num,16)).reshape(batch,-1,4,4)

    def zero_pose(self,shape_param,device,face_offset,joint_offset,return_mesh=False):
        needs_grad=torch.is_grad_enabled() and (self.head_context is None or shape_param.requires_grad)
        if needs_grad or face_offset is not None or joint_offset is not None or return_mesh:
            return self.original_joints(shape_param,device,face_offset,joint_offset,return_mesh)
        if shape_param.is_cuda and torch.cuda.is_current_stream_capturing(): return self.joints
        if self.shape is None or self.shape.device!=shape_param.device or not torch.equal(self.shape,shape_param):
            self.shape=shape_param.detach().clone()
            self.joints=self.original_joints(shape_param,device,face_offset,joint_offset,return_mesh)
        return self.joints


class RuntimeCache:
    def __init__(self,renderer):
        self.renderer=renderer;self.raster=renderer.model.renderer
        self.skin=self.raster.smplx_model
        self.skinning=SkinningCache(self.skin,self.signature)
        self.skin.get_transform_mat_vertex=self.skinning.matrices
        self.skin.get_zero_pose_human=self.skinning.zero_pose
        self.original_prepare=self.raster._prepare_smplx_data
        self.original_animate=self.raster.animate_gs_model
        self.original_single=self.raster.forward_single_view
        self.raster._prepare_smplx_data=self.prepare
        self.raster.animate_gs_model=self.animate
        self.raster.forward_single_view=self.single_view
        self.canonical=None;self.graph=None;self.inputs=None;self.output=None;self.graph_key=None
        self.graph_resources=None
        self.last_gaussian=None
        self.graph_shape=None;self.graph_error=None
        self.pose_only=True;self.canonical_models=None
        self.compiled=None;self.compile_error=None
        from backend.config import load_config
        render_config=load_config(renderer.root).get('render',{})
        self.compile_enabled=bool(render_config.get('compile_neural',False))
        self.raster_tile_size=int(render_config.get('raster_tile_size',16))
        if self.raster_tile_size not in {8,16}: raise ValueError('高斯分块尺寸必须为8或16')
        self.stats={'dynamic_weight_rows':len(self.skinning.dynamic),'features_only':True,'geometry_graph':False,
                    'neural_compiled':False,'raster_tile_size':self.raster_tile_size}
        head=getattr(renderer.model.neural_renderer,'dense_predictor',None)
        self.position=None
        if head is not None and getattr(head,'pos_embed',False):
            self.position=PositionCache(head)
            head._apply_pos_embed=self.position
            self.stats['neural_position_cache']=True
        from backend.avatar import kinematics
        for module_name in ['core.models.rendering.smplx_gsavatar.lbs','core.models.rendering.smplx.smplx.lbs','core.models.rendering.skinnings.smplx_voxel_skinning']:
            importlib.import_module(module_name).batch_rigid_transform=kinematics.batch_rigid_transform
        importlib.import_module('core.models.rendering.skinnings.smplx_voxel_skinning').axis_angle_to_matrix=kinematics.axis_angle_to_matrix
        importlib.import_module('core.models.rendering.base_gs_render').matrix_to_quaternion=kinematics.matrix_to_quaternion
        self.skin.smplx_layer.parents=self.skin.smplx_layer.parents.cpu()

    def active(self):
        return self.renderer._runtime_render and not torch.is_grad_enabled()

    def close(self):
        self.raster._prepare_smplx_data=self.original_prepare
        self.raster.animate_gs_model=self.original_animate
        self.raster.forward_single_view=self.original_single
        self.skin.get_transform_mat_vertex=self.skinning.original_matrices
        self.skin.get_zero_pose_human=self.skinning.original_joints
        if self.position is not None:
            self.renderer.model.neural_renderer.dense_predictor._apply_pos_embed=self.position.original
            self.position.entries.clear()
        self.graph=self.inputs=self.output=self.graph_resources=None
        self.last_gaussian=None
        self.graph_key=self.graph_shape=self.canonical=self.canonical_models=None
        self.compiled=None
        self.skinning.weights=self.skinning.joints=self.skinning.shape=None
        self.skinning.head_weights.clear()
        self.skinning.fixed=self.skinning.fixed.cpu()
        self.skinning.dynamic=self.skinning.dynamic.cpu()

    def neural(self,*args,**kwargs):
        original=self.renderer.model.neural_renderer
        if not self.active() or not self.compile_enabled or self.compile_error: return original(*args,**kwargs)
        started=time.perf_counter() if self.compiled is None else None
        try:
            if self.compiled is None:
                with torch.enable_grad(),torch.autocast('cuda',enabled=False):
                    self.compiled=torch.compile(original,fullgraph=True,dynamic=False,mode='reduce-overhead')
            result=self.compiled(*args,**kwargs)
            self.stats['neural_compiled']=True
            if started is not None: self.stats['neural_first_call_seconds']=time.perf_counter()-started
            return result
        except Exception as error:
            self.compile_error=str(error);self.compiled=None
            self.stats.update(neural_compiled=False,neural_compile_error=str(error)[:4000])
            return original(*args,**kwargs)

    def signature(self):
        state=self.renderer.state
        attributes=state['gs_model_list'][0]
        return (token(attributes.offset_xyz),token(state['query_points']['neutral_coords']))

    def prepare(self,data):
        if not self.active(): return self.original_prepare(data)
        if self.canonical is None:
            merged=self.original_prepare(data)
            self.canonical={key:merged[key][-1:].detach().clone() for key in ['root_pose','body_pose','jaw_pose','leye_pose','reye_pose','lhand_pose','rhand_pose','expr','trans']}
        merged={key:torch.cat([data[key],value],0) for key,value in self.canonical.items()}
        merged.update(betas=data['betas'],transform_mat_neutral_pose=data['transform_mat_neutral_pose'])
        return merged

    def animate(self,attributes,points,data,**kwargs):
        if not self.active() or self.graph_error:
            return self.original_animate(attributes,points,data,**kwargs)
        key=(self.pose_only,self.signature(),tuple((name,tuple(value.shape),value.dtype) for name,value in data.items()),
             tuple(token(getattr(attributes,name)) for name in ['opacity','rotation','scaling','shs']),
             token(self.renderer.state['parameters']['transform_mat_neutral_pose']))
        if key!=self.graph_key or self.graph_shape is None or not torch.equal(self.graph_shape,data['betas']):
            self.graph=self.inputs=self.output=self.graph_resources=None
            try:
                inputs={name:value.clone() for name,value in data.items()}
                if self.pose_only:
                    self.canonical_models=self.original_animate(attributes,points,inputs,**kwargs)[1]
                    forward=lambda:pose_requested_views(self.raster,attributes,points,inputs,kwargs.get('mesh_meta'),self.canonical_models)
                else:
                    forward=lambda:self.original_animate(attributes,points,inputs,**kwargs)
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream),torch.no_grad():
                    for _ in range(3): forward()
                torch.cuda.current_stream().wait_stream(stream)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph),torch.no_grad():
                    output=forward()
                self.graph,self.inputs,self.output=graph,inputs,output
                self.graph_resources=(self.skinning.weights,self.skinning.joints,points,
                    *(getattr(attributes,name) for name in ['offset_xyz','opacity','rotation','scaling','shs']),
                    *getattr(self.skinning,'head_weights',{}).values())
                self.graph_shape=data['betas'].detach().clone();self.graph_key=key
                self.stats['geometry_graph']=True
                self.stats['canonical_pose_cached']=self.pose_only
            except RuntimeError as error:
                self.graph_error=str(error);self.stats.update(geometry_graph=False,geometry_graph_error=str(error))
                self.graph=self.inputs=self.output=self.graph_resources=None
                return self.original_animate(attributes,points,data,**kwargs)
        for name,value in data.items():
            if name not in ['betas','transform_mat_neutral_pose','focal','princpt','img_size_wh']:
                self.inputs[name].copy_(value)
        self.graph.replay()
        return self.output

    def single_view(self,gaussian,camera,background,features=None,patch_size=14,**kwargs):
        self.last_gaussian=gaussian
        if not self.active() or features is None:
            return self.original_single(gaussian,camera,background,features=features,patch_size=patch_size,**kwargs)
        module=importlib.import_module('core.models.rendering.gsplat_renderer')
        xyz,_,_,opacity,scales,rotations,_=self.raster.get_gaussians_properties(camera,gaussian)
        height,width=int(camera.height)//patch_size,int(camera.width)//patch_size
        intrinsic=module.scale_intrs(camera.intrinsic.clone(),width/int(camera.width),height/int(camera.height))
        index=(background[:1]/.5).long()
        embedding=self.raster.background_embedding.index_select(0,index).squeeze(0)
        with torch.autocast('cuda',enabled=False):
            image,alpha,_=module.rasterization(means=xyz.float(),quats=rotations.float(),scales=scales.float(),
                opacities=opacity.squeeze(1).float(),colors=features.float(),
                viewmats=camera.world_view_transform.T.contiguous()[None].float(),Ks=intrinsic[None,:3,:3].float(),
                width=width,height=height,near_plane=camera.znear,far_plane=camera.zfar,eps2d=.3,render_mode='RGB',
                backgrounds=embedding[None].float(),camera_model='pinhole',channel_chunk=128,tile_size=self.raster_tile_size)
        return {'comp_features':image[0],'comp_mask':alpha[0]}


def install(renderer):
    raster=renderer.model.renderer
    if not hasattr(raster,'smplx_model') or renderer.model.neural_renderer_input!='feats': return None
    return RuntimeCache(renderer)
