"""Thin photo-only adapter around the pinned Ditto inference components."""
import copy
import gc
import math
import os
import sys
from backend import clock
import uuid
from pathlib import Path
import av
import cv2
import numpy as np
import torch
from backend.emotion import portrait_emotion
from backend.gpu import gpu_operation
from backend.media import read_audio
from backend.storage import read_json,sha256,atomic_json
from backend.portrait.assets import portrait_version,read_published_config,verified_model
from backend.portrait.cache import save_cache,load_cache


def photo_options(defaults, emotion):
    return {**defaults,'max_size':1024,'ch_info':None,'emo':portrait_emotion(emotion),
        'use_d_keys':{'exp':1.,'pitch':.25,'yaw':.25,'roll':.25,'t':0.},'fade_type':'s',
        'overall_ctrl_info':{},'online_mode':False}


def pasteback(photo, face, matrix, mask):
    height,width=photo.shape[:2]
    alpha=cv2.warpAffine(mask,matrix[:2],(width,height),flags=cv2.INTER_LINEAR).clip(0,1)[...,None]
    image=cv2.warpAffine(face,matrix[:2],(width,height),flags=cv2.INTER_LINEAR)
    return (image*alpha+photo*(1-alpha)).clip(0,255).astype(np.uint8)


class PhotoEngine:
    def __init__(self,root):
        self.root=Path(root);self.loaded=False;self.networks=[]

    def _load(self):
        if self.loaded:return
        folder=verified_model(self.root,'ditto')
        repo=self.root/'third_party/ditto'
        sys.path.insert(0,str(repo)) if str(repo) not in sys.path else None
        from core.atomic_components import cfg
        from core.atomic_components.avatar_registrar import AvatarRegistrar
        from core.atomic_components.audio2motion import Audio2Motion
        from core.atomic_components.motion_stitch import MotionStitch
        from core.atomic_components.warp_f3d import WarpF3D
        from core.atomic_components.decode_f3d import DecodeF3D
        from core.atomic_components.wav2feat import Wav2Feat
        config_path=folder/'ditto_cfg/v0.4_hubert_cfg_pytorch.pkl'
        native=read_published_config(config_path)
        cpu_keys=['insightface_det_cfg','landmark106_cfg','landmark203_cfg','landmark478_cfg','hubert_cfg']
        overrides={'base_cfg':{key:{**native['base_cfg'][key],'device':'cpu'} for key in cpu_keys}}
        original=cfg.load_pkl;cfg.load_pkl=read_published_config
        try:parts=cfg.parse_cfg(str(config_path),str(folder/'ditto_pytorch'),overrides)
        finally:cfg.load_pkl=original
        self.registrar=AvatarRegistrar(**parts[0]);self.condition_type=parts[1]
        self.audio2motion=Audio2Motion(parts[2]);self.stitch=MotionStitch(parts[3])
        self.warp=WarpF3D(parts[4]);self.decoder=DecodeF3D(parts[5]);self.wav2feat=Wav2Feat(**parts[6])
        self.defaults=parts[7]
        info=self.registrar.source2info
        self.networks=[info.appearance_extractor.model,info.motion_extractor.model,self.audio2motion.lmdm.model,
                       self.stitch.stitch_net.model,self.warp.warp_net.model,self.decoder.decoder.model]
        self.version=portrait_version(self.root)
        self.loaded=True

    def _activate(self):
        self._load()
        for network in self.networks:network.cuda().eval()

    def offload(self):
        for network in self.networks:network.cpu()
        gc.collect();torch.cuda.empty_cache()

    def prepare(self,photo,folder,cancelled=None):
        photo=Path(photo);folder=Path(folder)
        if photo.suffix.lower() not in {'.jpg','.jpeg','.png'}:raise ValueError('照片动画只接受图片')
        with gpu_operation(self.root,cancelled),torch.inference_mode():
            try:
                self._activate();expected={**self.version,'photo_sha256':sha256(photo)}
                try:source=load_cache(folder,expected)
                except (FileNotFoundError,ValueError,KeyError):
                    image=cv2.cvtColor(cv2.imread(str(photo)),cv2.COLOR_BGR2RGB)
                    detections,_=self.registrar.source2info.insightface_det(image)
                    areas=sorted(((box[2]-box[0])*(box[3]-box[1]) for box in detections),reverse=True)
                    if not areas or len(areas)>1 and areas[0]<4*areas[1]:raise ValueError('请使用只有一张清晰人脸的照片')
                    source=self.registrar(str(photo),max_dim=1024)
                    save_cache(folder,source,expected)
                self._write_idle(source,folder/'idle.mkv',cancelled)
                atomic_json(folder/'ready.json',{'metadata':expected,'idle_sha256':sha256(folder/'idle.mkv'),
                    'width':source['img_rgb_lst'][0].shape[1],'height':source['img_rgb_lst'][0].shape[0],
                    'video_format':'h264_rgb_lossless_v1'})
                return read_json(folder/'ready.json')
            finally:self.offload()

    def _frames(self, source, motions, path, options, cancelled=None, idle=False):
        from core.utils.get_mask import get_mask
        count=len(motions);photo=source['img_rgb_lst'][0];height,width=photo.shape[:2]
        self.stitch.setup(N_d=count,use_d_keys=options['use_d_keys'],relative_d=True,drive_eye=True,
            delta_eye_arr=self.defaults.get('delta_eye_arr'),delta_eye_open_n=55,
            fade_type='s',flag_stitching=True,is_image_flag=True,x_s_info=source['x_s_info_lst'][0],ch_info=None)
        mask=get_mask(512,512,.9,.9)[...,0]
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_name('.'+uuid.uuid4().hex+'.mkv')
        try:
            with av.open(str(temporary),'w') as output:
                stream=output.add_stream('libx264rgb',rate=25,
                    options={'crf':'0','preset':'ultrafast','tune':'zerolatency'})
                stream.width=width;stream.height=height;stream.pix_fmt='rgb24';stream.thread_count=2
                for index,motion in enumerate(motions):
                    if cancelled and cancelled():raise InterruptedError('当前回答已停止')
                    control={}
                    if idle:
                        control={'delta_yaw':.5*math.sin(index/count*2*math.pi),
                                 'delta_pitch':.3*math.sin(index/count*2*math.pi)}
                    else:control['fade_alpha']=min(1.,index/2,(count-1-index)/3)
                    x_s,x_d=self.stitch(source['x_s_info_lst'][0],motion,**control)
                    warped=self.warp(source['f_s_lst'][0],x_s,x_d)
                    face=self.decoder(warped)
                    image=pasteback(photo,face,source['M_c2o_lst'][0],mask)
                    frame=av.VideoFrame.from_ndarray(image,format='rgb24')
                    for packet in stream.encode(frame):output.mux(packet)
                for packet in stream.encode():output.mux(packet)
            os.replace(temporary,path)
        finally:temporary.unlink(missing_ok=True)

    def _write_idle(self, source, path, cancelled):
        source=copy.deepcopy(source)
        self._frames(source,[copy.deepcopy(source['x_s_info_lst'][0]) for _ in range(100)],path,
                     photo_options(self.defaults,'neutral'),cancelled,idle=True)

    def render(self, cache_folder, audio, output, emotion='neutral', cancelled=None):
        started=clock.monotonic()
        with gpu_operation(self.root,cancelled),torch.inference_mode():
            try:
                self._activate();torch.cuda.reset_peak_memory_stats()
                metadata=read_json(Path(cache_folder)/'source.json')['metadata']
                source=load_cache(Path(cache_folder),{**self.version,'photo_sha256':metadata['photo_sha256']})
                options=photo_options(self.defaults,emotion)
                from core.atomic_components.condition_handler import ConditionHandler
                condition=ConditionHandler(**self.condition_type)
                condition.setup(source,options['emo'],ch_info=None)
                self.audio2motion.setup(condition.x_s_info_0,overlap_v2=10,
                    fix_kp_cond=options.get('fix_kp_cond',1),fix_kp_cond_dim=options.get('fix_kp_cond_dim'),
                    sampling_timesteps=options.get('sampling_timesteps',50),online_mode=False,
                    v_min_max_for_clip=options.get('v_min_max_for_clip'),smo_k_d=3)
                pcm,rate=read_audio(audio,16000);count=math.ceil(len(pcm)/rate*25)
                features=self.wav2feat.wav2feat(pcm,sr=rate)
                if not len(features):raise ValueError('声音没有产生有效动作特征')
                sequence=None
                for index in range(0,len(features),self.audio2motion.valid_clip_len):
                    if cancelled and cancelled():raise InterruptedError('当前回答已停止')
                    values=features[index:index+self.audio2motion.seq_frames]
                    if len(values)<self.audio2motion.seq_frames:
                        values=np.concatenate([values,np.repeat(values[-1:],self.audio2motion.seq_frames-len(values),axis=0)])
                    sequence=self.audio2motion(condition(values,index),sequence)
                sequence=sequence[:,:count]
                if sequence.shape[1]<count:sequence=np.concatenate([sequence,np.repeat(sequence[:,-1:],count-sequence.shape[1],axis=1)],axis=1)
                sequence=self.audio2motion._smo(sequence,0,count)
                motions=self.audio2motion.cvt_fmt(sequence)
                self._frames(source,motions,output,options,cancelled)
                report={'frames':count,'fps':25,'emotion':emotion,'seconds':len(pcm)/rate,
                    'generation_seconds':clock.monotonic()-started,'peak_vram_mib':torch.cuda.max_memory_allocated()/1024**2,
                    'photo_sha256':metadata['photo_sha256']}
                return report
            finally:self.offload()
