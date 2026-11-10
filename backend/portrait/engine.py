"""Thin photo-only adapter around the pinned Ditto inference components."""
import copy
import gc
import math
import os
import sys
import time
from backend import clock
import uuid
from pathlib import Path
import av
import cv2
import numpy as np
import torch
from backend.config import load_config
from backend.emotion import normalize_emotion,portrait_emotion
from backend.style import expression_emo,expression_exp_gain
from backend.gpu import gpu_operation,session_operation,check_stream_headroom
from backend.errors import StreamMemoryUnavailable
from backend.media import read_audio
from backend.storage import read_json,sha256,atomic_json
from backend.portrait.assets import portrait_version,read_published_config,verified_model
from backend.portrait.cache import save_cache,load_cache
from backend.stage_timing import StageTiming


_RUNTIME_CACHE={}


def runtime_quality(root):
    """Read the speed knobs from configs/app.yaml, cached by file mtime."""
    path=Path(root)/'configs/app.yaml'
    try:stamp=path.stat().st_mtime_ns
    except OSError:return {'sampling_steps':None,'frame_batch':1}
    cached=_RUNTIME_CACHE.get(str(path))
    if cached and cached[0]==stamp:return cached[1]
    try:runtime=load_config(Path(root)).get('runtime') or {}
    except Exception:runtime={}
    steps=runtime.get('portrait_sampling_steps')
    batch=runtime.get('portrait_frame_batch') or 1
    stride=runtime.get('portrait_frame_stride') or 1
    quality={'sampling_steps':int(steps) if isinstance(steps,int) and steps>0 else None,
             'frame_batch':max(1,int(batch)) if isinstance(batch,int) else 1,
             'frame_stride':max(1,int(stride)) if isinstance(stride,int) else 1}
    _RUNTIME_CACHE[str(path)]=(stamp,quality)
    return quality


_SEED_CACHE={}


def runtime_seed(root):
    """Diagnostic only: a fixed seed helps A/B on one machine, not across GPUs."""
    path=Path(root)/'configs/app.yaml'
    try:stamp=path.stat().st_mtime_ns
    except OSError:return None
    cached=_SEED_CACHE.get(str(path))
    if cached and cached[0]==stamp:return cached[1]
    try:value=(load_config(Path(root)).get('runtime') or {}).get('portrait_seed')
    except Exception:value=None
    seed=int(value) if isinstance(value,int) else None
    _SEED_CACHE[str(path)]=(stamp,seed)
    return seed


def apply_seed(seed):
    if seed is not None:torch.manual_seed(seed)


def photo_options(defaults, emotion, quality=None, *, expression='natural'):
    """Build render options; a real emotion always beats the preset expression.

    The preset only shapes neutral turns: it picks the Ditto emotion condition
    and the expression gain of the driving signal. Emotional turns keep the
    label the dialogue model chose and the full gain, so the face still reacts
    to what is being said instead of freezing on one look.
    """
    neutral=normalize_emotion(emotion)=='neutral'
    options={**defaults,'max_size':1024,'ch_info':None,
        'emo':expression_emo(expression) if neutral else portrait_emotion(emotion),
        'use_d_keys':{'exp':expression_exp_gain(expression) if neutral else 1.,
                      'pitch':.25,'yaw':.25,'roll':.25,'t':0.},
        'fade_type':'s','overall_ctrl_info':{},'online_mode':False}
    steps=(quality or {}).get('sampling_steps')
    if steps:options['sampling_timesteps']=int(steps)
    return options


def pasteback(photo, face, matrix, mask):
    height,width=photo.shape[:2]
    alpha=cv2.warpAffine(mask,matrix[:2],(width,height),flags=cv2.INTER_LINEAR).clip(0,1)[...,None]
    image=cv2.warpAffine(face,matrix[:2],(width,height),flags=cv2.INTER_LINEAR)
    return (image*alpha+photo*(1-alpha)).clip(0,255).astype(np.uint8)


class PhotoCompositor:
    """Fixed-mask pasteback with preallocated float32 buffers.

    The mask and the background never change for one photo, so only the
    warped face and the two in-place fused passes run per frame.
    """
    def __init__(self,photo,matrix,mask):
        height,width=photo.shape[:2];self.size=(width,height);self.matrix=matrix[:2]
        alpha=cv2.warpAffine(mask,self.matrix,self.size,flags=cv2.INTER_LINEAR)
        if alpha.ndim==2:alpha=alpha[...,None]
        self.alpha=alpha.astype(np.float32,copy=False)
        np.clip(self.alpha,0.,1.,out=self.alpha)
        self.background=photo.astype(np.float32)
        self.background*=(1.-self.alpha)
        self.buffer=np.empty(self.background.shape,np.float32)

    def __call__(self,face):
        image=cv2.warpAffine(face,self.matrix,self.size,flags=cv2.INTER_LINEAR)
        buffer=self.buffer
        np.copyto(buffer,image,casting='unsafe')
        np.multiply(buffer,self.alpha,out=buffer)
        np.add(buffer,self.background,out=buffer)
        np.clip(buffer,0.,255.,out=buffer)
        return buffer.astype(np.uint8)


def write_video_block(images,path,width,height,timing):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name('.'+uuid.uuid4().hex+'.mkv')
    try:
        with av.open(str(temporary),'w') as output:
            stream=output.add_stream('libx264rgb',rate=25,
                options={'crf':'0','preset':'ultrafast','tune':'zerolatency'})
            stream.width=width;stream.height=height;stream.pix_fmt='rgb24';stream.thread_count=2
            for image in images:
                with timing.measure('encode_ms'):
                    frame=av.VideoFrame.from_ndarray(image,format='rgb24')
                    for packet in stream.encode(frame):output.mux(packet)
            with timing.measure('encode_ms'):
                for packet in stream.encode():output.mux(packet)
        os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)


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
        try:
            self.registrar=AvatarRegistrar(**parts[0]);self.condition_type=parts[1]
            info=self.registrar.source2info
            self.networks.extend([info.appearance_extractor.model,info.motion_extractor.model])
            self.audio2motion=Audio2Motion(parts[2]);self.networks.append(self.audio2motion.lmdm.model)
            self.stitch=MotionStitch(parts[3]);self.networks.append(self.stitch.stitch_net.model)
            self.warp=WarpF3D(parts[4]);self.networks.append(self.warp.warp_net.model)
            self.decoder=DecodeF3D(parts[5]);self.networks.append(self.decoder.decoder.model)
            self.wav2feat=Wav2Feat(**parts[6]);self.defaults=parts[7]
            self.version=portrait_version(self.root)
            self.loaded=True
        except BaseException:
            try:self.offload()
            finally:
                for name in ('registrar','condition_type','audio2motion','stitch','warp','decoder','wav2feat','defaults','version'):
                    self.__dict__.pop(name,None)
                self.networks.clear();self.loaded=False
            raise

    def _activate(self):
        self._load()
        for network in self.networks:network.cuda().eval()

    def warmup(self):
        self._activate();self.offload()

    def offload(self):
        for network in self.networks:network.cpu()
        gc.collect();torch.cuda.empty_cache()

    def _face_renderer(self,source):
        warp=getattr(self.warp,'warp_net',None);decoder=getattr(self.decoder,'decoder',None)
        if (getattr(warp,'model_type',None)!='pytorch' or getattr(decoder,'model_type',None)!='pytorch'):
            def single(xs,xd):return self.decoder(self.warp(source['f_s_lst'][0],xs,xd))
            single.many=lambda pairs:[single(*pair) for pair in pairs]
            return single
        feature=torch.from_numpy(source['f_s_lst'][0]).to(warp.device)
        def draw(xs,xd):
            with torch.autocast(device_type=warp.device[:4],dtype=torch.float16,enabled=True):
                warped=warp.model(feature,torch.from_numpy(xs).to(warp.device),
                                  torch.from_numpy(xd).to(warp.device)).float()
            with torch.autocast(device_type=decoder.device[:4],dtype=torch.float16,enabled=True):
                result=decoder.model(warped).float().cpu().numpy()
            return np.transpose(result[0],[1,2,0]).clip(0,1)*255
        def many(pairs):
            """Batched warping/decoding; silently falls back to one frame at a time."""
            if len(pairs)<2:return [draw(*pairs[0])]
            try:
                xs=torch.from_numpy(np.concatenate([pair[0] for pair in pairs],0)).to(warp.device)
                xd=torch.from_numpy(np.concatenate([pair[1] for pair in pairs],0)).to(warp.device)
                count=xs.shape[0]
                batch=feature.repeat(count,*([1]*(feature.dim()-1))) if feature.shape[0]==1 else feature
                if batch.shape[0]!=count:raise ValueError('批处理特征数量不匹配')
                with torch.autocast(device_type=warp.device[:4],dtype=torch.float16,enabled=True):
                    warped=warp.model(batch,xs,xd).float()
                with torch.autocast(device_type=decoder.device[:4],dtype=torch.float16,enabled=True):
                    result=decoder.model(warped).float().cpu().numpy()
                if result.shape[0]!=count:raise ValueError('批处理输出数量不匹配')
                return [np.transpose(item,[1,2,0]).clip(0,1)*255 for item in result]
            except Exception:
                return [draw(*pair) for pair in pairs]
        draw.many=many
        return draw

    def _cached_source(self,folder):
        folder=Path(folder);record=read_json(folder/'source.json')
        path=(folder/record['file']).resolve()
        if folder.resolve() not in path.parents:raise ValueError('照片特征路径越界')
        stat=path.stat()
        key=(str(folder.resolve()),record['sha256'],stat.st_size,stat.st_mtime_ns,
             tuple(sorted(self.version.items())),tuple(sorted(record['metadata'].items())))
        if getattr(self,'_source_cache_key',None)!=key:
            source=load_cache(folder,{**self.version,'photo_sha256':record['metadata']['photo_sha256']})
            self._source_cache=source;self._source_cache_key=key
        # Stitching may mutate source motion arrays. Cached identity features
        # remain immutable across sentences and are validated again on change.
        return copy.deepcopy(self._source_cache)

    def _stream_motions(self,folder,condition,timing,cancelled):
        from backend.portrait.stream import FeatureStream,MotionStream
        from backend.stream_jobs import AudioSpool
        from backend.stream_audio import StreamingResampler
        spool=AudioSpool(folder);state=spool.state();rate=state['rate']
        resampler=StreamingResampler(rate,16000);features=FeatureStream(self.wav2feat)
        motions=MotionStream(self.audio2motion,condition);index=0;samples=0
        deadline=clock.monotonic()+600
        while True:
            if cancelled and cancelled():raise InterruptedError('当前回答已停止')
            if clock.monotonic()>deadline:raise TimeoutError('等待声音分块超时')
            state=spool.state()
            if state['error']:raise RuntimeError(state['error'])
            if state['rate']!=rate:raise ValueError('声音流采样率发生变化')
            if index<state['chunks']:
                event=read_json(Path(folder)/f'audio-{index}.json')
                if event['sample_start']!=samples:raise ValueError('声音流采样范围不连续')
                pcm=spool.read(event);samples+=len(pcm);index+=1
                with timing.measure('audio_features_ms'):values=features.push(resampler.push(pcm))
                with timing.measure('motion_ms'):ready=motions.push(values)
                yield from ready
            elif state['finished']:break
            else:time.sleep(.02)
        with timing.measure('audio_features_ms'):
            values=np.concatenate([features.push(resampler.finish()),features.finish()])
        with timing.measure('motion_ms'):ready=motions.push(values)+motions.finish()
        yield from ready
        self._stream_audio_samples=samples;self._stream_audio_rate=rate

    def stream_render(self,cache_folder,folder,emotion='neutral',cancelled=None,*,session_id,token,
                      expression='natural'):
        from collections import deque
        from backend.stream_jobs import video_chunk_path,check_space
        timing=StageTiming();started=clock.monotonic();waiting=started
        with session_operation(self.root,session_id,token,cancelled),torch.inference_mode():
            timing.values['gpu_wait_ms']=(clock.monotonic()-waiting)*1000
            try:
                apply_seed(runtime_seed(self.root))
                with timing.measure('load_ms'):self._load()
                from core.atomic_components.condition_handler import ConditionHandler
                from core.utils.get_mask import get_mask
                if cancelled and cancelled():raise InterruptedError('当前回答已停止')
                check_stream_headroom(self.root,session_id,token)
                with timing.measure('gpu_transfer_ms'):self._activate()
                if self.networks:torch.cuda.reset_peak_memory_stats()
                with timing.measure('photo_cache_ms'):source=self._cached_source(cache_folder)
                quality=runtime_quality(self.root);options=photo_options(self.defaults,emotion,quality,
                    expression=expression)
                condition=ConditionHandler(**self.condition_type);condition.setup(source,options['emo'],ch_info=None)
                # Preserve the pinned offline diffusion/fusion parameters and
                # publish only stable prefixes, as the upstream chunk pipeline
                # also supports with online_mode=False. Diffusion steps follow
                # runtime.portrait_sampling_steps; batching stays off here so a
                # published block never waits for a full batch.
                self.audio2motion.setup(condition.x_s_info_0,overlap_v2=10,
                    fix_kp_cond=options.get('fix_kp_cond',1),fix_kp_cond_dim=options.get('fix_kp_cond_dim'),
                    sampling_timesteps=options.get('sampling_timesteps',50),online_mode=False,
                    v_min_max_for_clip=options.get('v_min_max_for_clip'),smo_k_d=3)
                self.stitch.setup(N_d=-1,use_d_keys=options['use_d_keys'],relative_d=True,drive_eye=True,
                    delta_eye_arr=self.defaults.get('delta_eye_arr'),delta_eye_open_n=55,
                    fade_type='s',flag_stitching=True,is_image_flag=True,x_s_info=source['x_s_info_lst'][0],ch_info=None)
                photo=source['img_rgb_lst'][0];height,width=photo.shape[:2]
                compose=PhotoCompositor(photo,source['M_c2o_lst'][0],get_mask(512,512,.9,.9)[...,0])
                draw=self._face_renderer(source);tail=deque();images=[];frame_index=0;chunk_index=0
                def image(motion,total=None):
                    if cancelled and cancelled():raise InterruptedError('当前回答已停止')
                    if frame_index%25==0:check_stream_headroom(self.root,session_id,token)
                    fade=min(1.,frame_index/2)
                    if total is not None:fade=min(fade,(total-1-frame_index)/3)
                    with timing.measure('frame_gpu_ms'):
                        xs,xd=self.stitch(source['x_s_info_lst'][0],motion,fade_alpha=fade)
                        face=draw(xs,xd)
                    with timing.measure('composite_ms'):return compose(face)
                for motion in self._stream_motions(folder,condition,timing,cancelled):
                    tail.append(motion)
                    if len(tail)<=3:continue
                    images.append(image(tail.popleft()));frame_index+=1
                    if len(images)==25:
                        check_space(self.root,folder)
                        write_video_block(images,video_chunk_path(folder,chunk_index),width,height,timing)
                        yield dict(type='chunk',chunk_index=chunk_index,frame_start=frame_index-25,frames=25)
                        images=[];chunk_index+=1
                # Keep the native online blink plan. Rebuilding it for the
                # final length would change an already-started blink.
                total=frame_index+len(tail)
                for motion in tail:
                    images.append(image(motion,total));frame_index+=1
                if images:
                    check_space(self.root,folder)
                    # A full final block and its remaining tail can coexist.
                    for offset in range(0,len(images),25):
                        block=images[offset:offset+25]
                        write_video_block(block,video_chunk_path(folder,chunk_index),width,height,timing)
                        yield dict(type='chunk',chunk_index=chunk_index,
                            frame_start=total-len(images)+offset,frames=len(block))
                        chunk_index+=1
                expected=(self._stream_audio_samples*25+self._stream_audio_rate-1)//self._stream_audio_rate
                if frame_index!=expected:raise ValueError(f'动作流末尾帧数不匹配：实际{frame_index}，应为{expected}')
                self.last_stream_report=dict(frames=frame_index,fps=25,
                    seconds=self._stream_audio_samples/self._stream_audio_rate,
                    generation_seconds=clock.monotonic()-started,timings=timing.values,timing_spans=timing.spans,
                    peak_vram_mib=torch.cuda.max_memory_allocated()/1024**2 if self.networks else 0.)
            except torch.cuda.OutOfMemoryError as error:
                raise StreamMemoryUnavailable('画面生成时显存不足') from error
            finally:
                with timing.measure('offload_ms'):self.offload()

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

    def _frames(self, source, motions, path, options, cancelled=None, idle=False, *, timing=None, batch=1, stride=1):
        from core.utils.get_mask import get_mask
        count=len(motions);photo=source['img_rgb_lst'][0];height,width=photo.shape[:2]
        self.stitch.setup(N_d=count,use_d_keys=options['use_d_keys'],relative_d=True,drive_eye=True,
            delta_eye_arr=self.defaults.get('delta_eye_arr'),delta_eye_open_n=55,
            fade_type='s',flag_stitching=True,is_image_flag=True,x_s_info=source['x_s_info_lst'][0],ch_info=None)
        mask=get_mask(512,512,.9,.9)[...,0]
        compose=PhotoCompositor(photo,source['M_c2o_lst'][0],mask)
        draw=self._face_renderer(source)
        timing=timing or StageTiming()
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_name('.'+uuid.uuid4().hex+'.mkv')
        size=max(1,int(batch or 1));step=max(1,int(stride or 1));pending=[];written=0
        try:
            with av.open(str(temporary),'w') as output:
                stream=output.add_stream('libx264rgb',rate=25,
                    options={'crf':'0','preset':'ultrafast','tune':'zerolatency'})
                stream.width=width;stream.height=height;stream.pix_fmt='rgb24';stream.thread_count=2
                def flush():
                    nonlocal written
                    if not pending:return
                    queued=list(pending);pending.clear()
                    with timing.measure('frame_gpu_ms'):
                        faces=draw.many(queued) if len(queued)>1 else [draw(*queued[0])]
                    for face in faces:
                        if written>=count:break
                        copies=min(step,count-written);written+=copies
                        with timing.measure('composite_ms'):image=compose(face)
                        with timing.measure('encode_ms'):
                            for _ in range(copies):
                                frame=av.VideoFrame.from_ndarray(image,format='rgb24')
                                for packet in stream.encode(frame):output.mux(packet)
                for index,motion in enumerate(motions):
                    if cancelled and cancelled():raise InterruptedError('当前回答已停止')
                    control={}
                    if idle:
                        control={'delta_yaw':.5*math.sin(index/count*2*math.pi),
                                 'delta_pitch':.3*math.sin(index/count*2*math.pi)}
                    else:control['fade_alpha']=min(1.,index/2,(count-1-index)/3)
                    with timing.measure('stitch_ms'):
                        x_s,x_d=self.stitch(source['x_s_info_lst'][0],motion,**control)
                    # 运动与眨眼状态逐帧推进；只有出图按 stride 抽样，抽到的帧重复写出补满 25 fps。
                    if index%step:continue
                    pending.append((x_s,x_d))
                    if len(pending)>=size:flush()
                flush()
                with timing.measure('encode_ms'):
                    for packet in stream.encode():output.mux(packet)
            os.replace(temporary,path)
        finally:temporary.unlink(missing_ok=True)

    def _write_idle(self, source, path, cancelled):
        source=copy.deepcopy(source)
        self._frames(source,[copy.deepcopy(source['x_s_info_lst'][0]) for _ in range(100)],path,
                     photo_options(self.defaults,'neutral',runtime_quality(self.root)),cancelled,idle=True,
                     batch=runtime_quality(self.root)['frame_batch'])

    def render(self, cache_folder, audio, output, emotion='neutral', cancelled=None, *, expression='natural'):
        started=clock.monotonic();timing=StageTiming();waiting=clock.monotonic()
        with gpu_operation(self.root,cancelled),torch.inference_mode():
            timing.values['gpu_wait_ms']=(clock.monotonic()-waiting)*1000
            try:
                apply_seed(runtime_seed(self.root))
                with timing.measure('load_ms'):self._load()
                with timing.measure('gpu_transfer_ms'):self._activate()
                torch.cuda.reset_peak_memory_stats()
                metadata=read_json(Path(cache_folder)/'source.json')['metadata']
                with timing.measure('photo_cache_ms'):
                    source=load_cache(Path(cache_folder),{**self.version,'photo_sha256':metadata['photo_sha256']})
                quality=runtime_quality(self.root);options=photo_options(self.defaults,emotion,quality,
                    expression=expression)
                from core.atomic_components.condition_handler import ConditionHandler
                condition=ConditionHandler(**self.condition_type)
                condition.setup(source,options['emo'],ch_info=None)
                self.audio2motion.setup(condition.x_s_info_0,overlap_v2=10,
                    fix_kp_cond=options.get('fix_kp_cond',1),fix_kp_cond_dim=options.get('fix_kp_cond_dim'),
                    sampling_timesteps=options.get('sampling_timesteps',50),online_mode=False,
                    v_min_max_for_clip=options.get('v_min_max_for_clip'),smo_k_d=3)
                pcm,rate=read_audio(audio,16000);count=(len(pcm)*25+rate-1)//rate
                with timing.measure('audio_features_ms'):features=self.wav2feat.wav2feat(pcm,sr=rate)
                if not len(features):raise ValueError('声音没有产生有效动作特征')
                sequence=None
                for index in range(0,len(features),self.audio2motion.valid_clip_len):
                    if cancelled and cancelled():raise InterruptedError('当前回答已停止')
                    values=features[index:index+self.audio2motion.seq_frames]
                    if len(values)<self.audio2motion.seq_frames:
                        values=np.concatenate([values,np.repeat(values[-1:],self.audio2motion.seq_frames-len(values),axis=0)])
                    with timing.measure('motion_ms'):sequence=self.audio2motion(condition(values,index),sequence)
                sequence=sequence[:,:count]
                if sequence.shape[1]<count:sequence=np.concatenate([sequence,np.repeat(sequence[:,-1:],count-sequence.shape[1],axis=1)],axis=1)
                sequence=self.audio2motion._smo(sequence,0,count)
                motions=self.audio2motion.cvt_fmt(sequence)
                self._frames(source,motions,output,options,cancelled,timing=timing,
                             batch=quality['frame_batch'],stride=quality['frame_stride'])
                report={'frames':count,'fps':25,'emotion':emotion,'seconds':len(pcm)/rate,
                    'generation_seconds':clock.monotonic()-started,'peak_vram_mib':torch.cuda.max_memory_allocated()/1024**2,
                    'photo_sha256':metadata['photo_sha256'],'timings':timing.values}
                return report
            finally:
                with timing.measure('offload_ms'):self.offload()
