from types import SimpleNamespace

import numpy as np
import pytest


def test_cached_compositor_matches_original_pixels():
    from backend.portrait.engine import PhotoCompositor,pasteback
    rng=np.random.default_rng(11)
    photo=rng.integers(0,256,(67,73,3),dtype=np.uint8)
    matrix=np.array([[1.13,.02,4],[-.02,1.01,3],[0,0,1]],np.float32)
    mask=rng.random((32,32),dtype=np.float32)
    compose=PhotoCompositor(photo,matrix,mask)
    for _ in range(3):
        face=rng.random((32,32,3),dtype=np.float32)*255
        np.testing.assert_array_equal(compose(face),pasteback(photo,face,matrix,mask))


def test_torch_face_path_keeps_features_between_warp_and_decoder(tmp_path):
    import torch
    from backend.portrait.engine import PhotoEngine
    pointers=[]
    class Warp(torch.nn.Module):
        def forward(self,feature,xs,xd):
            pointers.append(feature.data_ptr())
            return feature+xs.mean()-xd.mean()
    class Decode(torch.nn.Module):
        def forward(self,feature):return feature*.5
    engine=PhotoEngine(tmp_path)
    engine.warp=SimpleNamespace(warp_net=SimpleNamespace(model_type='pytorch',device='cpu',model=Warp()))
    engine.decoder=SimpleNamespace(decoder=SimpleNamespace(model_type='pytorch',device='cpu',model=Decode()))
    source={'f_s_lst':[np.full((1,3,4,4),.4,np.float32)]}
    draw=engine._face_renderer(source)
    for _ in range(2):
        actual=draw(np.ones((1,21,3),np.float32),np.ones((1,21,3),np.float32))
        expected=np.transpose(((source['f_s_lst'][0]+1-1)*.5)[0],[1,2,0])*255
        np.testing.assert_array_equal(actual,expected)
    assert pointers[0]==pointers[1]


@pytest.mark.parametrize('cold',[False,True])
@pytest.mark.parametrize('samples,blocks,total',[
    (100800,[25,25,25,25,5],105),
    (107520,[25,25,25,25,12],112),
])
def test_portrait_emits_sealed_video_while_audio_is_still_open(tmp_path,monkeypatch,cold,samples,blocks,total):
    import sys
    import av
    from backend.portrait.engine import PhotoEngine
    from backend.stream_jobs import AudioSpool
    from backend.gpu import SessionGpuLease
    from backend.storage import atomic_json

    class Condition:
        def __init__(self,**kwargs):pass
        def setup(self,*args,**kwargs):self.x_s_info_0={}
        def __call__(self,values,index):return values
    class Motion:
        def setup(self,*args,**kwargs):self.seq_frames=80;self.valid_clip_len=70;self.overlap_v2=10
        def __call__(self,values,sequence):
            block=np.zeros((1,80,265),np.float32)
            return block if sequence is None else np.concatenate([sequence[:,:-10],block],axis=1)
        def _smo(self,values,start,end):return values.copy()
        def cvt_fmt(self,values):return [{} for _ in range(values.shape[1])]
    class Stitch:
        def setup(self,**kwargs):pass
        def set_Nd(self,count):pass
        def __call__(self,*args,**kwargs):return None,None
    monkeypatch.setitem(sys.modules,'core.atomic_components.condition_handler',SimpleNamespace(ConditionHandler=Condition))
    monkeypatch.setitem(sys.modules,'core.utils.get_mask',SimpleNamespace(get_mask=lambda *args:np.zeros((512,512,3),np.float32)))
    engine=PhotoEngine(tmp_path);engine._load=lambda:None;engine._activate=lambda:None
    if cold:
        monkeypatch.setitem(sys.modules,'core.atomic_components.condition_handler',None)
        monkeypatch.setitem(sys.modules,'core.utils.get_mask',None)
        def load():
            monkeypatch.setitem(sys.modules,'core.atomic_components.condition_handler',SimpleNamespace(ConditionHandler=Condition))
            monkeypatch.setitem(sys.modules,'core.utils.get_mask',SimpleNamespace(get_mask=lambda *args:np.zeros((512,512,3),np.float32)))
        engine._load=load
    engine.defaults={};engine.condition_type={};engine.audio2motion=Motion();engine.stitch=Stitch()
    engine.wav2feat=lambda window:np.ones((5,1024),np.float32)
    engine.warp=lambda *args:np.zeros((512,512,3),np.float32);engine.decoder=lambda value:value
    engine.offload=lambda:None
    photo=np.full((16,16,3),91,np.uint8)
    engine._cached_source=lambda folder:{'img_rgb_lst':[photo],'x_s_info_lst':[{}],
        'f_s_lst':[None],'M_c2o_lst':[np.eye(3,dtype=np.float32)]}
    folder=tmp_path/'spool';spool=AudioSpool(folder,24000)
    spool.append(np.full(samples,.1,np.float32))
    owner=SessionGpuLease(tmp_path,'session').acquire()
    try:
        stream=engine.stream_render(tmp_path/'cache',folder,session_id='session',token=owner.token)
        first=next(stream)
        assert first['frame_start']==0 and first['frames']==25 and not spool.state()['finished']
        spool.finish()
        chunks=[first,*stream]
        assert [event['frames'] for event in chunks]==blocks
        for event in chunks:
            with av.open(str(folder/f'video-{event["chunk_index"]}.mkv')) as video:
                frames=list(video.decode(video=0))
                assert len(frames)==event['frames']
                for frame in frames:np.testing.assert_array_equal(frame.to_ndarray(format='rgb24'),photo)
        assert not list(folder.glob('.*.mkv'))
        assert engine.last_stream_report['frames']==total
    finally:owner.release()


def test_partial_constructor_oom_offloads_completed_components(tmp_path,monkeypatch):
    import sys
    import torch
    from backend.portrait.engine import PhotoEngine
    networks=[]
    class Network:
        def __init__(self):self.on_gpu=True;networks.append(self)
        def cpu(self):self.on_gpu=False;return self
    registrar=SimpleNamespace(source2info=SimpleNamespace(
        appearance_extractor=SimpleNamespace(model=Network()),motion_extractor=SimpleNamespace(model=Network())))
    motion=SimpleNamespace(lmdm=SimpleNamespace(model=Network()))
    stitch=SimpleNamespace(stitch_net=SimpleNamespace(model=Network()))
    cfg=SimpleNamespace(load_pkl=lambda *args:None,parse_cfg=lambda *args:[{} for _ in range(8)])
    monkeypatch.setitem(sys.modules,'core.atomic_components',SimpleNamespace(cfg=cfg))
    constructors={'avatar_registrar':('AvatarRegistrar',lambda **kwargs:registrar),
        'audio2motion':('Audio2Motion',lambda *args:motion),'motion_stitch':('MotionStitch',lambda *args:stitch)}
    def oom(*args):raise torch.cuda.OutOfMemoryError('injected partial load')
    constructors.update(warp_f3d=('WarpF3D',oom),decode_f3d=('DecodeF3D',lambda *args:None),
                        wav2feat=('Wav2Feat',lambda **kwargs:None))
    for module,(name,constructor) in constructors.items():
        monkeypatch.setitem(sys.modules,'core.atomic_components.'+module,SimpleNamespace(**{name:constructor}))
    monkeypatch.setattr('backend.portrait.engine.verified_model',lambda *args:tmp_path)
    keys=['insightface_det_cfg','landmark106_cfg','landmark203_cfg','landmark478_cfg','hubert_cfg']
    monkeypatch.setattr('backend.portrait.engine.read_published_config',lambda *args:{'base_cfg':{key:{} for key in keys}})
    monkeypatch.setattr('backend.portrait.engine.portrait_version',lambda *args:{})
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    engine=PhotoEngine(tmp_path)
    with pytest.raises(torch.cuda.OutOfMemoryError):engine._load()
    assert all(not network.on_gpu for network in networks)
    assert not engine.loaded and not engine.networks and not hasattr(engine,'registrar')


def test_stream_eof_keeps_the_existing_blink_phase(tmp_path,monkeypatch):
    import sys
    from backend.portrait.engine import PhotoEngine
    from backend.gpu import SessionGpuLease
    class Condition:
        def __init__(self,**kwargs):pass
        def setup(self,*args,**kwargs):self.x_s_info_0={}
    class Stitch:
        def setup(self,**kwargs):self.index=0;self.plan=[max(0,index-55) if index<70 else 0 for index in range(3000)]
        def set_Nd(self,total):self.plan=[0]*total
        def __call__(self,*args,**kwargs):
            seen.append(self.plan[self.index]);self.index+=1;return None,None
    monkeypatch.setitem(sys.modules,'core.atomic_components.condition_handler',SimpleNamespace(ConditionHandler=Condition))
    monkeypatch.setitem(sys.modules,'core.utils.get_mask',SimpleNamespace(get_mask=lambda *args:np.zeros((512,512,3),np.float32)))
    engine=PhotoEngine(tmp_path);engine._load=lambda:None;engine._activate=lambda:None;engine.offload=lambda:None
    engine.defaults={};engine.condition_type={};engine.audio2motion=SimpleNamespace(setup=lambda *args,**kwargs:None)
    engine.stitch=Stitch();seen=[]
    photo=np.zeros((16,16,3),np.uint8)
    engine._cached_source=lambda folder:{'img_rgb_lst':[photo],'x_s_info_lst':[{}],
        'f_s_lst':[None],'M_c2o_lst':[np.eye(3,dtype=np.float32)]}
    engine._face_renderer=lambda source:lambda *args:photo
    def motions(*args):
        engine._stream_audio_samples=65*640;engine._stream_audio_rate=16000
        yield from ({} for _ in range(65))
    engine._stream_motions=motions
    owner=SessionGpuLease(tmp_path,'session').acquire()
    try:list(engine.stream_render(tmp_path/'cache',tmp_path/'spool',session_id='session',token=owner.token))
    finally:owner.release()
    assert seen[55:]==list(range(10)),'EOF不能重建计划并截断正在进行的眨眼'
