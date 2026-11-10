import pickle
import numpy as np


def test_published_numpy2_config_loads_without_upgrading_other_models(tmp_path):
    from backend.portrait.assets import read_published_config
    value={'default_kwargs':{'eye_delta':np.arange(6,dtype=np.float32).reshape(2,3)}}
    payload=pickle.dumps(value,protocol=0).replace(b'numpy.core',b'numpy._core')
    path=tmp_path/'config.pkl';path.write_bytes(payload)
    restored=read_published_config(path)
    np.testing.assert_array_equal(restored['default_kwargs']['eye_delta'],value['default_kwargs']['eye_delta'])


def test_photo_animation_overrides_the_example_identity_and_emotion():
    from backend.portrait.engine import photo_options
    defaults={'ch_info':{'example_identity':True},'emo':np.zeros((600,8)),'max_size':1920}
    options=photo_options(defaults,'sad')
    assert options['ch_info'] is None
    assert options['emo'] == 5 and options['max_size'] == 1024
    assert defaults['ch_info'] == {'example_identity':True}


def test_photo_cache_rejects_changed_source(tmp_path):
    from backend.portrait.cache import save_cache,load_cache
    source={'x_s_info_lst':[{'exp':np.zeros((1,63),np.float32)}],
        'f_s_lst':[np.ones((1,2,2),np.float32)],'M_c2o_lst':[np.eye(3,dtype=np.float32)],
        'eye_open_lst':[np.ones((1,2),np.float32)],'eye_ball_lst':[np.zeros((1,6),np.float32)],
        'sc':np.ones(63,np.float32),'img_rgb_lst':[np.zeros((16,16,3),np.uint8)],'is_image_flag':True}
    save_cache(tmp_path,source,{'photo_sha256':'photo1','model_revision':'model1'})
    restored=load_cache(tmp_path,{'photo_sha256':'photo1','model_revision':'model1'})
    np.testing.assert_array_equal(restored['f_s_lst'][0],source['f_s_lst'][0])
    import pytest
    with pytest.raises(ValueError,match='照片'):
        load_cache(tmp_path,{'photo_sha256':'photo2','model_revision':'model1'})


def test_face_pasteback_preserves_the_photo_outside_the_face():
    from backend.portrait.engine import pasteback
    photo=np.full((8,8,3),200,np.uint8);face=np.full((4,4,3),50,np.uint8)
    matrix=np.array([[1,0,2],[0,1,2],[0,0,1]],np.float32)
    result=pasteback(photo,face,matrix,np.ones((4,4),np.float32))
    np.testing.assert_array_equal(result[0,0],photo[0,0])
    np.testing.assert_array_equal(result[3,3],face[1,1])
def test_idle_rebuild_overwrites_unverified_existing_video(tmp_path):
    from backend.portrait.engine import PhotoEngine
    engine=PhotoEngine(tmp_path)
    engine.defaults={}
    target=tmp_path/'idle.mkv'
    target.write_bytes(b'broken previous render')
    calls=[]
    def render(source,motions,path,options,cancelled,*,idle,batch=1,stride=1):
        calls.append((len(motions),idle))
        path.write_bytes(b'rebuilt idle')
    engine._frames=render
    engine._write_idle({'x_s_info_lst':[{}]},target,None)
    assert calls == [(100,True)]
    assert target.read_bytes() == b'rebuilt idle'


def test_rendered_photo_video_preserves_pixels_and_compresses_repeated_background(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    import av
    from backend.portrait.engine import PhotoEngine

    # Replace native inference only. Use the real compositor, video writer,
    # codec and decoder to check lossless pixels and inter-frame compression.
    monkeypatch.setitem(sys.modules, 'core.utils.get_mask',
        SimpleNamespace(get_mask=lambda *args: np.zeros((512, 512, 3), np.float32)))

    class Stitch:
        def setup(self, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return None, None

    engine = PhotoEngine(tmp_path)
    engine.defaults = {}
    engine.stitch = Stitch()
    engine.warp = lambda *args: np.zeros((512, 512, 3), np.uint8)
    engine.decoder = lambda face: face
    photo = np.random.default_rng(7).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    source = {'img_rgb_lst': [photo], 'x_s_info_lst': [{}],
              'f_s_lst': [None], 'M_c2o_lst': [np.eye(3, dtype=np.float32)]}
    target = tmp_path/'photo.mkv'
    engine._frames(source, [{} for _ in range(40)], target, {'use_d_keys': {}})
    with av.open(str(target)) as video:
        frames = list(video.decode(video=0))
        assert len(frames) == 40
        for frame in frames:
            np.testing.assert_array_equal(frame.to_ndarray(format='rgb24'), photo)
    assert target.stat().st_size < photo.nbytes*4, 'Repeated photo pixels must not be stored independently per frame'


def test_expression_only_shapes_neutral_turns():
    from backend.portrait.engine import photo_options
    defaults = {'ch_info': None, 'max_size': 1920}
    assert photo_options(defaults, 'neutral')['emo'] == 4
    assert photo_options(defaults, 'neutral')['use_d_keys']['exp'] == 1.
    assert photo_options(defaults, 'neutral', expression='smile')['emo'] == (3, 4)
    serious = photo_options(defaults, 'neutral', expression='serious')
    assert serious['emo'].shape == (1, 8) and serious['use_d_keys']['exp'] == .75
    surprise = photo_options(defaults, 'neutral', expression='surprise')
    assert surprise['emo'] == 6 and surprise['use_d_keys']['exp'] == .85
    emotional = photo_options(defaults, 'happy', expression='serious')
    assert emotional['emo'] == 3 and emotional['use_d_keys']['exp'] == 1.
    assert emotional['use_d_keys']['pitch'] == .25
    assert defaults == {'ch_info': None, 'max_size': 1920}
