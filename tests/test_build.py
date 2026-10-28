import pytest
from backend.storage import Store


def subject(tmp_path):
    video, body = tmp_path / 'video.mp4', tmp_path / 'body.png'
    video.write_bytes(b'original video')
    body.write_bytes(b'original body')
    store = Store(tmp_path)
    avatar = store.import_files('test', video, [body], [])
    job = store.new_job(avatar['id'], 'all')
    return store, avatar, job


def test_build_rejects_modified_source_before_loading_models(tmp_path):
    from backend.pipeline.build import build_avatar
    store, avatar, job = subject(tmp_path)
    (store.avatar_dir(avatar['id']) / 'source/video.mp4').write_bytes(b'changed')
    with pytest.raises(Exception, match='校验'):
        build_avatar(store, avatar['id'], job['id'], 'all')
    assert store.job(job['id'])['status'] == 'failed'
    assert store.manifest(avatar['id'])['status'] != 'ready'


def test_prepare_stops_before_gpu_and_never_marks_ready(tmp_path, monkeypatch):
    from backend.pipeline.build import build_avatar
    from backend.pipeline import media
    store, avatar, job = subject(tmp_path)
    def prepare(*_):
        from backend.storage import atomic_json
        atomic_json(store.avatar_dir(avatar['id'])/'processed/faces/frames.json',{'selected':[],'all':[]})
        return avatar
    monkeypatch.setattr(media, 'prepare_media', prepare)
    build_avatar(store, avatar['id'], job['id'], 'prepare')
    assert store.job(job['id'])['status'] == 'complete'
    assert store.manifest(avatar['id'])['status'] != 'ready'


def test_validate_requires_a_saved_reconstruction(tmp_path):
    from backend.pipeline.build import build_avatar
    store, avatar, job = subject(tmp_path)
    with pytest.raises(Exception, match='人物'):
        build_avatar(store, avatar['id'], job['id'], 'validate')
    assert store.job(job['id'])['status'] == 'failed'


@pytest.mark.parametrize('confirmed',[False,True])
def test_reconstruct_stage_finishes_before_voice_and_does_not_mark_ready(tmp_path,monkeypatch,confirmed):
    from backend.pipeline import build
    from backend.avatar.lhmpp import LHMRenderer
    from backend.storage import atomic_json
    store,avatar,job=subject(tmp_path)
    folder=store.avatar_dir(avatar['id'])
    atomic_json(folder/'processed/faces/frames.json',{'selected':[],'all':[]})
    manifest=store.manifest(avatar['id'])
    manifest['voice'].update(text='测试参考文本。',confirmed=confirmed)
    store.write_manifest(avatar['id'],manifest)
    atomic_json(folder/'build_signature.json',{'sources':manifest['sources'],'model_lock':None,'pipeline_version':build.PIPELINE_VERSION})
    atomic_json(folder/'state.json',{})
    monkeypatch.setattr(LHMRenderer,'load',lambda *_:object())
    monkeypatch.setattr(build,'validate_render',lambda *_:{'test_boundary':'cached renderer'})
    async def no_voice(*_): raise AssertionError('Reconstruction attempted voice inference')
    monkeypatch.setattr(build,'validate_voice',no_voice)
    build.build_avatar(store,avatar['id'],job['id'],'reconstruct')
    assert store.job(job['id'])['status']=='complete'
    assert store.job(job['id'])['stage']=='reconstructed'
    assert store.manifest(avatar['id'])['status']=='voice_pending'
    assert not store.manifest(avatar['id'])['validation']['voice']
