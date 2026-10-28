import pytest
from backend.storage import Store,atomic_json
from backend.pipeline import build
from backend.avatar import lhmpp,calibrate
from scripts import build_expression_map


@pytest.mark.parametrize('changed',['none','state','model'])
def test_face_retry_reuses_only_a_matching_body_checkpoint(tmp_path,monkeypatch,changed):
    video,body=tmp_path/'source.mp4',tmp_path/'body.png'
    video.write_bytes(b'video');body.write_bytes(b'body')
    store=Store(tmp_path);avatar=store.import_files('test',video,[body],[])
    folder=store.avatar_dir(avatar['id'])
    atomic_json(folder/'processed/faces/frames.json',{'selected':[],'all':[]})
    calls={'reconstruct':0,'calibrate':0}
    class Renderer:
        def save_preview(self,*args): pass
    def reconstruct(*args):
        calls['reconstruct']+=1
        atomic_json(folder/'state.json',{'body_attempt':calls['reconstruct']})
        return Renderer()
    def refine(*args):
        calls['calibrate']+=1
        if calls['calibrate']==1: raise RuntimeError('face refinement failed')
    monkeypatch.setattr(lhmpp,'reconstruct',reconstruct)
    monkeypatch.setattr(lhmpp.LHMRenderer,'load',lambda *_:Renderer())
    monkeypatch.setattr(calibrate,'calibrate_head',refine)
    monkeypatch.setattr(build_expression_map,'build_expression_map',lambda *_:None)
    monkeypatch.setattr(build,'validate_render',lambda *_:{'reload':'verified boundary'})
    job=store.new_job(avatar['id'],'reconstruct')
    with pytest.raises(RuntimeError,match='refinement'):
        build.build_avatar(store,avatar['id'],job['id'],'reconstruct')
    assert store.manifest(avatar['id'])['status']=='failed'
    assert not (folder/'build_signature.json').exists()
    if changed=='state': atomic_json(folder/'state.json',{'body_attempt':'modified'})
    if changed=='model': atomic_json(tmp_path/'models/lock.json',{'revision':'changed'})
    job=store.new_job(avatar['id'],'reconstruct')
    build.build_avatar(store,avatar['id'],job['id'],'reconstruct')
    assert calls['reconstruct']==(1 if changed=='none' else 2)
    assert calls['calibrate']==2
    assert store.job(job['id'])['stage']=='reconstructed'
    assert store.manifest(avatar['id'])['status']=='voice_pending'
