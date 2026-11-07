import json
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.gpu import SessionGpuLease
from backend.storage import Store,atomic_json,sha256


def fixture_profile(root):
    store=Store(root);folder=store.avatar_dir('person');folder.mkdir(parents=True)
    reference=folder/'reference.wav';reference.write_bytes(b'test reference')
    photo=folder/'photo.png';photo.write_bytes(b'test photo')
    manifest=dict(id='person',schema_version=2,mode='photo',status='ready',photo='photo.png',
        sources={'photo.png':sha256(photo)},portrait={'folder':'portrait/cache','width':16,'height':16},
        voice={'confirmed':True,'reference':'reference.wav','reference_sha256':sha256(reference)})
    store.write_manifest('person',manifest)
    cache=folder/'portrait/cache';cache.mkdir(parents=True)
    atomic_json(cache/'source.json',{'metadata':{'photo_sha256':sha256(photo)}})
    atomic_json(root/'data/sessions/session/generation.json',{'generation':0,'closed':False})
    return store


def test_voice_stream_checks_confirmation_and_publishes_verified_ranges(tmp_path,monkeypatch):
    from backend.voice import service
    from backend.stream_jobs import AudioSpool,stream_folder
    store=fixture_profile(tmp_path)
    def generate(*args,**kwargs):
        yield np.full(1200,.1,np.float32)
        yield np.full(1200,.2,np.float32)
    engine=SimpleNamespace(voice=SimpleNamespace(sample_rate=24000),stream_synthesize=generate,
        last_stream_report={'seconds':.1})
    monkeypatch.setattr(service,'ROOT',tmp_path);monkeypatch.setattr(service,'store',store)
    monkeypatch.setattr(service,'engine',engine)
    owner=SessionGpuLease(tmp_path,'session').acquire()
    body=dict(avatar_id='person',text='你好',session_id='session',generation=0,index=0,token=owner.token)
    try:
        with TestClient(service.app) as client:
            assert client.post('/tts/stream',json={**body,'token':'0'*64}).status_code==409
            response=client.post('/tts/stream',json=body)
            assert response.status_code==200
            events=[json.loads(line) for line in response.text.splitlines()]
            assert [event['type'] for event in events]==['start','chunk','chunk','complete']
            assert all(event['generation']==0 and event['sentence_index']==0 for event in events)
            spool=AudioSpool(stream_folder(tmp_path,'session',0,0))
            assert spool.state()['finished'] and spool.state()['samples']==2400
            assert [event['sample_start'] for event in events if event['type']=='chunk']==[0,1200]
            person=store.manifest('person');person['voice']['confirmed']=False;store.write_manifest('person',person)
            assert client.post('/tts/stream',json={**body,'index':1}).status_code==409
    finally:owner.release()


def test_portrait_stream_keeps_old_endpoint_and_generation_identity(tmp_path,monkeypatch):
    from backend.portrait import service
    store=fixture_profile(tmp_path)
    def render(*args,**kwargs):yield dict(type='chunk',chunk_index=0,frame_start=0,frames=25)
    monkeypatch.setattr(service,'ROOT',tmp_path);monkeypatch.setattr(service,'store',store)
    monkeypatch.setattr(service,'engine',SimpleNamespace(stream_render=render,last_stream_report={'frames':25}))
    owner=SessionGpuLease(tmp_path,'session').acquire()
    try:
        with TestClient(service.app) as client:
            response=client.post('/render/stream',json=dict(avatar_id='person',session_id='session',generation=0,index=0,token=owner.token))
            assert response.status_code==200
            events=[json.loads(line) for line in response.text.splitlines()]
            assert [event['type'] for event in events]==['start','chunk','complete']
            assert events[1]['generation']==0 and events[1]['sentence_index']==0
            assert any(route.path=='/render' for route in service.app.routes)
    finally:owner.release()
