from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from backend.api.app import create_app
from backend.avatar.lease import GpuLease


def imported(client,name='人物'):
    response=client.post('/api/avatars/import',data={'name':name},files=[
        ('video',('video.mp4',b'original-video','video/mp4')),
        ('body_photos',('body.png',b'original-photo','image/png'))])
    assert response.status_code==201
    return response.json()['avatar_id']


def test_delete_removes_only_selected_person_and_reassigns_default(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        first=imported(client,'删除的人物');second=imported(client,'保留的人物')
        folder=tmp_path/'data/avatars'/first
        (folder/'voice').mkdir();(folder/'voice/reference.wav').write_bytes(b'reference')
        marker=tmp_path/'.runtime/default_avatar.txt';marker.parent.mkdir(exist_ok=True);marker.write_text(first)
        shared=tmp_path/'models/shared.bin';shared.parent.mkdir();shared.write_bytes(b'shared-model')
        response=client.delete('/api/avatars/'+first)
        assert response.status_code==200,response.text
        assert response.json()['deleted'] is True
        assert not folder.exists()
        assert [a['id'] for a in client.get('/api/avatars').json()]==[second]
        assert marker.read_text().strip()==second
        assert shared.read_bytes()==b'shared-model'
        assert (tmp_path/'data/avatars'/second/'source/video.mp4').read_bytes()==b'original-video'
        assert client.delete('/api/avatars/'+first).status_code==404


def test_delete_last_person_clears_default_and_keeps_external_originals(tmp_path):
    from backend.storage import Store
    root=tmp_path/'project';source=tmp_path/'originals';source.mkdir()
    video=source/'clip.mp4';video.write_bytes(b'private-original')
    photo=source/'body.png';photo.write_bytes(b'original-photo')
    avatar=Store(root).import_files('人物',video,[photo],[])
    marker=root/'.runtime/default_avatar.txt';marker.parent.mkdir();marker.write_text(avatar['id'])
    with TestClient(create_app(root)) as client:
        assert client.delete('/api/avatars/'+avatar['id']).status_code==200
        assert client.get('/api/avatars').json()==[]
    assert not marker.exists()
    assert video.read_bytes()==b'private-original' and photo.read_bytes()==b'original-photo'


def test_delete_rejects_gpu_operations_started_by_other_processes(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        avatar_id=imported(client)
        lease=GpuLease(tmp_path).acquire()
        try:
            response=client.delete('/api/avatars/'+avatar_id)
            assert response.status_code==409
            assert client.get('/api/avatars/'+avatar_id).status_code==200
        finally:lease.release()
        assert client.delete('/api/avatars/'+avatar_id).status_code==200


def test_delete_rejects_queued_or_running_api_build(tmp_path,monkeypatch):
    import threading
    from backend.pipeline import build
    entered=threading.Event();release=threading.Event();finished=threading.Event()
    def slow_build(*args):
        entered.set()
        try:release.wait(5)
        finally:finished.set()
    monkeypatch.setattr(build,'build_avatar',slow_build)
    with TestClient(create_app(tmp_path)) as client:
        avatar_id=imported(client)
        try:
            assert client.post(f'/api/avatars/{avatar_id}/build',json={'stage':'prepare'}).status_code==202
            assert entered.wait(2)
            assert client.delete('/api/avatars/'+avatar_id).status_code==409
            assert client.get('/api/avatars/'+avatar_id).status_code==200
        finally:
            release.set();assert finished.wait(2)


def test_delete_refuses_avatar_directory_link_to_external_data(tmp_path):
    from backend.storage import atomic_json
    root=tmp_path/'project';outside=tmp_path/'external';outside.mkdir()
    atomic_json(outside/'avatar.json',{'id':'linked','name':'外部目录','status':'imported'})
    keep=outside/'important.bin';keep.write_bytes(b'do-not-delete')
    base=root/'data/avatars';base.mkdir(parents=True)
    try:(base/'linked').symlink_to(outside,target_is_directory=True)
    except OSError:pytest.skip('Directory links require Windows developer mode')
    with TestClient(create_app(root)) as client:
        assert client.delete('/api/avatars/linked').status_code==400
    assert keep.read_bytes()==b'do-not-delete'
