"""CPU API tests. Only media/model/transport I/O is replaced."""
import json
import sys
import threading
import time
from backend import clock
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.storage import Store, sha256


def imported(client):
    response = client.post('/api/avatars/import', data={'name': '照片人物'}, files={
        'photo': ('photo.png', b'photo-one', 'image/png'),
        'voice_video': ('voice.mp4', b'voice-one', 'video/mp4')})
    assert response.status_code == 201, response.text
    return response.json()['avatar_id']


def wait_job(client, job_id):
    deadline = clock.monotonic() + 5
    while clock.monotonic() < deadline:
        response = client.get('/api/jobs/' + job_id)
        assert response.status_code == 200
        job = response.json()
        if job['status'] in {'complete', 'waiting', 'failed'}:
            return job
        time.sleep(.01)
    pytest.fail('准备任务没有结束')


@pytest.fixture
def model_io(tmp_path, monkeypatch):
    from backend import media
    calls = {'portrait': 0, 'voice': 0}
    catalog_path = tmp_path/'configs/photo-models.json'
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(json.dumps({'models': {'ditto': {'revision': 'a'*40}},
        'sources': {'ditto': {'commit': 'b'*40}}}))

    def audio(store, avatar_id):
        person = store.manifest(avatar_id)
        folder = store.avatar_dir(avatar_id)
        media.write_wav(folder/'voice/reference.wav', np.sin(np.arange(16000)*.1)*.1, 16000)
        person['voice'].update(reference='voice/reference.wav',
            reference_sha256=sha256(folder/'voice/reference.wav'),
            source_sha256=person['sources'][person['voice_video']])
        store.write_manifest(avatar_id, person)
        return person

    class Portrait:
        def __init__(self, base_url):
            pass

        async def prepare(self, avatar_id, variant_id=None):
            calls['portrait'] += 1
            folder = tmp_path/'data/avatars'/avatar_id/'portrait/key'
            folder.mkdir(parents=True, exist_ok=True)
            (folder/'source.npz').write_bytes(b'portrait-state')
            (folder/'idle.mkv').write_bytes(b'cached-idle')
            person = Store(tmp_path).manifest(avatar_id)
            catalog = json.loads(catalog_path.read_text())
            metadata = {'photo_sha256': person['sources'][person['photo']],
                'model_revision': catalog['models']['ditto']['revision'],
                'code_commit': catalog['sources']['ditto']['commit'], 'max_size': 1024}
            (folder/'source.json').write_text(json.dumps({'schema_version': 1, 'metadata': metadata,
                'file': 'source.npz', 'sha256': sha256(folder/'source.npz')}))
            (folder/'ready.json').write_text(json.dumps({'metadata': metadata,
                'idle_sha256': sha256(folder/'idle.mkv'), 'width': 640, 'height': 800,
                'video_format': 'h264_rgb_lossless_v1'}))
            return {'folder': 'portrait/key', 'width': 640, 'height': 800, 'fps': 25}

    class Voice:
        def __init__(self, base_url):
            pass

        async def synthesize(self, text, folder, voice, *, emotion):
            assert text == '你好，我们可以开始聊天了。' and emotion == 'neutral'
            assert voice['confirmed']
            calls['voice'] += 1
            return np.sin(np.arange(24000)*.1).astype(np.float32)*.1, 24000

    class Session:
        def __init__(self, root, manifest, config):
            self.id = 'session-one'
            self.closed = False
            self.last_active = clock.monotonic()
            self.timeline = SimpleNamespace(queue_depth=0)
            self.peer = None

        async def close(self):
            self.closed = True

    async def negotiate(session, offer, *, fps):
        assert fps == 25
        session.peer = object()
        return {'type': 'answer', 'sdp': 'answer'}

    monkeypatch.setattr(media, 'prepare_voice', audio)
    monkeypatch.setitem(sys.modules, 'backend.portrait.client', SimpleNamespace(PortraitClient=Portrait))
    monkeypatch.setitem(sys.modules, 'backend.voice.client', SimpleNamespace(VoiceClient=Voice))
    monkeypatch.setitem(sys.modules, 'backend.transport.photo_session', SimpleNamespace(PhotoSession=Session))
    monkeypatch.setitem(sys.modules, 'backend.transport.photo_webrtc', SimpleNamespace(negotiate=negotiate))
    return calls


def prepare_ready(client, avatar_id):
    response = client.put(f'/api/avatars/{avatar_id}/voice', json={'confirmed': True})
    assert response.status_code == 200
    job = wait_job(client, client.post(f'/api/avatars/{avatar_id}/build').json()['job_id'])
    assert job['status'] == 'complete', job
    return client.get(f'/api/avatars/{avatar_id}').json()


def test_import_photo_fields_and_original_video_is_not_public(tmp_path):
    from backend.api.app import create_app
    with TestClient(create_app(tmp_path)) as client:
        avatar_id = imported(client)
        person = client.get('/api/avatars/' + avatar_id).json()
        assert person['schema_version'] == 3 and person['mode'] == 'photo'
        assert person['settings'] == {'tone': 'natural', 'expression': 'natural'}
        assert len(person['variants']) == 1 and person['active_variant'] == person['variants'][0]['id']
        assert person['variants'][0]['photo'] == person['photo']
        assert person['voice'] == {'confirmed': False, 'language': 'zh'}
        assert person['validation'] == {'photo': False, 'voice': False}
        assert person['portrait'] is None and person['last_job_id']
        assert client.get(f'/api/avatars/{avatar_id}/files/{person["photo"]}').content == b'photo-one'
        assert client.get(f'/api/avatars/{avatar_id}/files/{person["voice_video"]}').status_code == 404
        assert client.get(f'/api/avatars/{avatar_id}/heads').status_code == 404
        assert client.post(f'/api/avatars/{avatar_id}/heads/prepare').status_code == 404
        assert len(client.get('/api/avatars').json()) == 1


def test_preparation_waits_for_confirmation_then_reuses_photo(tmp_path, model_io):
    from backend.api.app import create_app
    with TestClient(create_app(tmp_path)) as client:
        avatar_id = imported(client)
        job = wait_job(client, client.post(f'/api/avatars/{avatar_id}/build').json()['job_id'])
        assert job['status'] == 'waiting' and job['stage'] == 'reference_confirmation'
        assert client.get('/api/avatars/' + avatar_id).json()['status'] == 'voice_pending'
        person = prepare_ready(client, avatar_id)
        assert person['status'] == 'ready' and person['validation'] == {'photo': True, 'voice': True}
        assert model_io == {'portrait': 1, 'voice': 1}
        assert set(person['artifact_hashes']) == {
            'portrait/key/source.json', 'portrait/key/source.npz', 'portrait/key/ready.json',
            'portrait/key/idle.mkv', 'voice/reference.wav', 'voice/sample.wav'}
        response = client.post('/api/sessions', json={'avatar_id': avatar_id})
        assert response.status_code == 201 and response.json() == {
            'session_id': 'session-one', 'width': 640, 'height': 800, 'fps': 25}
        assert client.get('/api/sessions/session-one/metrics').json()['queue_depth'] == 0
        assert client.post('/api/sessions/session-one/offer', json={'type': 'offer', 'sdp': 'offer'}).json()['type'] == 'answer'
        assert client.post('/api/sessions/session-one/offer', json={}).status_code == 409
        assert client.delete('/api/sessions/session-one').status_code == 200


def test_rebuild_upgrades_legacy_video_format_and_then_reuses_cache(tmp_path, model_io):
    from backend.api.app import create_app
    store = Store(tmp_path)
    with TestClient(create_app(tmp_path)) as client:
        avatar_id = imported(client)
        person = prepare_ready(client, avatar_id)
        ready_path = store.avatar_dir(avatar_id)/'portrait/key/ready.json'
        ready = json.loads(ready_path.read_text())
        ready.pop('video_format')
        ready_path.write_text(json.dumps(ready))
        person['artifact_hashes']['portrait/key/ready.json'] = sha256(ready_path)
        store.write_manifest(avatar_id, person)
        response = client.post(f'/api/avatars/{avatar_id}/build')
        assert wait_job(client, response.json()['job_id'])['status'] == 'complete'
        assert model_io['portrait'] == 2
        response = client.post(f'/api/avatars/{avatar_id}/build')
        assert wait_job(client, response.json()['job_id'])['status'] == 'complete'
        assert model_io['portrait'] == 2


@pytest.mark.parametrize('asset', [
    'portrait/key/source.json', 'portrait/key/source.npz', 'portrait/key/ready.json',
    'portrait/key/idle.mkv', 'voice/reference.wav', 'voice/sample.wav', 'photo', 'voice_video'])
def test_modified_assets_cannot_start_a_session(tmp_path, model_io, asset):
    from backend.api.app import create_app
    with TestClient(create_app(tmp_path)) as client:
        avatar_id = imported(client)
        person = prepare_ready(client, avatar_id)
        path = person.get(asset, asset) if asset in {'photo', 'voice_video'} else asset
        (tmp_path/'data/avatars'/avatar_id/path).write_bytes(b'changed')
        assert client.post('/api/sessions', json={'avatar_id': avatar_id}).status_code == 409


def test_missing_asset_or_revoked_confirmation_cannot_start(tmp_path, model_io):
    from backend.api.app import create_app
    with TestClient(create_app(tmp_path)) as client:
        avatar_id = imported(client)
        prepare_ready(client, avatar_id)
        sample = tmp_path/'data/avatars'/avatar_id/'voice/sample.wav'
        sample.unlink()
        assert client.post('/api/sessions', json={'avatar_id': avatar_id}).status_code == 409
        assert client.put(f'/api/avatars/{avatar_id}/voice', json={'confirmed': False}).status_code == 200
        assert client.post('/api/sessions', json={'avatar_id': avatar_id}).status_code == 409


def test_replacing_each_source_keeps_the_other_source_and_raw_file(tmp_path, model_io):
    from backend.api.app import create_app
    with TestClient(create_app(tmp_path)) as client:
        avatar_id = imported(client)
        old = prepare_ready(client, avatar_id)
        response = client.put(f'/api/avatars/{avatar_id}/photo', files={'photo': ('new.png', b'photo-two', 'image/png')})
        assert response.status_code == 200
        person = client.get('/api/avatars/' + avatar_id).json()
        assert person['voice_video'] == old['voice_video'] and person['voice']['confirmed']
        assert person['validation'] == {'photo': False, 'voice': True}
        assert (tmp_path/'data/avatars'/avatar_id/old['photo']).read_bytes() == b'photo-one'
        new_photo = person['photo']
        response = client.put(f'/api/avatars/{avatar_id}/voice-video', files={'voice_video': ('new.mp4', b'voice-two', 'video/mp4')})
        assert response.status_code == 200
        person = client.get('/api/avatars/' + avatar_id).json()
        assert person['photo'] == new_photo and not person['voice']['confirmed']


@pytest.mark.parametrize('legacy_status', ['imported', 'preparing'])
def test_legacy_get_does_not_write_and_first_build_keeps_backup(tmp_path, model_io, legacy_status):
    from backend.api.app import create_app
    store = Store(tmp_path)
    photo = tmp_path/'body.png'; photo.write_bytes(b'legacy-photo')
    video = tmp_path/'video.mp4'; video.write_bytes(b'legacy-voice')
    old = {'schema_version':1,'id':'legacy','name':'旧人物','created_at':'2026-01-01',
        'status':legacy_status,'video':'video.mp4','body_photos':['body.png'],'face_photos':[],
        'sources':{'body.png':sha256(photo),'video.mp4':sha256(video)},
        'voice':{'confirmed':False,'language':'zh','text':''}}
    source=store.avatar_dir(old['id'])/'source';source.mkdir(parents=True)
    (source/'body.png').write_bytes(photo.read_bytes())
    (source/'video.mp4').write_bytes(video.read_bytes())
    store.write_manifest(old['id'],old)
    avatar_id = old['id']; folder = store.avatar_dir(avatar_id)
    original = (folder/'avatar.json').read_bytes()
    with TestClient(create_app(tmp_path)) as client:
        assert client.get('/api/avatars/' + avatar_id).json()['schema_version'] == 3
        assert client.get('/api/avatars').json()[0]['schema_version'] == 3
        assert (folder/'avatar.json').read_bytes() == original
        wait_job(client, client.post(f'/api/avatars/{avatar_id}/build').json()['job_id'])
        assert store.manifest(avatar_id)['schema_version'] == 3
        assert (folder/'avatar.legacy.json').read_bytes() == original
        assert (folder/'source/body.png').read_bytes() == b'legacy-photo'


def test_build_and_session_block_mutation_and_delete(tmp_path, model_io, monkeypatch):
    from backend.api.app import create_app
    from backend import media
    entered, release = threading.Event(), threading.Event()
    real = media.prepare_voice
    def blocking(*args):
        entered.set()
        assert release.wait(5)
        return real(*args)
    with TestClient(create_app(tmp_path)) as client:
        avatar_id = imported(client)
        monkeypatch.setattr(media, 'prepare_voice', blocking)
        response = client.post(f'/api/avatars/{avatar_id}/build')
        assert entered.wait(2)
        try:
            assert client.delete('/api/avatars/' + avatar_id).status_code == 409
            assert client.put(f'/api/avatars/{avatar_id}/voice', json={'confirmed': True}).status_code == 409
            assert client.post(f'/api/avatars/{avatar_id}/build').status_code == 409
            assert client.put(f'/api/avatars/{avatar_id}/photo', files={'photo': ('x.png', b'x')}).status_code == 409
        finally:
            release.set()
        wait_job(client, response.json()['job_id'])
        prepare_ready(client, avatar_id)
        assert client.post('/api/sessions', json={'avatar_id': avatar_id}).status_code == 201
        assert client.delete('/api/avatars/' + avatar_id).status_code == 409
        assert client.post(f'/api/avatars/{avatar_id}/build').status_code == 409
        assert client.put(f'/api/avatars/{avatar_id}/voice', json={'confirmed': False}).status_code == 409
        client.delete('/api/sessions/session-one')
        assert client.delete('/api/avatars/' + avatar_id).status_code == 200


def test_delete_honors_cross_process_lock_and_rejects_linked_folder(tmp_path):
    from backend.api.app import create_app
    from backend.gpu import GpuLease
    with TestClient(create_app(tmp_path)) as client:
        avatar_id = imported(client)
        lease = GpuLease(tmp_path).acquire()
        try:
            assert client.delete('/api/avatars/' + avatar_id).status_code == 409
        finally:
            lease.release()
        outside = tmp_path/'outside'; outside.mkdir()
        source = tmp_path/'data/avatars'/avatar_id
        (outside/'avatar.json').write_bytes((source/'avatar.json').read_bytes())
        source.rename(tmp_path/'retained-avatar')
        try:source.symlink_to(outside, target_is_directory=True)
        except OSError:pytest.skip('Windows directory links require developer mode')
        assert client.delete('/api/avatars/' + avatar_id).status_code == 400
        assert (outside/'avatar.json').is_file()


def test_cache_descriptor_cannot_reference_state_outside_cache(tmp_path, model_io):
    from backend.api.app import create_app
    with TestClient(create_app(tmp_path)) as client:
        avatar_id = imported(client)
        person = prepare_ready(client, avatar_id)
        folder = tmp_path/'data/avatars'/avatar_id
        descriptor = json.loads((folder/'portrait/key/source.json').read_text())
        outside = folder/'outside.npz'; outside.write_bytes(b'outside')
        descriptor.update(file='../../outside.npz', sha256=sha256(outside))
        (folder/'portrait/key/source.json').write_text(json.dumps(descriptor))
        person['artifact_hashes']['portrait/key/source.json'] = sha256(folder/'portrait/key/source.json')
        Store(tmp_path).write_manifest(avatar_id, person)
        assert client.post('/api/sessions', json={'avatar_id': avatar_id}).status_code == 409


def test_invalid_pcm_fails_at_voice_stage_without_ready_status(tmp_path, model_io, monkeypatch):
    from backend.api.app import create_app
    async def silence(*args, **kwargs):
        return np.zeros(24000, np.float32), 24000
    monkeypatch.setattr(sys.modules['backend.voice.client'].VoiceClient, 'synthesize', silence)
    with TestClient(create_app(tmp_path)) as client:
        avatar_id = imported(client)
        client.put(f'/api/avatars/{avatar_id}/voice', json={'confirmed': True})
        job = wait_job(client, client.post(f'/api/avatars/{avatar_id}/build').json()['job_id'])
        assert job['status'] == 'failed' and job['stage'] == 'voice'
        assert 'PCM' in job['error']
        assert client.get('/api/avatars/' + avatar_id).json()['status'] == 'failed'
        assert client.post('/api/sessions', json={'avatar_id': avatar_id}).status_code == 409


def test_loading_blocks_mutation_and_releases_guard_on_constructor_failure(tmp_path, model_io, monkeypatch):
    from backend.api.app import create_app
    entered, release = threading.Event(), threading.Event()
    def blocked(*args):
        entered.set()
        assert release.wait(5)
        raise RuntimeError('加载失败')
    monkeypatch.setattr(sys.modules['backend.transport.photo_session'], 'PhotoSession', blocked)
    response = []
    with TestClient(create_app(tmp_path), raise_server_exceptions=False) as client:
        avatar_id = imported(client)
        prepare_ready(client, avatar_id)
        request = threading.Thread(target=lambda: response.append(client.post('/api/sessions', json={'avatar_id': avatar_id})))
        request.start()
        assert entered.wait(2)
        try:
            assert client.delete('/api/avatars/' + avatar_id).status_code == 409
            assert client.put(f'/api/avatars/{avatar_id}/voice', json={'confirmed': False}).status_code == 409
        finally:
            release.set(); request.join(5)
        assert response[0].status_code == 500
        assert client.delete('/api/avatars/' + avatar_id).status_code == 200


def test_session_lifespan_closes_resources(tmp_path, model_io, monkeypatch):
    from backend.api.app import create_app
    original = sys.modules['backend.transport.photo_session'].PhotoSession
    closed = []
    class Session(original):
        async def close(self):
            closed.append(self.id)
            await super().close()
    monkeypatch.setattr(sys.modules['backend.transport.photo_session'], 'PhotoSession', Session)
    with TestClient(create_app(tmp_path)) as client:
        avatar_id = imported(client)
        prepare_ready(client, avatar_id)
        assert client.post('/api/sessions', json={'avatar_id': avatar_id}).status_code == 201
    assert closed == ['session-one']


@pytest.mark.parametrize(('job_status', 'avatar_status'), [('queued', 'ready'), ('running', 'preparing')])
def test_restart_recovers_interrupted_preparation_and_allows_retry(tmp_path, model_io, job_status, avatar_status):
    from backend.api.app import create_app
    store = Store(tmp_path)
    with TestClient(create_app(tmp_path)) as client:
        avatar_id = imported(client)
        person = prepare_ready(client, avatar_id)
    original_files = {name: (store.avatar_dir(avatar_id)/name).read_bytes()
                      for name in [person['photo'], person['voice_video'], *person['artifact_hashes']]}
    job = store.new_job(avatar_id, 'prepare')
    store.update_job(job['id'], status=job_status, stage='portrait', progress=.4)
    person.update(status=avatar_status, last_job_id=job['id'])
    store.write_manifest(avatar_id, person)

    with TestClient(create_app(tmp_path)) as client:
        recovered = client.get('/api/avatars/' + avatar_id).json()
        interrupted = client.get('/api/jobs/' + job['id']).json()
        assert interrupted['status'] == 'failed'
        assert interrupted['error'] and interrupted['progress'] == .4
        assert recovered['status'] == ('failed' if avatar_status == 'preparing' else 'ready')
        assert recovered['voice'] == person['voice']
        assert recovered['artifact_hashes'] == person['artifact_hashes']
        assert original_files == {name: (store.avatar_dir(avatar_id)/name).read_bytes() for name in original_files}
        response = client.post(f'/api/avatars/{avatar_id}/build')
        assert response.status_code == 202
        assert wait_job(client, response.json()['job_id'])['status'] == 'complete'
        assert model_io['portrait'] == 1


@pytest.mark.parametrize('confirmed', [False, True])
def test_restart_preserves_completed_or_confirmation_waiting_preparation(tmp_path, model_io, confirmed):
    from backend.api.app import create_app
    with TestClient(create_app(tmp_path)) as client:
        avatar_id = imported(client)
        if confirmed:
            person = prepare_ready(client, avatar_id)
        else:
            wait_job(client, client.post(f'/api/avatars/{avatar_id}/build').json()['job_id'])
            person = client.get('/api/avatars/' + avatar_id).json()
        job = client.get('/api/jobs/' + person['last_job_id']).json()
    with TestClient(create_app(tmp_path)) as client:
        assert client.get('/api/avatars/' + avatar_id).json() == person
        assert client.get('/api/jobs/' + job['id']).json() == job


def test_restart_recovers_preparing_avatar_without_an_active_job(tmp_path, model_io):
    from backend.api.app import create_app
    store = Store(tmp_path)
    with TestClient(create_app(tmp_path)) as client:
        avatar_id = imported(client)
        person = prepare_ready(client, avatar_id)
    person['status'] = 'preparing'
    store.write_manifest(avatar_id, person)
    completed_job = store.job(person['last_job_id'])
    with TestClient(create_app(tmp_path)) as client:
        assert client.get('/api/avatars/' + avatar_id).json()['status'] == 'failed'
        assert client.get('/api/jobs/' + completed_job['id']).json() == completed_job
        retry = client.post(f'/api/avatars/{avatar_id}/build')
        assert retry.status_code == 202
        assert wait_job(client, retry.json()['job_id'])['status'] == 'complete'


@pytest.mark.parametrize('disconnected', [False, True])
def test_reaper_closes_idle_or_disconnected_session(tmp_path, model_io, monkeypatch, disconnected):
    from backend.api.app import create_app
    configs = tmp_path/'configs'; configs.mkdir(exist_ok=True)
    (configs/'app.yaml').write_text('runtime:\n  idle_timeout_seconds: 0\n')
    original = sys.modules['backend.transport.photo_session'].PhotoSession
    instances = []
    closed = []
    class Session(original):
        def __init__(self, *args):
            super().__init__(*args)
            instances.append(self)

        async def close(self):
            closed.append(self.id)
            await super().close()
    monkeypatch.setattr(sys.modules['backend.transport.photo_session'], 'PhotoSession', Session)
    with TestClient(create_app(tmp_path)) as client:
        avatar_id = imported(client)
        prepare_ready(client, avatar_id)
        assert client.post('/api/sessions', json={'avatar_id': avatar_id}).status_code == 201
        if disconnected:
            # A closed transport must be reaped even when it was active recently.
            instances[0].closed = True
            instances[0].last_active = clock.monotonic() + 100
        deadline = clock.monotonic() + 12
        while client.get('/api/sessions/session-one/metrics').status_code != 404:
            assert clock.monotonic() < deadline
            time.sleep(.05)
        assert closed == ['session-one']
        assert client.delete('/api/avatars/' + avatar_id).status_code == 200


@pytest.mark.parametrize(('role', 'filename'), [('photo', 'C:escaped.png'), ('voice_video', 'voice:stream.mp4')])
@pytest.mark.parametrize('replace', [False, True])
def test_upload_rejects_drive_or_colon_filename_before_import(tmp_path, role, filename, replace):
    from backend.api.app import create_app
    with TestClient(create_app(tmp_path)) as client:
        if replace:
            avatar_id = imported(client)
            endpoint = 'photo' if role == 'photo' else 'voice-video'
            response = client.put(f'/api/avatars/{avatar_id}/{endpoint}', files={role: (filename, b'unsafe')})
        else:
            files = {'photo': ('safe.png', b'photo'), 'voice_video': ('safe.mp4', b'voice')}
            files[role] = (filename, b'unsafe')
            response = client.post('/api/avatars/import', files=files)
        assert response.status_code == 400
        assert len(client.get('/api/avatars').json()) == int(replace)


@pytest.mark.parametrize('changed', ['model_revision', 'code_commit', 'max_size'])
def test_changed_portrait_version_rejects_ready_and_regenerates(tmp_path, model_io, changed):
    from backend.api.app import create_app
    with TestClient(create_app(tmp_path)) as client:
        avatar_id = imported(client)
        person = prepare_ready(client, avatar_id)
        if changed in {'model_revision', 'code_commit'}:
            path = tmp_path/'configs/photo-models.json'
            catalog = json.loads(path.read_text())
            if changed == 'model_revision':
                catalog['models']['ditto']['revision'] = 'c'*40
            else:
                catalog['sources']['ditto']['commit'] = 'd'*40
            path.write_text(json.dumps(catalog))
        else:
            folder = tmp_path/'data/avatars'/avatar_id
            for name in ('source.json', 'ready.json'):
                relative = 'portrait/key/' + name
                path = folder/relative
                cached = json.loads(path.read_text())
                cached['metadata']['max_size'] = 512
                path.write_text(json.dumps(cached))
                person['artifact_hashes'][relative] = sha256(path)
            Store(tmp_path).write_manifest(avatar_id, person)
        assert client.post('/api/sessions', json={'avatar_id': avatar_id}).status_code == 409
        rebuilt = prepare_ready(client, avatar_id)
        assert rebuilt['status'] == 'ready' and model_io['portrait'] == 2
        assert client.post('/api/sessions', json={'avatar_id': avatar_id}).status_code == 201


def test_add_variant_then_switch_uses_its_own_cache(tmp_path, model_io, monkeypatch):
    from backend.api.app import create_app
    from backend.photo_profiles import migrate_profile, resolve_variant
    Portrait = sys.modules['backend.portrait.client'].PortraitClient
    catalog = json.loads((tmp_path/'configs/photo-models.json').read_text())

    async def prepare(self, avatar_id, variant_id=None):
        store = Store(tmp_path)
        person = migrate_profile(store.manifest(avatar_id))
        variant = resolve_variant(person, variant_id)
        folder = store.avatar_dir(avatar_id)
        digest = sha256(folder/variant['photo'])
        name = 'k'+digest[:16]
        cache = folder/'portrait'/name
        cache.mkdir(parents=True, exist_ok=True)
        (cache/'source.npz').write_bytes(b'state')
        (cache/'idle.mkv').write_bytes(b'idle')
        metadata = {'photo_sha256': digest,
                    'model_revision': catalog['models']['ditto']['revision'],
                    'code_commit': catalog['sources']['ditto']['commit'], 'max_size': 1024}
        (cache/'source.json').write_text(json.dumps({'schema_version': 1, 'metadata': metadata,
            'file': 'source.npz', 'sha256': sha256(cache/'source.npz')}))
        (cache/'ready.json').write_text(json.dumps({'metadata': metadata,
            'idle_sha256': sha256(cache/'idle.mkv'), 'width': 640, 'height': 800,
            'video_format': 'h264_rgb_lossless_v1'}))
        return {'folder': f'portrait/{name}', 'width': 640, 'height': 800, 'fps': 25}

    monkeypatch.setattr(Portrait, 'prepare', prepare)
    with TestClient(create_app(tmp_path)) as client:
        avatar_id = imported(client)
        client.put(f'/api/avatars/{avatar_id}/voice', json={'confirmed': True})
        job = client.post(f'/api/avatars/{avatar_id}/build').json()['job_id']
        assert wait_job(client, job)['status'] == 'complete'
        response = client.post(f'/api/avatars/{avatar_id}/variants',
            files={'photo': ('second.png', b'photo-two', 'image/png')})
        assert response.status_code == 201, response.text
        person = response.json()
        second = [item for item in person['variants'] if item['photo'] != person['variants'][0]['photo']][0]
        assert person['active_variant'] != second['id']
        assert client.put(f"/api/avatars/{avatar_id}/variants/{second['id']}/active").status_code == 409
        job = client.post(f'/api/avatars/{avatar_id}/build').json()['job_id']
        assert wait_job(client, job)['status'] == 'complete'
        switched = client.put(f"/api/avatars/{avatar_id}/variants/{second['id']}/active")
        assert switched.status_code == 200, switched.text
        person = switched.json()
        assert person['active_variant'] == second['id'] and person['photo'] == second['photo']
        assert person['status'] == 'ready'
        folders = {item['portrait']['folder'] for item in person['variants']}
        assert len(folders) == 2
        for folder in folders:
            assert f'{folder}/idle.mkv' in person['artifact_hashes']
        assert 'voice/sample.wav' in person['artifact_hashes']
        assert client.post('/api/sessions', json={'avatar_id': avatar_id}).status_code == 201


def test_settings_put_persists_and_validates(tmp_path, model_io):
    from backend.api.app import create_app
    with TestClient(create_app(tmp_path)) as client:
        avatar_id = imported(client)
        response = client.put(f'/api/avatars/{avatar_id}/settings', json={'tone': 'lively'})
        assert response.status_code == 200
        assert response.json()['settings'] == {'tone': 'lively', 'expression': 'natural'}
        person = client.get('/api/avatars/' + avatar_id).json()
        assert person['settings'] == {'tone': 'lively', 'expression': 'natural'}
        assert client.put(f'/api/avatars/{avatar_id}/settings',
            json={'expression': 'grin'}).status_code == 400
        assert client.get('/api/avatars/' + avatar_id).json()['settings']['expression'] == 'natural'


def _fake_outfit(command, **kwargs):
    """Stand in for the outfit subprocess: write a PNG and report success."""
    from pathlib import Path
    from PIL import Image
    out = Path(command[command.index('--out') + 1])
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((16, 16, 3), 200, np.uint8)).save(out)
    return SimpleNamespace(stdout=json.dumps({'label': '换装·藏青', 'mode': 'recolor'}),
                           stderr='', returncode=0)


def test_outfit_adds_a_variant_without_switching(tmp_path, model_io, monkeypatch):
    from pathlib import Path
    import backend.api.app as app_module
    from backend.api.app import create_app
    monkeypatch.setattr(app_module.subprocess, 'run', _fake_outfit)
    with TestClient(create_app(tmp_path)) as client:
        avatar_id = imported(client)
        before = client.get('/api/avatars/' + avatar_id).json()
        response = client.post(f'/api/avatars/{avatar_id}/outfit',
                               json={'mode': 'recolor', 'color': '#1e2a44'})
        assert response.status_code == 201, response.text
        person = response.json()
        assert len(person['variants']) == len(before['variants']) + 1
        assert person['variants'][-1]['label'] == '换装·藏青'
        assert person['active_variant'] == before['active_variant']
        assert person['status'] == before['status']
        assert person['variants'][-1]['portrait'] is None
        # The edited photo is a real source file, not a reference to the old one.
        folder = Store(tmp_path).avatar_dir(avatar_id)
        assert (folder / person['variants'][-1]['photo']).is_file()
        assert person['variants'][-1]['photo'] != before['photo']


def test_outfit_rejects_unknown_mode_missing_model_and_changed_photo(tmp_path, model_io, monkeypatch):
    import backend.api.app as app_module
    from backend.api.app import create_app
    monkeypatch.setattr(app_module.subprocess, 'run', _fake_outfit)
    with TestClient(create_app(tmp_path)) as client:
        avatar_id = imported(client)
        assert client.post(f'/api/avatars/{avatar_id}/outfit',
                           json={'mode': 'paint'}).status_code == 400
        assert client.post(f'/api/avatars/{avatar_id}/outfit',
                           json={'mode': 'generate', 'style': 'shirt'}).status_code == 503
        person = client.get('/api/avatars/' + avatar_id).json()
        photo = Store(tmp_path).avatar_dir(avatar_id) / person['photo']
        photo.write_bytes(photo.read_bytes() + b'x')
        assert client.post(f'/api/avatars/{avatar_id}/outfit',
                           json={'mode': 'recolor', 'color': '#1e2a44'}).status_code == 400


def test_wear_requires_the_reference_model(tmp_path, model_io):
    from backend.api.app import create_app
    with TestClient(create_app(tmp_path)) as client:
        avatar_id = imported(client)
        response = client.post(f'/api/avatars/{avatar_id}/outfit/wear',
                               files={'garment': ('g.png', b'plain', 'image/png')})
        assert response.status_code == 503
        assert '参考模型' in response.json()['detail']


def test_wear_adds_a_variant_from_an_uploaded_garment(tmp_path, model_io, monkeypatch):
    import backend.api.app as app_module
    from backend.api.app import create_app
    from backend import outfit as outfit_module
    adapter = tmp_path/'models/outfit/ip-adapter/models'
    (adapter/'image_encoder').mkdir(parents=True)
    (adapter/outfit_module.ADAPTER_WEIGHT).write_bytes(b'weights')
    (adapter/'image_encoder/model.safetensors').write_bytes(b'encoder')
    (tmp_path/'models/outfit/inpaint').mkdir(parents=True)
    (tmp_path/'models/outfit/inpaint/model_index.json').write_text('{}')
    monkeypatch.setattr(app_module.subprocess, 'run', _fake_outfit)
    with TestClient(create_app(tmp_path)) as client:
        avatar_id = imported(client)
        response = client.post(f'/api/avatars/{avatar_id}/outfit/wear',
                               files={'garment': ('denim.png', b'denim-bytes', 'image/png')})
        assert response.status_code == 201, response.text
        person = response.json()
        assert person['variants'][-1]['label'] == '换装·藏青'
        assert person['active_variant'] != person['variants'][-1]['id']
