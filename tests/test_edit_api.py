"""The editing workbench API: analysis, plan, compose preview and edit jobs."""
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend import clock
from backend.storage import Store


def png_bytes(size=(400, 500), tag=1):
    from io import BytesIO
    image = Image.new('RGB', size, (180, 160, 140))
    image.putpixel((0, 0), (tag % 256, 7, 11))
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


def imported(client, name='照片人物'):
    response = client.post('/api/avatars/import', data={'name': name}, files={
        'photo': ('photo.png', png_bytes(), 'image/png'),
        'voice_video': ('voice.mp4', b'voice-one', 'video/mp4')})
    assert response.status_code == 201, response.text
    return response.json()['avatar_id']


def wait_job(client, job_id, timeout=8):
    deadline = clock.monotonic() + timeout
    while clock.monotonic() < deadline:
        job = client.get('/api/jobs/' + job_id).json()
        if job['status'] in {'complete', 'failed'}:
            return job
        time.sleep(.02)
    pytest.fail('编辑任务没有结束')


ANALYSIS = {'size': [400, 500], 'box': [120, 20, 300, 430], 'head': 110,
            'colors': {'lower': [46, 50, 62], 'skin': [205, 180, 158]}}

EXTEND_REPORT = {'mode': 'extend', 'label': '全身', 'quality': 'good',
                 'face_diff_ratio': 0., 'protected_diff_ratio': 0.,
                 'leg_ratio': .4}


def fake_edit(command, **kwargs):
    if '--analyze' in command:
        return SimpleNamespace(stdout=json.dumps(ANALYSIS), stderr='', returncode=0)
    out = Path(command[command.index('--out') + 1])
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((760, 570, 3), 200, np.uint8)).save(out)
    return SimpleNamespace(stdout=json.dumps(EXTEND_REPORT), stderr='', returncode=0)


@pytest.fixture
def edit_io(tmp_path, monkeypatch):
    import backend.api.app as app_module
    models = tmp_path/'models/outfit/inpaint'
    models.mkdir(parents=True)
    (models/'model_index.json').write_text('{}')
    monkeypatch.setattr(app_module.subprocess, 'run', fake_edit)
    return tmp_path


def adapter_files(root, outfit_module):
    folder = root/'models/outfit/ip-adapter/models'
    (folder/'image_encoder').mkdir(parents=True, exist_ok=True)
    (folder/outfit_module.ADAPTER_WEIGHT).write_bytes(b'weights')
    (folder/'image_encoder/model.safetensors').write_bytes(b'encoder')


def test_analysis_is_cached_per_photo(tmp_path, edit_io):
    from backend.api.app import create_app
    with TestClient(create_app(tmp_path)) as client:
        avatar_id = imported(client)
        first = client.post(f'/api/avatars/{avatar_id}/edit/analyze')
        assert first.status_code == 200, first.text
        assert first.json()['box'] == ANALYSIS['box']
        photo = client.get('/api/avatars/' + avatar_id).json()['photo']
        cached = tmp_path/'work/edit'
        assert len(list(cached.glob('*/analysis.json'))) == 1
        second = client.post(f'/api/avatars/{avatar_id}/edit/analyze')
        assert second.json() == first.json()
        assert photo


def test_analysis_rejects_a_picture_without_a_person(tmp_path, edit_io, monkeypatch):
    import backend.api.app as app_module

    def no_person(command, **kwargs):
        if '--analyze' in command:
            return SimpleNamespace(stdout=json.dumps({'error': '没有识别到完整的人物'}),
                                   stderr='', returncode=2)
        return fake_edit(command, **kwargs)

    monkeypatch.setattr(app_module.subprocess, 'run', no_person)
    from backend.api.app import create_app
    with TestClient(create_app(tmp_path)) as client:
        avatar_id = imported(client)
        response = client.post(f'/api/avatars/{avatar_id}/edit/analyze')
        assert response.status_code == 400
        assert '没有识别到完整的人物' in response.json()['detail']


def test_plan_needs_an_analysis_first(tmp_path, edit_io):
    from backend.api.app import create_app
    with TestClient(create_app(tmp_path)) as client:
        avatar_id = imported(client)
        response = client.post(f'/api/avatars/{avatar_id}/edit/plan', json={'aspect': '3:4'})
        assert response.status_code == 409
        assert '先分析' in response.json()['detail']


def test_plan_builds_a_taller_canvas_and_a_compose_picture(tmp_path, edit_io):
    from backend.api.app import create_app
    with TestClient(create_app(tmp_path)) as client:
        avatar_id = imported(client)
        client.post(f'/api/avatars/{avatar_id}/edit/analyze')
        response = client.post(f'/api/avatars/{avatar_id}/edit/plan', json={'aspect': '3:4'})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body['plan']['canvas'][1] > 500
        assert body['compose']['url']
        picture = client.get(body['compose']['url'])
        assert picture.status_code == 200
        assert picture.headers['content-type'] == 'image/png'
        # The second call reuses the file on disk.
        again = client.post(f'/api/avatars/{avatar_id}/edit/plan', json={'aspect': '3:4'})
        assert again.json()['compose']['url'] == body['compose']['url']
        assert len(list((tmp_path/'work/edit').glob('*/compose-3x4.png'))) == 1


def test_plan_warns_about_the_two_pass_aspect(tmp_path, edit_io):
    from backend.api.app import create_app
    with TestClient(create_app(tmp_path)) as client:
        avatar_id = imported(client)
        client.post(f'/api/avatars/{avatar_id}/edit/analyze')
        body = client.post(f'/api/avatars/{avatar_id}/edit/plan',
                           json={'aspect': '9:16'}).json()
        assert any('两段' in warning for warning in body['warnings'])


def test_plan_warns_when_a_close_up_cannot_become_a_full_body(tmp_path, edit_io, monkeypatch):
    import backend.api.app as app_module
    close_up = {**ANALYSIS, 'box': [60, 10, 340, 500], 'head': 380}

    def analyzer(command, **kwargs):
        if '--analyze' in command:
            return SimpleNamespace(stdout=json.dumps(close_up), stderr='', returncode=0)
        return fake_edit(command, **kwargs)

    monkeypatch.setattr(app_module.subprocess, 'run', analyzer)
    from backend.api.app import create_app
    with TestClient(create_app(tmp_path)) as client:
        avatar_id = imported(client)
        client.post(f'/api/avatars/{avatar_id}/edit/analyze')
        body = client.post(f'/api/avatars/{avatar_id}/edit/plan',
                           json={'aspect': '3:4'}).json()
        assert body['plan']['capped'] is True
        assert any('七分身' in warning for warning in body['warnings'])


def test_plan_rejects_an_unknown_aspect(tmp_path, edit_io):
    from backend.api.app import create_app
    with TestClient(create_app(tmp_path)) as client:
        avatar_id = imported(client)
        client.post(f'/api/avatars/{avatar_id}/edit/analyze')
        response = client.post(f'/api/avatars/{avatar_id}/edit/plan', json={'aspect': '1:1'})
        assert response.status_code == 400


def test_compose_refuses_an_unknown_name(tmp_path, edit_io):
    from backend.api.app import create_app
    with TestClient(create_app(tmp_path)) as client:
        avatar_id = imported(client)
        client.post(f'/api/avatars/{avatar_id}/edit/analyze')
        assert client.get(
            f'/api/avatars/{avatar_id}/edit/compose/../../avatar.json').status_code == 404


def test_prompt_endpoint_translates_three_kinds(tmp_path, edit_io, monkeypatch):
    from backend.api.app import create_app
    from backend.api import app as app_module
    monkeypatch.setattr(app_module.prompting, 'translate_cached',
                        lambda text, kind, **kwargs: f'{kind}:{text}')
    with TestClient(create_app(tmp_path)) as client:
        avatar_id = imported(client)
        body = client.post(f'/api/avatars/{avatar_id}/edit/prompt',
                           json={'action': '走路', 'tone': '温柔', 'expression': '微笑'}).json()
        assert body['action'] == {'zh': '走路', 'en': 'action:走路'}
        assert body['tone']['en'] == 'tone:温柔'
        assert body['expression']['en'] == 'expression:微笑'


def test_extend_runs_as_a_job_and_adds_a_variant(tmp_path, edit_io):
    from backend.api.app import create_app
    with TestClient(create_app(tmp_path)) as client:
        avatar_id = imported(client)
        before = client.get('/api/avatars/' + avatar_id).json()
        response = client.post(f'/api/avatars/{avatar_id}/extend',
                               json={'aspect': '3:4', 'action': '走路'})
        assert response.status_code == 202, response.text
        job = wait_job(client, response.json()['job_id'])
        assert job['status'] == 'complete', job
        assert job['result']['variant_id']
        assert job['result']['quality'] == 'good'
        person = client.get('/api/avatars/' + avatar_id).json()
        assert len(person['variants']) == len(before['variants']) + 1
        assert person['active_variant'] == before['active_variant']
        assert person['variants'][-1]['id'] == job['result']['variant_id']


def test_extend_requires_the_model(tmp_path):
    from backend.api.app import create_app
    with TestClient(create_app(tmp_path)) as client:
        avatar_id = imported(client)
        response = client.post(f'/api/avatars/{avatar_id}/extend', json={'aspect': '3:4'})
        assert response.status_code == 503
        assert '扩图模型' in response.json()['detail']


def test_extend_rejects_an_unknown_aspect(tmp_path, edit_io):
    from backend.api.app import create_app
    with TestClient(create_app(tmp_path)) as client:
        avatar_id = imported(client)
        response = client.post(f'/api/avatars/{avatar_id}/extend', json={'aspect': '2:1'})
        assert response.status_code == 400


def test_extend_reports_a_skipped_photo_without_a_variant(tmp_path, edit_io, monkeypatch):
    import backend.api.app as app_module

    def already_full(command, **kwargs):
        if '--analyze' in command:
            return fake_edit(command, **kwargs)
        return SimpleNamespace(stdout=json.dumps({'skipped': 'already_full_body',
                                                  'label': '全身·已完整'}),
                               stderr='', returncode=0)

    monkeypatch.setattr(app_module.subprocess, 'run', already_full)
    from backend.api.app import create_app
    with TestClient(create_app(tmp_path)) as client:
        avatar_id = imported(client)
        before = client.get('/api/avatars/' + avatar_id).json()
        job = wait_job(client, client.post(f'/api/avatars/{avatar_id}/extend',
                                           json={'aspect': '3:4'}).json()['job_id'])
        assert job['status'] == 'complete'
        assert job['result']['skipped'] == 'already_full_body'
        assert len(client.get('/api/avatars/' + avatar_id).json()['variants']) == \
            len(before['variants'])


def test_extend_blocks_other_writes_while_it_runs(tmp_path, edit_io, monkeypatch):
    import backend.api.app as app_module
    gate = threading.Event()

    def slow(command, **kwargs):
        if '--analyze' in command:
            return fake_edit(command, **kwargs)
        gate.wait(timeout=5)
        return fake_edit(command, **kwargs)

    monkeypatch.setattr(app_module.subprocess, 'run', slow)
    from backend.api.app import create_app
    with TestClient(create_app(tmp_path)) as client:
        avatar_id = imported(client)
        try:
            response = client.post(f'/api/avatars/{avatar_id}/extend', json={'aspect': '3:4'})
            assert response.status_code == 202
            blocked = client.post(f'/api/avatars/{avatar_id}/outfit',
                                  json={'mode': 'recolor', 'color': '#1e2a44'})
            assert blocked.status_code == 409
            assert '编辑' in blocked.json()['detail']
        finally:
            gate.set()
        job = wait_job(client, response.json()['job_id'])
        assert job['status'] == 'complete'


def test_wear_job_adds_a_variant(tmp_path, edit_io):
    from backend.api.app import create_app
    from backend import outfit as outfit_module
    adapter_files(tmp_path, outfit_module)
    with TestClient(create_app(tmp_path)) as client:
        avatar_id = imported(client)
        response = client.post(f'/api/avatars/{avatar_id}/outfit/wear/job',
                               files={'garment': ('shirt.png', png_bytes((320, 400), tag=2),
                                                  'image/png')})
        assert response.status_code == 202, response.text
        job = wait_job(client, response.json()['job_id'])
        assert job['status'] == 'complete', job
        assert job['result']['variant_id']
        person = client.get('/api/avatars/' + avatar_id).json()
        assert person['variants'][-1]['id'] == job['result']['variant_id']


def test_wear_job_rejects_a_bad_garment(tmp_path, edit_io):
    from backend.api.app import create_app
    from backend import outfit as outfit_module
    adapter_files(tmp_path, outfit_module)
    with TestClient(create_app(tmp_path)) as client:
        avatar_id = imported(client)
        response = client.post(f'/api/avatars/{avatar_id}/outfit/wear/job',
                               files={'garment': ('shirt.png', b'not-an-image', 'image/png')})
        assert response.status_code == 400
        assert '图片无法读取' in response.json()['detail']


def test_preview_serves_the_intermediate_picture(tmp_path, edit_io):
    from backend.api.app import create_app
    store = Store(tmp_path)
    with TestClient(create_app(tmp_path)) as client:
        avatar_id = imported(client)
        job = store.new_job(avatar_id, 'extend')
        folder = tmp_path/'work/previews'/job['id']
        folder.mkdir(parents=True)
        Image.fromarray(np.full((64, 48, 3), 90, np.uint8)).save(folder/'step-0007.png')
        store.update_job(job['id'], preview={'version': 2, 'step': 7, 'file': 'step-0007.png'})
        assert client.get(f'/api/jobs/{job["id"]}/preview').status_code == 200
        assert client.get(f'/api/jobs/{job["id"]}/preview?v=2').status_code == 200
        stale = client.get(f'/api/jobs/{job["id"]}/preview?v=1')
        assert stale.status_code == 409


def test_preview_is_missing_before_the_first_step(tmp_path, edit_io):
    from backend.api.app import create_app
    store = Store(tmp_path)
    with TestClient(create_app(tmp_path)) as client:
        avatar_id = imported(client)
        job = store.new_job(avatar_id, 'extend')
        assert client.get(f'/api/jobs/{job["id"]}/preview').status_code == 404
        assert client.get('/api/jobs/unknown/preview').status_code == 404
