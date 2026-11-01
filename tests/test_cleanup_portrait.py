"""Keep portrait metadata and existing cache directory identities stable."""
import hashlib
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.portrait.assets import portrait_version
from backend.storage import sha256


@pytest.fixture
def portrait_catalog(tmp_path):
    catalog = {'models': {'ditto': {'revision': 'a'*40}},
               'sources': {'ditto': {'commit': 'b'*40}}}
    (tmp_path/'configs').mkdir()
    (tmp_path/'configs/photo-models.json').write_text(json.dumps(catalog))
    return catalog


def test_portrait_version_uses_the_current_catalog(tmp_path, portrait_catalog):
    assert portrait_version(tmp_path) == {
        'model_revision': 'a'*40, 'code_commit': 'b'*40, 'max_size': 1024}
    portrait_catalog['models']['ditto']['revision'] = 'c'*40
    portrait_catalog['sources']['ditto']['commit'] = 'd'*40
    (tmp_path/'configs/photo-models.json').write_text(json.dumps(portrait_catalog))
    assert portrait_version(tmp_path) == {
        'model_revision': 'c'*40, 'code_commit': 'd'*40, 'max_size': 1024}


def test_prepare_keeps_the_existing_cache_key_and_response(tmp_path, portrait_catalog, monkeypatch):
    from backend.portrait import service
    folder = tmp_path/'person'
    folder.mkdir()
    photo = folder/'photo.png'
    photo.write_bytes(b'fixture photo')
    legacy_identity = {'photo': sha256(photo), 'model': 'a'*40, 'code': 'b'*40, 'size': 1024}
    key = hashlib.sha256(json.dumps(legacy_identity, sort_keys=True).encode()).hexdigest()[:24]
    calls = []

    def prepare(image, target):
        calls.append((image, target))
        return {'width': 640, 'height': 480}

    monkeypatch.setattr(service, 'ROOT', tmp_path)
    monkeypatch.setattr(service, 'profile', lambda _: ({}, folder, photo))
    monkeypatch.setattr(service, 'engine', SimpleNamespace(prepare=prepare))
    with TestClient(service.app) as client:
        response = client.post('/prepare', json={'avatar_id': 'person'})
    assert response.status_code == 200
    assert response.json() == {'folder': 'portrait/'+key, 'width': 640, 'height': 480, 'fps': 25}
    assert calls == [(photo, folder/'portrait'/key)]


@pytest.mark.parametrize('extra_metadata', [False, True])
def test_portrait_readiness_keeps_strict_metadata_matching(tmp_path, portrait_catalog, extra_metadata):
    from backend.photo_profiles import portrait_artifacts
    cache = tmp_path/'portrait/key'
    cache.mkdir(parents=True)
    state = cache/'features.npz'
    state.write_bytes(b'fixture feature bytes')
    idle = cache/'idle.mkv'
    idle.write_bytes(b'fixture video bytes')
    metadata = {'photo_sha256': 'e'*64, **portrait_version(tmp_path)}
    if extra_metadata:
        metadata['extra'] = True
    (cache/'source.json').write_text(json.dumps({
        'schema_version': 1, 'file': state.name, 'sha256': sha256(state), 'metadata': metadata}))
    (cache/'ready.json').write_text(json.dumps({
        'metadata': metadata, 'idle_sha256': sha256(idle), 'width': 640, 'height': 480}))
    manifest = {'photo': 'photo.png', 'sources': {'photo.png': 'e'*64},
                'portrait': {'folder': 'portrait/key', 'width': 640, 'height': 480, 'fps': 25}}
    if extra_metadata:
        with pytest.raises(ValueError, match='版本不一致'):
            portrait_artifacts(tmp_path, manifest, root=tmp_path)
    else:
        assert portrait_artifacts(tmp_path, manifest, root=tmp_path) == [
            'portrait/key/source.json', 'portrait/key/features.npz',
            'portrait/key/ready.json', 'portrait/key/idle.mkv']
