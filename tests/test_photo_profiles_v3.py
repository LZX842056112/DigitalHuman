import json
from pathlib import Path

import numpy as np
import pytest

from backend.storage import Store, sha256


def _catalog(tmp_path):
    (tmp_path/'configs').mkdir(parents=True, exist_ok=True)
    (tmp_path/'configs/photo-models.json').write_text(json.dumps({
        'models': {'ditto': {'revision': 'a'*40}},
        'sources': {'ditto': {'commit': 'b'*40}}}))


def _cache(folder, photo_digest):
    from backend.portrait.assets import portrait_version
    cache = folder/'portrait/key'
    cache.mkdir(parents=True, exist_ok=True)
    state = cache/'features.npz'; state.write_bytes(b'state')
    idle = cache/'idle.mkv'; idle.write_bytes(b'idle')
    metadata = {'photo_sha256': photo_digest, **(portrait_version(folder.parents[2]))}
    (cache/'source.json').write_text(json.dumps({'schema_version': 1, 'file': state.name,
        'sha256': sha256(state), 'metadata': metadata}))
    (cache/'ready.json').write_text(json.dumps({'metadata': metadata,
        'idle_sha256': sha256(idle), 'width': 640, 'height': 480,
        'video_format': 'h264_rgb_lossless_v1'}))
    return {'folder': 'portrait/key', 'width': 640, 'height': 480, 'fps': 25}


def _avatar(tmp_path):
    _catalog(tmp_path)
    store = Store(tmp_path)
    folder = store.avatar_dir('person')
    (folder/'source/photo/a').mkdir(parents=True)
    (folder/'source/photo/b').mkdir(parents=True)
    (folder/'source/voice_video/c').mkdir(parents=True)
    (folder/'voice').mkdir(parents=True)
    one = folder/'source/photo/a/one.png'; one.write_bytes(b'one')
    two = folder/'source/photo/b/two.png'; two.write_bytes(b'two')
    video = folder/'source/voice_video/c/ref.mp4'; video.write_bytes(b'video')
    reference = folder/'voice/reference.wav'; reference.write_bytes(b'reference')
    portrait = _cache(folder, sha256(one))
    manifest = {
        'schema_version': 3, 'mode': 'photo', 'id': 'person', 'name': '测试',
        'created_at': '2026-01-01', 'status': 'ready',
        'settings': {'tone': 'warm', 'expression': 'smile'},
        'variants': [
            {'id': 'v1', 'label': '日常', 'photo': 'source/photo/a/one.png',
             'added_at': '2026-01-01', 'portrait': dict(portrait), 'validation': {'photo': True}},
            {'id': 'v2', 'label': '礼服', 'photo': 'source/photo/b/two.png',
             'added_at': '2026-01-01', 'portrait': None, 'validation': {'photo': False}}],
        'active_variant': 'v1',
        'photo': 'source/photo/a/one.png', 'portrait': dict(portrait),
        'voice_video': 'source/voice_video/c/ref.mp4',
        'sources': {'source/photo/a/one.png': sha256(one),
                    'source/photo/b/two.png': sha256(two),
                    'source/voice_video/c/ref.mp4': sha256(video)},
        'voice': {'confirmed': True, 'language': 'zh', 'reference': 'voice/reference.wav',
                  'source_sha256': sha256(video), 'reference_sha256': sha256(reference)},
        'validation': {'photo': True, 'voice': True},
        'artifact_hashes': {},
        'last_job_id': 'job-one'}
    (folder/'voice/sample.wav').write_bytes(b'sample')
    manifest['artifact_hashes'] = {
        name: sha256(folder/name) for name in
        ['portrait/key/source.json', 'portrait/key/features.npz',
         'portrait/key/ready.json', 'portrait/key/idle.mkv',
         'voice/reference.wav', 'voice/sample.wav']}
    store.write_manifest('person', manifest)
    return store, manifest


def test_import_creates_one_active_variant(tmp_path):
    from backend.photo_profiles import import_photo
    store = Store(tmp_path/'project')
    photo = tmp_path/'a.png'; photo.write_bytes(b'photo')
    video = tmp_path/'a.mp4'; video.write_bytes(b'video')
    manifest = import_photo(store, '新人物', photo, video)
    assert manifest['schema_version'] == 3
    assert manifest['settings'] == {'tone': 'natural', 'expression': 'natural',
                                    'tone_prompt': ''}
    assert len(manifest['variants']) == 1
    assert manifest['active_variant'] == manifest['variants'][0]['id']
    assert manifest['variants'][0]['photo'] == manifest['photo']


def test_v2_migration_is_pure_and_mirrors_the_active_variant():
    from backend.photo_profiles import migrate_profile
    source = {'schema_version': 2, 'mode': 'photo', 'id': 'old', 'name': '旧',
              'status': 'ready', 'photo': 'source/photo/a/x.png',
              'voice_video': 'source/voice_video/b/y.mp4', 'sources': {},
              'voice': {'confirmed': True}, 'validation': {'photo': True, 'voice': True},
              'portrait': {'folder': 'portrait/key', 'width': 10, 'height': 10, 'fps': 25}}
    migrated = migrate_profile(source)
    assert source['schema_version'] == 2
    assert migrated['schema_version'] == 3
    assert migrated['variants'][0]['photo'] == source['photo']
    assert migrated['portrait'] == source['portrait']
    assert migrated['settings'] == {'tone': 'natural', 'expression': 'natural',
                                    'tone_prompt': ''}
    again = migrate_profile(migrated)
    assert again == migrated and again is not migrated


def test_ready_check_ignores_an_unprepared_extra_variant(tmp_path):
    from backend.photo_profiles import require_photo_ready
    store, manifest = _avatar(tmp_path)
    assert require_photo_ready(store, 'person')['active_variant'] == 'v1'


def test_tampered_extra_variant_photo_is_rejected(tmp_path):
    from backend.photo_profiles import validate_photo_sources
    store, manifest = _avatar(tmp_path)
    folder = store.avatar_dir('person')
    (folder/'source/photo/b/two.png').write_bytes(b'changed')
    with pytest.raises(ValueError, match='来源校验失败'):
        validate_photo_sources(folder, manifest)


def test_switching_requires_a_prepared_variant(tmp_path):
    from backend.photo_profiles import set_active_variant
    store, _ = _avatar(tmp_path)
    with pytest.raises(ValueError, match='先准备'):
        set_active_variant(store, 'person', 'v2')
    assert store.manifest('person')['active_variant'] == 'v1'


def test_variant_deletion_rules(tmp_path):
    from backend.photo_profiles import delete_variant
    store, _ = _avatar(tmp_path)
    with pytest.raises(ValueError, match='使用中'):
        delete_variant(store, 'person', 'v1')
    manifest = delete_variant(store, 'person', 'v2')
    assert len(manifest['variants']) == 1
    with pytest.raises(ValueError, match='至少保留'):
        delete_variant(store, 'person', 'v1')


def test_pruned_hashes_drop_removed_variants(tmp_path):
    from backend.photo_profiles import pruned_hashes
    store, manifest = _avatar(tmp_path)
    manifest['artifact_hashes']['portrait/gone/source.json'] = 'x'
    kept = pruned_hashes(manifest)
    assert 'portrait/gone/source.json' not in kept
    assert 'portrait/key/idle.mkv' in kept
    assert 'voice/reference.wav' in kept and 'voice/sample.wav' in kept


def test_portrait_gc_keeps_shared_and_referenced_caches(tmp_path):
    from backend.photo_profiles import portrait_gc
    store, manifest = _avatar(tmp_path)
    folder = store.avatar_dir('person')
    shared = folder/'portrait/shared'; shared.mkdir()
    (shared/'source.json').write_text(json.dumps(
        {'metadata': {'photo_sha256': manifest['sources']['source/photo/b/two.png']}}))
    orphan = folder/'portrait/orphan'; orphan.mkdir()
    (orphan/'source.json').write_text(json.dumps({'metadata': {'photo_sha256': 'f'*64}}))
    (orphan/'data.bin').write_bytes(b'x'*10)
    result = portrait_gc(store, 'person')
    assert result['removed'] == ['portrait/orphan']
    assert result['freed_bytes'] > 0
    assert shared.is_dir() and (folder/'portrait/key').is_dir()


def test_variant_limit_is_enforced(tmp_path):
    from backend.photo_profiles import add_variant
    store, _ = _avatar(tmp_path)
    photo = tmp_path/'extra.png'; photo.write_bytes(b'extra')
    with pytest.raises(ValueError, match='最多保存'):
        add_variant(store, 'person', photo, limit=2)


def test_adding_a_variant_does_not_break_readiness(tmp_path):
    from backend.photo_profiles import add_variant, require_photo_ready
    store, _ = _avatar(tmp_path)
    photo = tmp_path/'extra.png'; photo.write_bytes(b'extra')
    manifest = add_variant(store, 'person', photo, '新造型')
    assert len(manifest['variants']) == 3
    assert manifest['active_variant'] == 'v1'
    assert manifest['status'] == 'ready'
    assert require_photo_ready(store, 'person')['active_variant'] == 'v1'


def test_portrait_artifacts_can_target_one_variant(tmp_path):
    from backend.photo_profiles import portrait_artifacts
    store, manifest = _avatar(tmp_path)
    names = portrait_artifacts(store.avatar_dir('person'), manifest, root=store.root,
                               variant=manifest['variants'][0])
    assert names[0] == 'portrait/key/source.json'
    with pytest.raises(ValueError, match='尚未准备'):
        portrait_artifacts(store.avatar_dir('person'), manifest, root=store.root,
                           variant=manifest['variants'][1])
