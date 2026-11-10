import json
from pathlib import Path
import numpy as np
import pytest


def test_reply_keeps_text_and_maps_one_emotion_for_both_engines():
    from backend.emotion import Reply, parse_reply, voice_instruction, portrait_emotion
    reply=parse_reply('{"text":"太好了，祝贺你！","emotion":"happy"}')
    assert reply == Reply('太好了，祝贺你！','happy')
    assert '开心' in voice_instruction(reply.emotion)
    assert portrait_emotion(reply.emotion) == 3
    assert parse_reply('{"text":"你好。","emotion":"invalid"}').emotion == 'neutral'
    assert parse_reply('你好。') == Reply('你好。','neutral')
    with pytest.raises(ValueError):parse_reply('{"text":"","emotion":"happy"}')


def test_import_photo_profile_and_replacements_keep_sources_independent(tmp_path):
    from backend.storage import Store
    from backend.photo_profiles import replace_source
    photo=tmp_path/'photo.png';photo.write_bytes(b'photo1')
    video=tmp_path/'voice.mp4';video.write_bytes(b'video1')
    store=Store(tmp_path/'project')
    person=store.import_photo('photo person',photo,video)
    assert person['schema_version'] == 3 and person['mode'] == 'photo'
    assert 'body_photos' not in person and 'head_choice' not in person
    folder=store.avatar_dir(person['id'])
    voice=person['voice_video'];voice_hash=person['sources'][voice]
    photo.write_bytes(b'photo2')
    changed=replace_source(store,person['id'],'photo',photo)
    assert changed['voice_video'] == voice and changed['sources'][voice] == voice_hash
    assert (folder/person['photo']).read_bytes() == b'photo1'
    assert (folder/changed['photo']).read_bytes() == b'photo2'
    current_photo=changed['photo'];current_hash=changed['sources'][current_photo]
    video.write_bytes(b'video2')
    changed=replace_source(store,person['id'],'voice_video',video)
    assert changed['photo'] == current_photo and changed['sources'][current_photo] == current_hash
    assert (folder/voice).read_bytes() == b'video1'
    assert (folder/changed['voice_video']).read_bytes() == b'video2'
    assert Store(store.root).manifest(person['id']) == changed


def test_legacy_profile_uses_first_photo_and_keeps_voice_confirmation(tmp_path):
    from backend.photo_profiles import migrate_profile
    source={'schema_version':1,'id':'old','name':'existing','status':'ready','created_at':'now',
        'body_photos':['body.png'],'face_photos':['face.png'],'video':'voice.mp4',
        'sources':{'body.png':'a','face.png':'b','voice.mp4':'c'},
        'voice':{'confirmed':True,'text':'old transcript','reference':'voice/reference.wav'},
        'head_choice':{'candidate_id':'video'}}
    changed=migrate_profile(source)
    assert changed['photo'] == 'source/body.png' and changed['voice_video'] == 'source/voice.mp4'
    assert changed['voice']['confirmed'] is True
    assert changed['status'] == 'imported' and changed['validation'] == {'photo':False,'voice':False}
    assert 'head_choice' not in changed and 'body_photos' not in changed
    assert source['schema_version'] == 1


def test_voice_preparation_decodes_only_the_audio_track(tmp_path, monkeypatch):
    import av
    from backend import media
    from backend.storage import Store,sha256
    video=tmp_path/'voice.mp4'
    with av.open(str(video),'w') as output:
        picture=output.add_stream('mpeg4',rate=4);picture.width=64;picture.height=64;picture.pix_fmt='yuv420p'
        audio=output.add_stream('aac',rate=32000);audio.layout='mono'
        for _ in range(32):
            frame=av.VideoFrame.from_ndarray(np.zeros((64,64,3),np.uint8),format='rgb24')
            for packet in picture.encode(frame):output.mux(packet)
        for packet in picture.encode():output.mux(packet)
        pcm=(np.sin(np.arange(8*32000)/32000*2*np.pi*220)*6000).astype(np.int16)
        frame=av.AudioFrame.from_ndarray(pcm[None],format='s16',layout='mono');frame.sample_rate=32000
        for packet in audio.encode(frame):output.mux(packet)
        for packet in audio.encode():output.mux(packet)
    photo=tmp_path/'photo.png';photo.write_bytes(b'photo')
    store=Store(tmp_path/'project');person=store.import_photo('test',photo,video)
    original_open=media.av.open
    calls=[]
    class AudioOnly:
        def __init__(self,*args,**kwargs):self.container=original_open(*args,**kwargs)
        def __enter__(self):return self
        def __exit__(self,*args):self.container.close()
        @property
        def streams(self):return self.container.streams
        def decode(self,*args,**kwargs):
            calls.append(kwargs)
            assert kwargs == {'audio':0},'video frames were decoded'
            return self.container.decode(*args,**kwargs)
    monkeypatch.setattr(media.av,'open',AudioOnly)
    changed=media.prepare_voice(store,person['id'])
    assert calls == [{'audio':0}]
    assert changed['voice']['duration_seconds'] == 6
    assert not changed['voice']['confirmed']
    assert sha256(photo) == changed['sources'][person['photo']]
    assert (store.avatar_dir(person['id'])/'voice/reference.wav').is_file()
    changed['voice']['confirmed']=True
    changed['voice'].pop('source_sha256')
    changed['voice'].pop('reference_sha256')
    store.write_manifest(person['id'],changed)
    (store.avatar_dir(person['id'])/'voice/reference.wav').write_bytes(b'changed recording')
    assert not media.prepare_voice(store,person['id'])['voice']['confirmed']


@pytest.mark.parametrize('role', ['photo', 'voice_video'])
@pytest.mark.parametrize('damage', ['missing', 'corrupt'])
def test_same_bytes_upload_repairs_damaged_source_and_invalidates_only_its_role(tmp_path, role, damage):
    from backend.storage import Store
    from backend.photo_profiles import replace_source
    photo = tmp_path/'photo.png'; photo.write_bytes(b'original-photo')
    video = tmp_path/'voice.mp4'; video.write_bytes(b'original-video')
    store = Store(tmp_path/'project')
    person = store.import_photo('test', photo, video)
    person.update(status='ready', validation={'photo': True, 'voice': True})
    person['voice']['confirmed'] = True
    store.write_manifest(person['id'], person)
    folder = store.avatar_dir(person['id'])
    source = folder/person[role]
    if damage == 'missing':
        source.unlink()
    else:
        source.write_bytes(b'corrupt')
    incoming = photo if role == 'photo' else video
    repaired = replace_source(store, person['id'], role, incoming)
    assert (folder/repaired[role]).read_bytes() == incoming.read_bytes()
    assert repaired['status'] == 'imported'
    if role == 'photo':
        assert repaired['validation'] == {'photo': False, 'voice': True}
        assert repaired['voice'] == person['voice']
        assert repaired['voice_video'] == person['voice_video']
    else:
        assert repaired['validation'] == {'photo': True, 'voice': False}
        assert not repaired['voice']['confirmed']
        assert repaired['photo'] == person['photo']


@pytest.mark.parametrize('role', ['photo', 'voice_video'])
def test_same_bytes_upload_preserves_valid_ready_source(tmp_path, role):
    from backend.storage import Store
    from backend.photo_profiles import replace_source
    photo = tmp_path/'photo.png'; photo.write_bytes(b'photo')
    video = tmp_path/'voice.mp4'; video.write_bytes(b'voice')
    store = Store(tmp_path/'project')
    person = store.import_photo('test', photo, video)
    person.update(status='ready', validation={'photo': True, 'voice': True})
    person['voice']['confirmed'] = True
    store.write_manifest(person['id'], person)
    assert replace_source(store, person['id'], role, photo if role == 'photo' else video) == person
