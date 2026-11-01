import io,json,wave
from pathlib import Path
import httpx
import numpy as np
import pytest


def transport_client(monkeypatch,handler):
    original=httpx.AsyncClient
    transport=httpx.MockTransport(handler)
    def client(*args,**kwargs):return original(*args,transport=transport,**kwargs)
    monkeypatch.setattr(httpx,'AsyncClient',client)


@pytest.mark.asyncio
async def test_dialogue_returns_text_with_one_bounded_emotion(tmp_path,monkeypatch):
    from backend.dialogue.client import Dialogue
    from backend.emotion import Reply
    def answer(request):
        payload=json.loads(request.content)
        assert payload['response_format']['type']=='json_object'
        return httpx.Response(200,json={'choices':[{'message':{'content':'{"text":"祝贺你！","emotion":"happy"}'}}]})
    transport_client(monkeypatch,answer)
    reply=await Dialogue(tmp_path,{}).answer('我今天很开心',[])
    assert reply == Reply('祝贺你！','happy')


@pytest.mark.asyncio
async def test_dialogue_request_requires_answer_text_and_bounded_emotion(tmp_path, monkeypatch):
    from backend.dialogue.client import Dialogue
    from backend.emotion import Reply
    requests = []

    def answer(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={'choices': [{'message': {
            'content': '{"text":"保持每天学习。","emotion":"neutral"}'}}]})

    transport_client(monkeypatch, answer)
    reply = await Dialogue(tmp_path, {}).answer('请告诉我保持学习习惯的方法。', [
        {'role': 'user', 'content': '请介绍春天。'},
        {'role': 'assistant', 'content': '春天万物复苏。'}])
    assert reply == Reply('保持每天学习。', 'neutral')
    schema = requests[0]['response_format'].get('schema', {})
    assert schema.get('type') == 'object'
    assert set(schema.get('required', [])) == {'text', 'emotion'}, 'An empty JSON object must be invalid'
    assert schema['properties']['text']['type'] == 'string'
    assert schema['properties']['text']['minLength'] >= 1
    assert schema['properties']['text']['maxLength'] <= 300
    assert set(schema['properties']['emotion']['enum']) == {'neutral', 'happy', 'sad', 'angry'}
    assert schema.get('additionalProperties') is False


@pytest.mark.asyncio
async def test_voice_client_accepts_confirmed_audio_without_reference_transcript(tmp_path,monkeypatch):
    from backend.voice.client import VoiceClient
    output=io.BytesIO()
    with wave.open(output,'wb') as wav:
        wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(24000)
        wav.writeframes((np.sin(np.arange(12000)*.1)*3000).astype('<i2').tobytes())
    def speak(request):
        payload=json.loads(request.content)
        assert payload['avatar_id']=='person' and payload['emotion']=='happy'
        assert payload['text']=='你好。'
        return httpx.Response(200,content=output.getvalue(),headers={'content-type':'audio/wav'})
    transport_client(monkeypatch,speak)
    pcm,rate=await VoiceClient().synthesize('你好。',tmp_path/'person',{'confirmed':True},emotion='happy')
    assert rate == 24000 and len(pcm)==12000 and np.max(np.abs(pcm))>.05
