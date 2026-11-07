import json

import httpx
import pytest

from backend.emotion import Reply


def transport(monkeypatch, handler):
    original = httpx.AsyncClient
    mock = httpx.MockTransport(handler)

    def client(*args, **kwargs):
        return original(*args, transport=mock, **kwargs)

    monkeypatch.setattr(httpx, 'AsyncClient', client)


def extract(chunks):
    from backend.dialogue.client import ReplyStream
    reader = ReplyStream()
    events = []
    for chunk in chunks:
        events.extend(reader.feed(chunk))
    return events, reader


@pytest.mark.parametrize('payload', [
    '{"emotion":"happy","text":"祝贺你！"}',
    '{"text":"保持每天学习。","emotion":"neutral"}',
    '{"emotion":"sad","text":"他说：\\"别走\\"。然后离开。"}',
    '{"emotion":"neutral","text":"你好\\u4e16界"}',
])
@pytest.mark.parametrize('size', [1, 2, 3, 7])
def test_reply_stream_rebuilds_text_from_any_chunk_boundary(payload, size):
    events, reader = extract([payload[i:i + size] for i in range(0, len(payload), size)])
    text = ''.join(event['delta'] for event in events if event['type'] == 'text')
    assert text == reader.reply().text
    assert reader.reply().text
    assert [event['emotion'] for event in events if event['type'] == 'emotion'] == [reader.reply().emotion]


def test_reply_stream_does_not_confuse_a_quoted_key_with_a_field():
    events, reader = extract(['{"emotion":"neutral","text":"他说','\\"text\\"','是这个意思。"}'])
    assert reader.reply() == Reply('他说"text"是这个意思。', 'neutral')
    assert ''.join(event['delta'] for event in events if event['type'] == 'text') == '他说"text"是这个意思。'


@pytest.mark.asyncio
async def test_dialogue_streams_emotion_before_text_and_one_final_reply(tmp_path, monkeypatch):
    from backend.dialogue.client import Dialogue
    payloads = []

    def answer(request):
        payloads.append(json.loads(request.content))
        chunks = ['{"emotion":', '"happy"', ',"text":"祝', '贺你！"}']
        body = ''.join('data: ' + json.dumps({'choices': [{'delta': {'content': chunk}}]},
                                             ensure_ascii=False) + '\n\n' for chunk in chunks)
        return httpx.Response(200, content=(body + 'data: [DONE]\n\n').encode(),
                              headers={'content-type': 'text/event-stream'})

    transport(monkeypatch, answer)
    events = [event async for event in Dialogue(tmp_path, {}).stream('我今天很开心', [])]
    kinds = [event['type'] for event in events]
    assert payloads[0]['stream'] is True
    assert payloads[0]['response_format']['type'] == 'json_object'
    assert list(payloads[0]['response_format']['schema']['properties']) == ['emotion', 'text']
    assert kinds.count('done') == 1 and kinds[-1] == 'done'
    assert kinds.index('emotion') < kinds.index('text')
    assert events[-1]['reply'] == Reply('祝贺你！', 'happy')
    assert ''.join(event['delta'] for event in events if event['type'] == 'text') == '祝贺你！'


@pytest.mark.asyncio
async def test_dialogue_accepts_a_server_that_ignores_stream(tmp_path, monkeypatch):
    from backend.dialogue.client import Dialogue

    def answer(request):
        assert json.loads(request.content)['stream'] is True
        return httpx.Response(200, json={'choices': [{'message': {
            'content': '{"emotion":"neutral","text":"保持每天学习。"}'}}]})

    transport(monkeypatch, answer)
    reply = await Dialogue(tmp_path, {}).answer('请告诉我保持学习习惯的方法。', [])
    assert reply == Reply('保持每天学习。', 'neutral')


@pytest.mark.asyncio
async def test_dialogue_rejects_a_stream_that_ends_mid_sentence(tmp_path, monkeypatch):
    from backend.dialogue.client import Dialogue

    def answer(request):
        body = 'data: ' + json.dumps({'choices': [{'delta': {'content': '{"emotion":"neutral","text":"说了一半'}}]}) + '\n\n'
        return httpx.Response(200, content=body.encode(), headers={'content-type': 'text/event-stream'})

    transport(monkeypatch, answer)
    with pytest.raises(RuntimeError, match='断开'):
        await Dialogue(tmp_path, {}).answer('问题', [])


@pytest.mark.asyncio
async def test_dialogue_surfaces_http_errors_before_streaming(tmp_path, monkeypatch):
    from backend.dialogue.client import Dialogue

    def answer(request):
        return httpx.Response(400, json={'error': 'bad request'})

    transport(monkeypatch, answer)
    with pytest.raises(httpx.HTTPStatusError):
        await Dialogue(tmp_path, {}).answer('问题', [])
