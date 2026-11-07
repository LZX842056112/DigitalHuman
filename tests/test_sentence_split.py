import asyncio
import json
from types import SimpleNamespace

import pytest

from test_photo_session import clients, session, until


@pytest.mark.parametrize('text,expected', [
    ('第一句。第二句。', '第一句。第二句。'),
    ('你好！！真的吗？？', '你好！！真的吗？？'),
    ('他说：“走吧。”然后离开。', '他说：“走吧。”然后离开。'),
    ('没有标点的一句话', '没有标点的一句话'),
    ('第一行\n第二行。第三行', '第一行第二行。第三行'),
    ('换行结尾\n', '换行结尾'),
])
def test_split_sentences_keeps_every_word(text, expected):
    from backend.transport.photo_session import split_sentences
    sentences, rest = split_sentences(text)
    assert rest == ''
    assert sentences and all(sentences)
    assert ''.join(sentences) == expected


def test_leading_punctuation_is_not_spoken():
    from backend.transport.photo_session import split_sentences
    assert split_sentences('。你好。')[0] == ['你好。']


def test_streamed_sentences_match_one_batch_split():
    from backend.transport.photo_session import SentenceSplitter, split_sentences
    text = '第一句。第二句！！第三句？最后一句话没有句号'
    whole, _ = split_sentences(text)
    splitter = SentenceSplitter(0)
    streamed = []
    for index in range(0, len(text), 2):
        streamed.extend(splitter.feed(text[index:index + 2]))
    streamed.extend(splitter.flush())
    assert streamed == whole


def test_only_the_first_sentence_is_cut_at_a_comma():
    from backend.transport.photo_session import SentenceSplitter
    splitter = SentenceSplitter(8)
    assert splitter.feed('今天天气很好，我们一起') == ['今天天气很好，']
    assert splitter.feed('去公园吧。明天') == ['我们一起去公园吧。']
    assert splitter.feed('也行，真的。好') == ['明天也行，真的。']
    assert splitter.flush() == ['好']


@pytest.mark.asyncio
async def test_sentences_are_spoken_while_the_model_is_still_writing(tmp_path):
    from backend.emotion import Reply
    instance, events = session(tmp_path)
    calls = clients(instance)
    release = asyncio.Event()

    async def stream(text, history):
        yield {'type': 'emotion', 'emotion': 'happy'}
        yield {'type': 'text', 'delta': '第一句。'}
        yield {'type': 'text', 'delta': '第二'}
        await release.wait()
        yield {'type': 'text', 'delta': '句。'}
        yield {'type': 'done', 'reply': Reply('第一句。第二句。', 'happy')}

    instance.dialogue = SimpleNamespace(stream=stream)
    instance.message(json.dumps({'type': 'chat_text', 'text': '说两句'}))
    try:
        await until(lambda: any(call[0] == 'voice' for call in calls) or instance.task.done())
        first = [call for call in calls if call[0] == 'voice']
        assert len(first) == 1, events
        assert any(event['type'] == 'reply_partial' and event['text'] == '第一句。' for event in events)
        assert not any(event.get('role') == 'assistant' for event in events)
        release.set()
        await until(lambda: any(event.get('role') == 'assistant' for event in events))
        assert [event['text'] for event in events if event.get('role') == 'assistant'] == ['第一句。第二句。']
        await until(lambda: len([call for call in calls if call[0] == 'voice']) == 2)
    finally:
        release.set()
        await instance.close()


@pytest.mark.asyncio
async def test_interrupted_reply_never_publishes_a_partial_transcript(tmp_path):
    from backend.emotion import Reply
    instance, events = session(tmp_path)
    clients(instance)
    release = asyncio.Event()

    async def stream(text, history):
        yield {'type': 'emotion', 'emotion': 'neutral'}
        yield {'type': 'text', 'delta': '第一句。'}
        yield {'type': 'text', 'delta': '第二'}
        await release.wait()
        yield {'type': 'text', 'delta': '句。'}
        yield {'type': 'done', 'reply': Reply('第一句。第二句。', 'neutral')}

    instance.dialogue = SimpleNamespace(stream=stream)
    instance.message(json.dumps({'type': 'chat_text', 'text': '说两句'}))
    try:
        await until(lambda: any(event['type'] == 'reply_partial' for event in events))
        instance.interrupt()
        release.set()
        await asyncio.sleep(.05)
        assert not any(event.get('role') == 'assistant' for event in events)
    finally:
        release.set()
        await instance.close()
