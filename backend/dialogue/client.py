"""Local chat model client: one SSE request that streams the reply as it is written."""
from pathlib import Path
import asyncio
import json
import threading

import httpx

from backend.emotion import Reply, normalize_emotion

MAX_TEXT = 300
SYSTEM_PROMPT = ('你是本机照片数字人助手。用自然中文回答，每次最多两句。'
                 '不要虚构照片人物的经历、身份或记忆。只返回JSON对象：'
                 '{"emotion":"neutral","text":"回答文字"}，先写 emotion 再写 text。'
                 'emotion只能是neutral、happy、sad、angry。按回答语气选择情感，'
                 '优先自然克制，不要照搬用户的情绪。')
REPLY_SCHEMA = {'type': 'object', 'properties': {
        'emotion': {'type': 'string', 'enum': ['neutral', 'happy', 'sad', 'angry']},
        'text': {'type': 'string', 'minLength': 1, 'maxLength': MAX_TEXT}},
    'required': ['emotion', 'text'], 'additionalProperties': False}


class ReplyStream:
    """Read emotion and text out of a JSON reply while the model still writes it.

    The schema writes emotion first, so the first sentence can be spoken while
    the rest of the answer is still being generated. Text is released as its
    characters arrive; escapes and ``\\u`` sequences are decoded first.
    """

    def __init__(self):
        self._state = 'seek'
        self._name = self._key = self._raw = ''
        self._escape = False
        self._unicode = 0
        self._emitted = 0
        self._text = []
        self._length = 0
        self._emotion = None

    def feed(self, chunk):
        out = []
        for char in chunk:
            state = self._state
            if state == 'seek':
                if char == '"':
                    self._state = 'key'
                    self._key = ''
                    self._escape = False
                    self._unicode = 0
            elif state == 'key':
                if self._unicode:
                    self._key += char
                    self._unicode -= 1
                elif self._escape:
                    self._key += char
                    self._escape = False
                    if char == 'u':
                        self._unicode = 4
                elif char == '\\':
                    self._key += char
                    self._escape = True
                elif char == '"':
                    self._name = json.loads('"' + self._key + '"') if '\\' in self._key else self._key
                    self._key = ''
                    self._state = 'colon'
                else:
                    self._key += char
            elif state == 'colon':
                if char == ':':
                    self._state = 'value'
                elif not char.isspace():
                    self._state = 'seek'
            elif state == 'value':
                if char == '"':
                    self._state = 'string'
                    self._raw = ''
                    self._escape = False
                    self._unicode = 0
                    self._emitted = 0
                elif not char.isspace():
                    # Numbers, booleans and nested objects are not streamed.
                    self._state = 'seek'
            elif state == 'string':
                if self._unicode:
                    self._raw += char
                    self._unicode -= 1
                elif self._escape:
                    self._raw += char
                    self._escape = False
                    if char == 'u':
                        self._unicode = 4
                elif char == '\\':
                    self._raw += char
                    self._escape = True
                elif char == '"':
                    out.extend(self._close())
                else:
                    self._raw += char
        if self._state == 'string':
            out.extend(self._release())
        return out

    def _accept(self, value):
        start = self._emitted
        self._emitted = len(value)
        if start >= len(value) or self._length >= MAX_TEXT:
            return None
        delta = value[start:start + MAX_TEXT - self._length]
        if not delta:
            return None
        self._length += len(delta)
        self._text.append(delta)
        return {'type': 'text', 'delta': delta}

    def _release(self):
        if self._name != 'text':
            return []
        try:
            value = json.loads('"' + self._raw + '"')
        except ValueError:
            # An escape or \u sequence is still incomplete; wait for more bytes.
            return []
        event = self._accept(value)
        return [event] if event else []

    def _close(self):
        name, raw = self._name, self._raw
        self._raw = ''
        self._state = 'seek'
        value = json.loads('"' + raw + '"') if '\\' in raw else raw
        if name == 'emotion':
            self._emotion = normalize_emotion(value)
            return [{'type': 'emotion', 'emotion': self._emotion}]
        if name == 'text':
            event = self._accept(value)
            return [event] if event else []
        return []

    @property
    def finished(self):
        """True once the JSON text value is closed, even if there are no more chunks."""
        return self._state == 'seek' and bool(self._text)

    def reply(self):
        text = ''.join(self._text).strip()
        if not text:
            raise ValueError('对话模型没有返回有效回答')
        return Reply(text, normalize_emotion(self._emotion))


class Dialogue:
    def __init__(self, root: Path, config):
        self.root = root
        self.config = config
        self._recognizer = None
        self._recognizer_lock = threading.Lock()

    def transcribe(self, audio):
        if self._recognizer is None:
            with self._recognizer_lock:
                if self._recognizer is None:
                    from faster_whisper import WhisperModel
                    model = self.root / self.config.get("asr", {}).get("model", "models/asr/small")
                    if not (model / "model.bin").exists():
                        raise RuntimeError("语音识别模型尚未安装，请运行 setup.ps1")
                    self._recognizer = WhisperModel(str(model), device="cpu", compute_type="int8",
                                                    cpu_threads=4, local_files_only=True)
        segments, _ = self._recognizer.transcribe(audio, language="zh", vad_filter=True, beam_size=3)
        return "".join(segment.text for segment in segments).strip()

    def _payload(self, text, history):
        cfg = self.config.get("dialogue", {})
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    *history[-cfg.get("context_turns", 6) * 2:], {"role": "user", "content": text}]
        return cfg, {"model": cfg.get("model", "qwen2.5-3b-instruct"), "messages": messages,
                     "max_tokens": cfg.get("max_tokens", 128), "temperature": .6, "stream": True,
                     "response_format": {"type": "json_object", "schema": REPLY_SCHEMA}}

    async def stream(self, text, history):
        """Yield emotion/text events and a final Reply without waiting for the whole answer.

        Servers that ignore ``stream`` and return one JSON body are accepted too.
        """
        cfg, payload = self._payload(text, history)
        url = cfg.get("base_url", "http://127.0.0.1:8081/v1") + "/chat/completions"
        client = await asyncio.to_thread(httpx.AsyncClient, timeout=120, trust_env=False)
        reader = ReplyStream()
        async with client:
            async with client.stream('POST', url, json=payload) as response:
                if response.is_error:
                    await response.aread()
                    response.raise_for_status()
                kind = response.headers.get('content-type', '')
                if 'text/event-stream' not in kind:
                    body = await response.aread()
                    content = json.loads(body)["choices"][0]["message"]["content"].strip()
                    if not content:
                        raise RuntimeError("本地对话模型返回空回答")
                    for event in reader.feed(content):
                        yield event
                    yield {'type': 'done', 'reply': reader.reply()}
                    return
                done = False
                async for line in response.aiter_lines():
                    if not line or line.startswith((':', 'event:')):
                        continue
                    if not line.startswith('data:'):
                        continue
                    data = line[5:].strip()
                    if data == '[DONE]':
                        done = True
                        break
                    if len(data) > 65536:
                        raise ValueError('对话模型事件超过长度上限')
                    try:
                        chunk = json.loads(data)
                    except ValueError:
                        continue
                    choices = chunk.get('choices') or []
                    if not choices:
                        continue
                    content = (choices[0].get('delta') or {}).get('content')
                    if not content:
                        continue
                    for event in reader.feed(content):
                        yield event
                if not done and not reader.finished:
                    raise RuntimeError('对话流在完成前断开')
                yield {'type': 'done', 'reply': reader.reply()}

    async def answer(self, text, history):
        reply = None
        async for event in self.stream(text, history):
            if event['type'] == 'done':
                reply = event['reply']
        if reply is None:
            raise RuntimeError("本地对话模型返回空回答")
        return reply
