from pathlib import Path
import asyncio
import threading
import httpx
from backend.emotion import parse_reply


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

    async def answer(self, text, history):
        cfg = self.config.get("dialogue", {})
        messages = [{"role": "system", "content": '你是本机照片数字人助手。用自然中文回答，每次最多两句。不要虚构照片人物的经历、身份或记忆。只返回JSON对象：{"text":"回答文字","emotion":"neutral"}。emotion只能是neutral、happy、sad、angry。按回答语气选择情感，优先自然克制，不要照搬用户的情绪。'},
                    *history[-cfg.get("context_turns", 6) * 2:], {"role": "user", "content": text}]
        client = await asyncio.to_thread(httpx.AsyncClient, timeout=120, trust_env=False)
        async with client:
            response = await client.post(cfg.get("base_url", "http://127.0.0.1:8081/v1") + "/chat/completions",
                                         json={"model": cfg.get("model", "qwen2.5-3b-instruct"),
                                               "messages": messages, "max_tokens": cfg.get("max_tokens", 128),
                                               "temperature": .6, "stream": False,
                                               "response_format": {"type": "json_object", "schema": {
                                                   "type": "object",
                                                   "properties": {
                                                       "text": {"type": "string", "minLength": 1, "maxLength": 300},
                                                       "emotion": {"type": "string", "enum": ["neutral", "happy", "sad", "angry"]}},
                                                   "required": ["text", "emotion"],
                                                   "additionalProperties": False}}})
            response.raise_for_status()
        reply = response.json()["choices"][0]["message"]["content"].strip()
        if not reply:
            raise RuntimeError("本地对话模型返回空回答")
        return parse_reply(reply)
