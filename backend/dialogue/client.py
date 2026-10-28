from pathlib import Path
import httpx


class Dialogue:
    def __init__(self, root: Path, config):
        self.root = root
        self.config = config
        self._recognizer = None

    def transcribe(self, audio):
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
        messages = [{"role": "system", "content": "你是本机数字人助手。用自然中文回答，每次最多三句。不要虚构素材中人物的经历、身份或记忆。"},
                    *history[-cfg.get("context_turns", 6) * 2:], {"role": "user", "content": text}]
        async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
            response = await client.post(cfg.get("base_url", "http://127.0.0.1:8081/v1") + "/chat/completions",
                                         json={"model": cfg.get("model", "qwen2.5-3b-instruct"),
                                               "messages": messages, "max_tokens": cfg.get("max_tokens", 128),
                                               "temperature": .6, "stream": False})
            response.raise_for_status()
        reply = response.json()["choices"][0]["message"]["content"].strip()
        if not reply:
            raise RuntimeError("本地对话模型返回空回答")
        return reply[:300]
