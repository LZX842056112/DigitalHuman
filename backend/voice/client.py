import io
import wave
import httpx
import numpy as np
from backend.errors import NotReady


def normalize_speech(pcm, target_rms=.1, peak_limit=.891):
    """Apply one gain to the utterance; keep silence, timing and pitch intact."""
    pcm=np.asarray(pcm,dtype=np.float32).reshape(-1)
    if not len(pcm) or not np.isfinite(pcm).all():
        raise ValueError('声音数据为空或含无效数值')
    peak=float(np.max(np.abs(pcm)))
    if peak<1e-5: raise ValueError('声音数据没有可播放的语音')
    # Ignore silence when estimating loudness. Do not apply per-sample AGC,
    # which would amplify pauses or change the speaker's volume dynamics.
    active=pcm[np.abs(pcm)>peak*.05].astype(np.float64)
    rms=float(np.sqrt(np.mean(active**2)))
    gain=min(target_rms/rms,peak_limit/peak)
    return (pcm*gain).astype(np.float32)


class VoiceClient:
    def __init__(self, base_url="http://127.0.0.1:9880"):
        self.base_url = base_url

    async def synthesize(self, text, avatar_dir, profile):
        if not profile.get("confirmed") or not profile.get("text", "").strip():
            raise NotReady("请先试听参考录音并核对文字")
        request = {"text": text, "text_lang": "zh", "ref_audio_path": str(avatar_dir / "voice/reference.wav"),
                   "prompt_lang": profile.get("language", "zh"), "prompt_text": profile["text"],
                   "media_type": "wav", "streaming_mode": False, "batch_size": 1,
                   "text_split_method": "cut5", "speed_factor": 1.0}
        async with httpx.AsyncClient(timeout=180, trust_env=False) as client:
            response = await client.post(f"{self.base_url}/tts", json=request)
            response.raise_for_status()
        with wave.open(io.BytesIO(response.content)) as wav:
            if wav.getsampwidth() != 2:
                raise RuntimeError("声音服务未返回16位PCM")
            rate = wav.getframerate()
            pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
            pcm = pcm.reshape(-1, wav.getnchannels()).mean(axis=1)
        if not len(pcm) or np.max(np.abs(pcm)) < 1e-4:
            raise RuntimeError("声音服务返回空音频")
        return normalize_speech(pcm), rate
