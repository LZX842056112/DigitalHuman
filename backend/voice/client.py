import io
import asyncio
import wave
import httpx
import numpy as np
from backend.errors import NotReady
from backend.emotion import normalize_emotion


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

    async def synthesize(self, text, avatar_dir, profile, *, emotion='neutral', session_id=None, generation=None):
        if not profile.get("confirmed"):
            raise NotReady("请先试听并确认参考声音")
        request = {"text": text, "avatar_id": avatar_dir.name,"emotion":normalize_emotion(emotion)}
        if session_id is not None:request.update(session_id=session_id,generation=generation)
        client = await asyncio.to_thread(httpx.AsyncClient, timeout=600, trust_env=False)
        async with client:
            response = await client.post(f"{self.base_url}/tts", json=request)
            if response.is_error:raise RuntimeError('声音服务：'+response.text[:500])
        with wave.open(io.BytesIO(response.content)) as wav:
            if wav.getsampwidth() != 2:
                raise RuntimeError("声音服务未返回16位PCM")
            rate = wav.getframerate()
            pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
            pcm = pcm.reshape(-1, wav.getnchannels()).mean(axis=1)
        if not len(pcm) or np.max(np.abs(pcm)) < 1e-4:
            raise RuntimeError("声音服务返回空音频")
        return normalize_speech(pcm), rate
