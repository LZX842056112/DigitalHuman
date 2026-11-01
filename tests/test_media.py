import wave
from pathlib import Path
import numpy as np
import pytest


def test_reference_selection_avoids_silence_and_clipping():
    from backend.media import choose_reference
    sr = 16000
    t = np.arange(sr * 6) / sr
    speech = (.15 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    pcm = np.concatenate([np.zeros(sr * 7), speech, np.ones(sr * 7)])
    start, duration = choose_reference(pcm, sr, 6)
    assert 6.5 <= start <= 7.5
    assert duration == 6


def test_reference_selection_rejects_silence():
    from backend.media import choose_reference
    from backend.errors import PipelineError
    with pytest.raises(PipelineError, match="语音"):
        choose_reference(np.zeros(16000 * 12), 16000)


def test_audio_decode_resamples_without_mutating_source(tmp_path):
    from backend.media import read_audio, write_wav
    src = tmp_path / "voice.wav"
    with wave.open(str(src), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(8000)
        f.writeframes(np.full(8000, 1000, dtype="<i2").tobytes())
    original = src.read_bytes()
    pcm, sr = read_audio(src, 16000)
    assert sr == 16000
    assert abs(len(pcm) - 16000) <= 2
    assert src.read_bytes() == original
    write_wav(tmp_path / "copy.wav", pcm, sr)
    with wave.open(str(tmp_path / "copy.wav")) as f:
        assert f.getframerate() == 16000
