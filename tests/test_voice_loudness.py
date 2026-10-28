import numpy as np
import pytest
from backend.voice.client import normalize_speech


def test_quiet_synthesized_voice_reaches_audible_level_without_changing_pitch():
    t=np.arange(32000,dtype=np.float32)/32000
    source=(np.sin(2*np.pi*220*t)*.001).astype(np.float32)
    output=normalize_speech(np.concatenate([np.zeros(16000,np.float32),source]))
    assert .07<np.sqrt(np.mean(output[16000:]**2))<.13
    assert np.max(np.abs(output))<.90
    np.testing.assert_array_equal(np.signbit(output[16000:]),np.signbit(source))
    assert not output[:16000].any()


def test_loud_voice_is_limited_by_peak_without_hard_clipping():
    source=np.full(32000,.2,np.float32);source[16000]=1.
    output=normalize_speech(source)
    assert np.max(np.abs(output))<=.892
    np.testing.assert_allclose(output/source,np.full_like(source,output[0]/source[0]),rtol=1e-5)


@pytest.mark.parametrize('source',[np.zeros(1000,np.float32),np.array([np.nan],np.float32)])
def test_empty_or_invalid_voice_is_rejected_instead_of_amplified(source):
    with pytest.raises(ValueError):normalize_speech(source)


async def test_voice_http_response_is_normalized_before_expression_and_playback(tmp_path,monkeypatch):
    import io,wave,httpx
    from backend.voice.client import VoiceClient
    values=(np.sin(np.arange(32000)*2*np.pi*220/32000)*50).astype('<i2')
    buffer=io.BytesIO()
    with wave.open(buffer,'wb') as wav:
        wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(32000);wav.writeframes(values.tobytes())
    transport=httpx.MockTransport(lambda request:httpx.Response(200,content=buffer.getvalue()))
    original=httpx.AsyncClient
    monkeypatch.setattr(httpx,'AsyncClient',lambda **kwargs:original(transport=transport,**kwargs))
    pcm,rate=await VoiceClient().synthesize('你好',tmp_path,{'confirmed':True,'text':'参考文字'})
    assert rate==32000 and np.sqrt(np.mean(pcm**2))>.07
