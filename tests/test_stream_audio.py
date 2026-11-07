import numpy as np
import pytest


def test_continuous_resampling_matches_single_input_including_tail():
    from backend.stream_audio import StreamingResampler
    pcm=np.random.default_rng(7).normal(0,.03,31017).astype(np.float32)
    def convert(blocks):
        resampler=StreamingResampler(22050,48000)
        return np.concatenate([*(resampler.push(block) for block in blocks),resampler.finish()])
    expected=convert([pcm])
    actual=convert(np.array_split(pcm,37))
    assert len(actual)==len(expected)
    np.testing.assert_array_equal(actual,expected)
    assert abs(len(actual)-len(pcm)*48000/22050)<=1


def test_stream_gain_is_frozen_and_limiting_is_chunk_invariant():
    from backend.stream_audio import SpeechNormalizer
    rate=24000
    prefix=np.full(rate,.05,np.float32)
    tail=np.concatenate([np.full(1000,.1,np.float32),np.full(1000,-2,np.float32),np.zeros(300,np.float32)])
    def normalized(parts):
        normalizer=SpeechNormalizer(rate)
        values=[normalizer.push(prefix)]
        gain=normalizer.gain
        values.extend(normalizer.push(part) for part in parts)
        values.append(normalizer.finish())
        assert normalizer.gain==gain
        return np.concatenate(values)
    whole=normalized([tail]);split=normalized(np.array_split(tail,17))
    np.testing.assert_allclose(whole,split,atol=1e-7)
    assert len(whole)==len(prefix)+len(tail)
    assert np.max(np.abs(whole))<=.891001
    assert np.all(np.isfinite(whole))


def test_short_speech_flushes_once_and_invalid_pcm_is_rejected():
    from backend.stream_audio import SpeechNormalizer,StreamingResampler
    normalizer=SpeechNormalizer(24000)
    first=normalizer.push(np.full(100,.05,np.float32))
    tail=normalizer.finish()
    assert len(first)+len(tail)==100
    assert len(normalizer.finish())==0
    with pytest.raises(ValueError):SpeechNormalizer(24000).push(np.array([np.nan]))
    with pytest.raises(ValueError):StreamingResampler(0,48000)
