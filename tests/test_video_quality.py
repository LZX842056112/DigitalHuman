import numpy as np
import av
import pytest
from aiortc.codecs import vpx
from aiortc.codecs.vpx import Vp8Decoder
from backend.transport.video_quality import LocalVp8Encoder

pytestmark=pytest.mark.skipif(not hasattr(vpx,'ffi'),reason='Run in the pinned WSL aiortc 1.11 environment')


def test_local_encoder_keeps_fine_luminance_detail_after_codec_roundtrip():
    from fractions import Fraction
    x,y=np.meshgrid(np.arange(512),np.arange(512))
    values=(110+30*np.sin(x*1.2)+25*np.cos(y*.8)+20*((x//3+y//3)%2)).clip(0,255).astype(np.uint8)
    rgb=np.repeat(values[...,None],3,axis=2)
    frame=av.VideoFrame.from_ndarray(rgb,format='rgb24');frame.time_base=Fraction(1,90000)
    encoder=LocalVp8Encoder(6_000_000);decoder=Vp8Decoder()
    decoded=None
    for index in range(4):
        frame.pts=index*6000
        payloads,timestamp=encoder.encode(frame)
        # Remove the VP8 RTP descriptors before passing a complete encoded
        # picture to the decoder, as aiortc's jitter buffer normally does.
        from aiortc.codecs.vpx import VpxPayloadDescriptor
        from aiortc.jitterbuffer import JitterFrame
        data=b''.join(VpxPayloadDescriptor.parse(payload)[1] for payload in payloads)
        output=decoder.decode(JitterFrame(data=data,timestamp=timestamp))
        if output:decoded=output[-1].to_ndarray(format='rgb24')
    assert decoded is not None
    error=np.mean(np.abs(decoded.astype(np.float32)-rgb.astype(np.float32)))
    assert error<3.,f'Fine detail was lost during video transmission: MAE={error}'
    encoder.target_bitrate=100_000
    assert encoder.target_bitrate>=3_000_000


def test_local_video_policy_keeps_the_original_audio_encoder(monkeypatch):
    from aiortc import rtcrtpsender
    from aiortc.rtcrtpparameters import RTCRtpCodecParameters
    from backend.transport.video_quality import install_local_video_encoder
    original=rtcrtpsender.get_encoder
    monkeypatch.setattr(rtcrtpsender,'get_encoder',original)
    install_local_video_encoder(768)
    audio=rtcrtpsender.get_encoder(RTCRtpCodecParameters(mimeType='audio/opus',clockRate=48000,channels=2))
    video=rtcrtpsender.get_encoder(RTCRtpCodecParameters(mimeType='video/VP8',clockRate=90000))
    assert type(audio).__name__=='OpusEncoder'
    assert isinstance(video,LocalVp8Encoder) and video.target_bitrate==6_000_000
