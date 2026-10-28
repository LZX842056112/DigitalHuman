"""Detail-preserving VP8 for this project's single local WebRTC session."""
from aiortc.codecs import get_encoder as standard_encoder
from aiortc.codecs import vpx


class LocalVp8Encoder(vpx.Vp8Encoder):
    def __init__(self,bitrate=6_000_000):
        super().__init__()
        if not hasattr(self,'cfg') or not hasattr(vpx,'ffi'):
            raise RuntimeError('清晰传输需要项目固定的 aiortc 1.11.0 环境')
        self.local_minimum=int(bitrate)//2;self.local_maximum=int(bitrate)*2
        self.target_bitrate=int(bitrate)
        self.configured_size=None

    @property
    def target_bitrate(self):
        return self._Vp8Encoder__target_bitrate

    @target_bitrate.setter
    def target_bitrate(self,bitrate):
        value=max(self.local_minimum,min(int(bitrate),self.local_maximum))
        if value!=self._Vp8Encoder__target_bitrate:
            # aiortc 1.11's public setter caps every stream at 1.5 Mbps. Keep
            # its normal update path, with a per-encoder local quality budget.
            self._Vp8Encoder__target_bitrate=value
            self._Vp8Encoder__update_config_needed=True

    def encode(self,frame,force_keyframe=False):
        result=super().encode(frame,force_keyframe)
        size=(frame.width,frame.height)
        if size!=self.configured_size:
            # The upstream camera denoiser treats clothing texture as noise.
            # Disable it after codec initialization; retain realtime encoding.
            vpx.lib.vpx_codec_control_(self.codec,vpx.lib.VP8E_SET_NOISE_SENSITIVITY,vpx.ffi.cast('int',0))
            self.cfg.rc_max_quantizer=32
            self._Vp8Encoder__update_config_needed=True
            self.configured_size=size
        return result


def install_local_video_encoder(size):
    from aiortc import rtcrtpsender
    bitrate={512:4_000_000,768:6_000_000,1024:8_000_000}[size]
    def factory(codec):
        if codec.mimeType.lower()=='video/vp8':return LocalVp8Encoder(bitrate)
        return standard_encoder(codec)
    # One owned avatar session per process. Audio retains aiortc's original
    # encoder factory, and no installed package files are modified.
    rtcrtpsender.get_encoder=factory
    return bitrate
