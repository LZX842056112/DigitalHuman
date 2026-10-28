import asyncio
import fractions
import time
import av
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, MediaStreamTrack
from aiortc import RTCRtpSender
from aiortc.mediastreams import MediaStreamError


class AvatarVideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, session, fps):
        super().__init__()
        self.session = session
        self.fps = fps
        self.start = None
        self.frames = 0
        self._pending=asyncio.Queue(maxsize=1)
        self._producer=None
        self._failure=None

    async def recv(self):
        if self.readyState!='live': raise MediaStreamError from self._failure
        if self._producer is None:
            self._producer=asyncio.create_task(self._produce(),name='avatar video renderer')
        frame=await self._pending.get()
        self.session.video_queue_depth=0
        if frame is None: raise MediaStreamError from self._failure
        return frame

    def _end_queue(self):
        if self._pending.full(): self._pending.get_nowait()
        self._pending.put_nowait(None)
        self.session.video_queue_depth=0

    def stop(self):
        super().stop()
        if self._producer is not None and not self._producer.done(): self._producer.cancel()
        self._end_queue()

    async def aclose(self):
        self.stop()
        producer=self._producer
        if producer is not None and producer is not asyncio.current_task():
            await asyncio.gather(producer,return_exceptions=True)
        self._producer=None

    async def _produce(self):
        self.start=time.monotonic();next_frame=self.start
        try:
            while self.readyState=='live' and not self.session.closed:
                await asyncio.sleep(max(0,next_frame-time.monotonic()))
                exp,jaw=self.session.timeline.expression()
                before=time.monotonic()
                image=await self.session.render_frame(exp,jaw)
                self.session.render_seconds+=time.monotonic()-before
                self.session.frames+=1
                frame=av.VideoFrame.from_ndarray(image,format='rgb24')
                frame.pts=int((time.monotonic()-self.start)*90000)
                frame.time_base=fractions.Fraction(1,90000)
                if self._pending.full():
                    self._pending.get_nowait()
                    self.session.video_frames_dropped=getattr(self.session,'video_frames_dropped',0)+1
                self._pending.put_nowait(frame)
                self.session.video_queue_depth=self._pending.qsize()
                self.frames+=1
                # Preserve cadence after a short delay. Limit recovery to one frame.
                next_frame=max(next_frame+1/self.fps,time.monotonic()-1/self.fps)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._failure=error
            emit=getattr(self.session,'emit',None)
            if callable(emit): emit('error',text=f'画面渲染失败：{error}')
        finally:
            super().stop()
            self._end_queue()


class AvatarAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, session):
        super().__init__()
        self.session, self.start, self.count = session, None, 0

    async def recv(self):
        if self.start is None:
            self.start = time.monotonic()
        await asyncio.sleep(max(0, self.start + self.count / 48000 - time.monotonic()))
        pcm, pts = self.session.timeline.read_audio(960)
        frame = av.AudioFrame.from_ndarray((np.clip(pcm, -1, 1) * 32767).astype(np.int16).reshape(1, -1),
                                          format="s16", layout="mono")
        frame.sample_rate = 48000
        frame.pts, frame.time_base = pts, fractions.Fraction(1, 48000)
        self.count += 960
        return frame


async def negotiate(session, offer, fps=15):
    # No STUN or TURN: all traffic remains on the local machine.
    peer = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
    from backend.transport.video_quality import install_local_video_encoder
    session.video_bitrate=install_local_video_encoder(getattr(session.renderer,'render_size',512))
    session.peer = peer
    session.video_track=AvatarVideoTrack(session, fps)
    peer.addTrack(session.video_track)
    peer.addTrack(AvatarAudioTrack(session))
    preferences=[codec for codec in RTCRtpSender.getCapabilities('video').codecs
                 if codec.mimeType.lower() in {'video/vp8','video/rtx'}]
    for transceiver in peer.getTransceivers():
        if transceiver.kind=='video':transceiver.setCodecPreferences(preferences)

    @peer.on("datachannel")
    def channel_open(channel):
        session.channel = channel
        channel.on("message", session.message)

    @peer.on("track")
    def input_track(track):
        if track.kind == "audio":
            asyncio.create_task(consume_microphone(session, track))

    @peer.on("connectionstatechange")
    async def state_changed():
        if peer.connectionState in {"failed", "closed"} and not session.closed:
            await session.close()

    await peer.setRemoteDescription(RTCSessionDescription(sdp=offer["sdp"], type=offer["type"]))
    await peer.setLocalDescription(await peer.createAnswer())
    return {"sdp": peer.localDescription.sdp, "type": peer.localDescription.type}


async def consume_microphone(session, track):
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
    try:
        while not session.closed:
            frame = await track.recv()
            for out in resampler.resample(frame):
                if session.recording and session.recorded_samples < 30 * 16000:
                    data = out.to_ndarray().reshape(-1).copy()
                    session.recorded.append(data)
                    session.recorded_samples += len(data)
                elif session.recording:
                    session.recording = False
                    session.emit("error", text="录音已达到30秒，请松开录音按钮")
    except Exception:
        session.recording = False
