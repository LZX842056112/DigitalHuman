"""Local photo transport with native audio and one pending video frame."""
import asyncio
import fractions
import time

import av
import numpy as np
from backend import clock
from aiortc import MediaStreamTrack, RTCConfiguration, RTCPeerConnection, RTCRtpSender, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError


class PhotoVideoTrack(MediaStreamTrack):
    kind = 'video'

    def __init__(self, session, fps=25):
        super().__init__()
        self.session, self.fps = session, fps
        self._pending = asyncio.Queue(maxsize=1)
        self._producer = self._reading = self._failure = None
        self._refresh = asyncio.Event()
        self._refresh_slot = None
        self._space = asyncio.Event()
        self._space.set()
        self._queued_at = 0.
        self._last_sent_pts = -1
        self.send_at = 0.
        self.drop_reasons = {name: 0 for name in ('stale_queued_generation', 'cleared_pending',
            'stale_decoded_generation', 'pcm_retry', 'queue_replaced')}

    def _drop(self, reason):
        self.session.video_frames_dropped += 1
        self.drop_reasons[reason] += 1

    async def recv(self):
        if self.readyState != 'live' or self.session.closed:
            raise MediaStreamError from self._failure
        if self._producer is None:
            self._producer = asyncio.create_task(self._produce(), name='photo video decoder')
        while self.readyState == 'live' and not self.session.closed:
            item = await self._pending.get()
            self._space.set()
            self.session.video_queue_depth = 0
            if item is None:
                raise MediaStreamError from self._failure
            generation, frame, slot = item
            if (generation == self.session.generation
                    and clock.monotonic() <= self._queued_at+1/self.fps):
                self._last_sent_pts = frame.pts
                self.send_at = slot + 1 / self.fps
                return frame
            self._drop('stale_queued_generation' if generation != self.session.generation else 'queue_replaced')
            self._refresh_slot = slot
            self._refresh.set()
        raise MediaStreamError from self._failure

    def clear_pending(self):
        if self._pending.empty():
            return
        item = self._pending.get_nowait()
        if item is None:
            # A stop sentinel must still wake a receiver during teardown.
            self._pending.put_nowait(None)
        else:
            self._drop('cleared_pending')
            # Replace the discarded slot now. Waiting another frame period
            # loses one transmitted frame on every interrupted queued picture.
            self._refresh_slot = item[2]
            self._refresh.set()
            self._space.set()
        self.session.video_queue_depth = 0

    def _end_queue(self):
        if self._pending.full():
            self._pending.get_nowait()
        self._pending.put_nowait(None)
        self.session.video_queue_depth = 0

    def stop(self):
        super().stop()
        if self._producer is not None and not self._producer.done():
            self._producer.cancel()
        self._end_queue()

    async def aclose(self):
        self.stop()
        if self._producer is not None and self._producer is not asyncio.current_task():
            await asyncio.gather(self._producer, return_exceptions=True)
        if self._reading is not None:
            await asyncio.gather(self._reading, return_exceptions=True)
        self._producer = self._reading = None

    async def _produce(self):
        next_frame = clock.monotonic()
        try:
            while self.readyState == 'live' and not self.session.closed:
                refreshed = self._refresh.is_set()
                if not refreshed:
                    try:
                        await asyncio.wait_for(self._refresh.wait(),
                                               max(0., next_frame - clock.monotonic()))
                        refreshed = True
                    except asyncio.TimeoutError:
                        pass
                self._refresh.clear()
                # The one pending picture owns its presentation slot. Do not
                # consume another slot while the RTP sender is still busy.
                await self._space.wait()
                refreshed = refreshed or self._refresh.is_set()
                self._refresh.clear()
                slot = self._refresh_slot if refreshed and self._refresh_slot is not None else next_frame
                before = clock.monotonic()
                generation = self.session.generation
                self._reading = asyncio.create_task(asyncio.to_thread(self.session.timeline.video_frame))
                image, audio_pts = await asyncio.shield(self._reading)
                self._reading = None
                self.session.render_seconds += clock.monotonic() - before
                if generation != self.session.generation:
                    self._drop('stale_decoded_generation')
                    if refreshed:
                        self._refresh.set()
                    continue
                video_pts = audio_pts * 90000 // 48000
                if video_pts <= self._last_sent_pts:
                    # Catch-up after a decode stall can outrun the audio clock.
                    # A repeated RTP timestamp merges two encoded pictures in
                    # receivers. Wait for real PCM progress instead.
                    self._drop('pcm_retry')
                    if refreshed:
                        self._refresh.set()
                    # A retry is not a new presentation slot. Moving the
                    # deadline here accumulates cadence drift on each retry.
                    await asyncio.sleep(.005)
                    continue
                # A replacement fills a slot already counted before clearing
                # the queue. Even when late, it must not consume the next slot.
                if not refreshed:
                    next_frame = max(next_frame + 1 / self.fps, clock.monotonic() - 1 / self.fps)
                self._refresh_slot = None
                self.session.frames += 1
                frame = av.VideoFrame.from_ndarray(image, format='rgb24')
                frame.pts = video_pts
                frame.time_base = fractions.Fraction(1, 90000)
                if self._pending.full():
                    self._pending.get_nowait()
                    self._drop('queue_replaced')
                self._queued_at = clock.monotonic()
                self._pending.put_nowait((generation, frame, slot))
                self._space.clear()
                self.session.video_queue_depth = self._pending.qsize()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._failure = exc
            self.session.emit('error', text=f'画面播放失败：{exc}')
        finally:
            super().stop()
            self._end_queue()


class PhotoAudioTrack(MediaStreamTrack):
    kind = 'audio'

    def __init__(self, session):
        super().__init__()
        self.session = session
        self._next = None

    async def recv(self):
        if self.readyState != 'live' or self.session.closed:
            raise MediaStreamError
        now = clock.monotonic()
        if self._next is None:
            self._next = now
        await asyncio.sleep(max(0., self._next - now))
        if self.readyState != 'live' or self.session.closed:
            raise MediaStreamError
        pcm, pts = self.session.timeline.read_audio(960)
        frame = av.AudioFrame.from_ndarray((np.clip(pcm, -1, 1) * 32767).astype(np.int16).reshape(1, -1),
                                          format='s16', layout='mono')
        frame.sample_rate = 48000
        frame.pts, frame.time_base = pts, fractions.Fraction(1, 48000)
        self._next = max(self._next + .02, clock.monotonic() - .02)
        return frame


async def negotiate(session, offer, fps=25):
    from backend.transport.video_quality import install_local_video_encoder

    if session.closed or session.peer is not None:
        raise ValueError('会话已关闭或已连接')
    clock.install_rtc_clock()
    peer = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
    session.peer = peer
    try:
        dimension = max(session.manifest['portrait']['width'], session.manifest['portrait']['height'])
        encoder_size = 512 if dimension <= 512 else 768 if dimension <= 768 else 1024
        session.video_bitrate = install_local_video_encoder(encoder_size, session=session)
        session.video_track = PhotoVideoTrack(session, fps)
        session.audio_track = PhotoAudioTrack(session)
        peer.addTrack(session.video_track)
        peer.addTrack(session.audio_track)
        codecs = [codec for codec in RTCRtpSender.getCapabilities('video').codecs
                  if codec.mimeType.lower() in {'video/vp8', 'video/rtx'}]
        for transceiver in peer.getTransceivers():
            if transceiver.kind == 'video':
                transceiver.setCodecPreferences(codecs)

        @peer.on('datachannel')
        def channel_open(channel):
            session.channel = channel
            channel.on('message', session.message)

        @peer.on('track')
        def input_track(track):
            session.input_tracks.add(track)
            if track.kind == 'audio':
                task = asyncio.create_task(consume_microphone(session, track))
                session._transport_tasks.add(task)
                task.add_done_callback(session._transport_tasks.discard)

        @peer.on('connectionstatechange')
        async def state_changed():
            if peer.connectionState in {'failed', 'closed', 'disconnected'} and not session.closed:
                await session.close()

        await peer.setRemoteDescription(RTCSessionDescription(sdp=offer['sdp'], type=offer['type']))
        await peer.setLocalDescription(await peer.createAnswer())
        return {'sdp': peer.localDescription.sdp, 'type': peer.localDescription.type}
    except BaseException:
        await session.close()
        raise


async def consume_microphone(session, track):
    resampler = av.AudioResampler(format='fltp', layout='mono', rate=16000)
    limit = 30 * 16000
    try:
        while not session.closed:
            frame = await track.recv()
            for out in resampler.resample(frame):
                if not session.recording:
                    continue
                remaining = limit - session.recorded_samples
                data = out.to_ndarray().reshape(-1)[:max(0, remaining)].copy()
                if len(data):
                    session.recorded.append(data)
                    session.recorded_samples += len(data)
                if session.recorded_samples >= limit:
                    session.recording = False
                    session.emit('error', text='录音已达到30秒，请松开录音按钮')
    except asyncio.CancelledError:
        raise
    except MediaStreamError:
        session.recording = False
    except Exception as exc:
        session.recording = False
        if not session.closed:
            session.emit('error', text=f'麦克风读取失败：{exc}')
