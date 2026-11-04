"""Photo dialogue with sentence jobs and a PCM-controlled playback timeline."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import re
import stat
import time
from backend import clock
import uuid

import numpy as np

from backend.dialogue.client import Dialogue
from backend.emotion import parse_reply
from backend.media import write_wav
from backend.model_jobs import sentence_paths, session_folder
from backend.photo_profiles import source_path
from backend.portrait.client import PortraitClient
from backend.storage import atomic_json, safe_id
from backend.transport.photo_timeline import PhotoTimeline, SentenceClip
from backend.voice.client import VoiceClient


class PhotoSession:
    def __init__(self, root, manifest, config):
        # The API constructs sessions in a worker thread without an event loop.
        self.root = Path(root).resolve()
        self.id = uuid.uuid4().hex
        self.manifest, self.config = manifest, config
        self.avatar_dir = self.root / 'data/avatars' / safe_id(manifest['id'])
        portrait = manifest['portrait']
        idle = source_path(self.avatar_dir, portrait['folder']) / 'idle.mkv'
        self.timeline = PhotoTimeline(np.zeros((int(portrait['height']), int(portrait['width']), 3), np.uint8),
                                      idle_video=idle)
        self.folder = session_folder(self.root, self.id)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.dialogue = Dialogue(self.root, config)
        self.voice = VoiceClient(config.get('voice', {}).get('base_url', 'http://127.0.0.1:9880'))
        self.portrait = PortraitClient(config.get('portrait', {}).get('base_url', 'http://127.0.0.1:9881'))
        self.started = self.last_active = clock.monotonic()
        self.closed = False
        self.peer = self.channel = self.video_track = self.audio_track = self.task = None
        self.recording = False
        self.recorded, self.history = [], []
        self.recorded_samples = 0
        self.frames = self.video_queue_depth = self.video_frames_dropped = 0
        self.render_seconds = 0.
        self._generation = 0
        self._tasks, self._model_tasks, self._transport_tasks, self.input_tracks = set(), set(), set(), set()
        self._owned_paths = {}
        self._close_lock = None
        self._close_task = None
        self._stats = {'turns': 0, 'sentences': 0}
        self._timing = None
        self._timing_started = None
        self._save_generation()

    @property
    def generation(self):
        return self._generation

    @property
    def metrics(self):
        return {**self._stats, 'frames': self.frames, 'render_seconds': self.render_seconds,
                'clock_source': clock.source_name(),
                'video_queue_depth': self.video_queue_depth, 'video_frames_dropped': self.video_frames_dropped,
                'video_drop_reasons': dict(getattr(self.video_track, 'drop_reasons', {})),
                'recording': self.recording, 'recorded_samples': self.recorded_samples,
                'last_turn_timing': dict(self._timing) if self._timing is not None else None,
                'seconds': clock.monotonic() - self.started}

    def _begin_timing(self, generation, input_kind, started):
        if not self._current(generation):
            return
        self._timing_started = started
        self._timing = dict.fromkeys(('asr_ms', 'llm_ms', 'tts_first_ms',
            'portrait_first_ms', 'first_enqueue_ms', 'total_ms'))
        self._timing.update(generation=generation, input_kind=input_kind, status='running')

    def _phase_timing(self, generation, phase, started):
        if (self._timing is not None and self._timing['generation'] == generation
                and self._timing['status'] == 'running'):
            self._timing[phase] = max(0., (clock.monotonic() - started) * 1000)

    def _finish_timing(self, generation, status):
        if (self._timing is not None and self._timing['generation'] == generation
                and self._timing['status'] == 'running'):
            self._phase_timing(generation, 'total_ms', self._timing_started)
            self._timing['status'] = status

    def _save_generation(self):
        atomic_json(self.folder / 'generation.json', {'generation': self._generation, 'closed': self.closed})

    def emit(self, kind, **payload):
        if self.channel is not None and self.channel.readyState == 'open':
            try:
                self.channel.send(json.dumps({'type': kind, **payload}, ensure_ascii=False))
            except Exception:
                # A channel can close between the ready-state check and send.
                pass

    def _current(self, generation):
        return not self.closed and self.timeline.current(generation)

    def _cleanup(self, generation, *, defer_busy=True):
        complete = True
        for path in self._owned_paths.get(generation, ()):
            # Only exact sentence paths created by this session are owned here.
            if path.parent == self.folder:
                try:
                    if stat.S_ISDIR(path.lstat().st_mode):
                        continue
                    path.unlink(missing_ok=True)
                except FileNotFoundError:
                    # SentenceClip.close already removed normal playback files.
                    continue
                except PermissionError:
                    complete = False
                    # Windows keeps a worker's open WAV locked. The model's
                    # completion callback retries after its reader has closed.
                    if not defer_busy:
                        raise
        # A late model job may recreate these exact paths. Retain ownership
        # while any worker is active; once drained, do not revisit old paths
        # during every future interrupt on the mounted project drive.
        if complete and not self._model_tasks:
            self._owned_paths.pop(generation, None)

    @staticmethod
    def _consume_result(task):
        if not task.cancelled():
            task.exception()

    def _start(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.task = task
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        task.add_done_callback(self._consume_result)
        return task

    def _invalidate(self):
        self._finish_timing(self._generation, 'interrupted')
        self._generation = self.timeline.begin_turn()
        self._save_generation()
        if self.video_track is not None:
            self.video_track.clear_pending()
        current = asyncio.current_task()
        for task in tuple(self._tasks):
            if task is not current and not task.done():
                task.cancel()
        self.task = None
        for generation in tuple(self._owned_paths):
            if generation != self._generation:
                self._cleanup(generation)
        return self._generation

    def interrupt(self):
        self._invalidate()
        self.recording = False
        self.recorded = []
        self.recorded_samples = 0
        self.emit('status', text='已停止')

    def message(self, raw):
        if self.closed:
            return
        self.last_active = clock.monotonic()
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError('对话消息格式无效')
            kind = data.get('type')
            if kind == 'interrupt':
                self.interrupt()
            elif kind == 'chat_text':
                text = str(data.get('text', '')).strip()[:500]
                if not text:
                    raise ValueError('请输入文字')
                self.interrupt()
                self._begin_timing(self._generation, 'text', self.last_active)
                self._start(self.respond(text, self._generation))
            elif kind == 'start_recording':
                self.interrupt()
                self.recording = True
            elif kind == 'stop_recording':
                if not self.recording and not self.recorded:
                    return
                self.recording = False
                audio = np.concatenate(self.recorded) if self.recorded else np.zeros(0, np.float32)
                self.recorded = []
                self.recorded_samples = 0
                self._begin_timing(self._generation, 'audio', self.last_active)
                if len(audio) >= 3200:
                    self._start(self.respond_audio(audio, self._generation))
                else:
                    self._finish_timing(self._generation, 'completed')
                    self.emit('status', text='未检测到语音')
                    self.emit('turn_complete', generation=self._generation, seconds=0.)
            else:
                raise ValueError('不支持的对话消息')
        except (ValueError, TypeError, KeyError) as exc:
            self.emit('error', text=str(exc))

    async def _model(self, coroutine, generation, *, discard=None):
        # Cancel the waiting turn immediately, but keep ownership of late jobs.
        # Workers read generation.json and remove their output if it is stale.
        task = asyncio.create_task(coroutine)
        self._model_tasks.add(task)

        def finished(done):
            self._model_tasks.discard(done)
            self._consume_result(done)
            if not self._current(generation):
                self._cleanup(generation)

        task.add_done_callback(finished)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if discard is not None:
                async def discard_late_result():
                    try:
                        result = await asyncio.shield(task)
                    except (Exception, asyncio.CancelledError):
                        return
                    try:
                        await asyncio.to_thread(discard, result)
                    finally:
                        self._cleanup(generation)
                disposal = asyncio.create_task(discard_late_result(), name='discard late sentence reader')
                self._model_tasks.add(disposal)
                disposal.add_done_callback(self._model_tasks.discard)
                disposal.add_done_callback(self._consume_result)
            raise

    async def respond_audio(self, audio, generation=None):
        if self.closed:
            return
        generation = self._invalidate() if generation is None else generation
        if not self._current(generation):
            return
        if self._timing is None or self._timing['generation'] != generation:
            self._begin_timing(generation, 'audio', clock.monotonic())
        try:
            self.emit('status', text='正在识别语音')
            phase_started = clock.monotonic()
            text = await self._model(asyncio.to_thread(self.dialogue.transcribe, audio), generation)
            self._phase_timing(generation, 'asr_ms', phase_started)
            if not self._current(generation):
                return
            if not text:
                self._finish_timing(generation, 'completed')
                self.emit('status', text='未检测到语音')
                self.emit('turn_complete', generation=generation, seconds=0.)
                return
            await self.respond(text, generation)
        except asyncio.CancelledError:
            self._finish_timing(generation, 'interrupted')
            raise
        except Exception as exc:
            if self._current(generation):
                self._finish_timing(generation, 'failed')
                self._invalidate()
                self.emit('error', text=str(exc))

    async def respond(self, text, generation=None):
        if self.closed:
            return
        generation = self._invalidate() if generation is None else generation
        started = clock.monotonic()
        if self._timing is None or self._timing['generation'] != generation:
            self._begin_timing(generation, 'text', started)
        try:
            if not self._current(generation):
                return
            self.emit('transcript', role='user', text=text)
            self.emit('status', text='正在生成回答')
            phase_started = clock.monotonic()
            reply = parse_reply(await self.dialogue.answer(text, list(self.history)))
            self._phase_timing(generation, 'llm_ms', phase_started)
            if not self._current(generation):
                return
            self.emit('transcript', role='assistant', text=reply.text)
            self.history.extend([{'role': 'user', 'content': text}, {'role': 'assistant', 'content': reply.text}])
            self.history = self.history[-12:]
            sentences = re.findall(r'[^。！？!?\n]+[。！？!?]?', reply.text)
            for index, sentence in enumerate(sentences):
                # Back pressure applies before inference, so there is no third
                # full sentence waiting outside the playback queue.
                while self._current(generation) and self.timeline.queue_depth >= 2:
                    await asyncio.sleep(.02)
                if not self._current(generation):
                    return
                self.emit('status', text='正在生成声音和画面')
                audio, video = sentence_paths(self.root, self.id, generation, index)
                self._owned_paths.setdefault(generation, set()).update((audio, video))
                phase_started = clock.monotonic()
                pcm, rate = await self._model(self.voice.synthesize(sentence.strip(), self.avatar_dir,
                    self.manifest['voice'], emotion=reply.emotion, session_id=self.id, generation=generation), generation)
                if index == 0:
                    self._phase_timing(generation, 'tts_first_ms', phase_started)
                if not self._current(generation):
                    return
                await self._model(asyncio.to_thread(write_wav, audio, pcm, rate), generation)
                if not self._current(generation):
                    return
                phase_started = clock.monotonic()
                await self._model(self.portrait.animate(self.manifest['id'], self.id, generation, index, reply.emotion), generation)
                if index == 0:
                    self._phase_timing(generation, 'portrait_first_ms', phase_started)
                if not self._current(generation):
                    return
                clip = await self._model(asyncio.to_thread(SentenceClip, pcm, rate, video,
                    delete=True, audio=audio), generation, discard=lambda result: result.close())
                if not self._current(generation):
                    clip.close()
                    return
                self.timeline.append(generation, clip)
                if index == 0:
                    self._phase_timing(generation, 'first_enqueue_ms', self._timing_started)
                self._stats['sentences'] += 1
                self.emit('status', text='正在播放')
            while self._current(generation) and self.timeline.busy:
                await asyncio.sleep(.02)
            if self._current(generation):
                self._finish_timing(generation, 'completed')
                self._stats['turns'] += 1
                self._stats['last_turn_seconds'] = clock.monotonic() - started
                self.emit('turn_complete', generation=generation, seconds=self._stats['last_turn_seconds'])
                self.emit('status', text='等待输入')
        except asyncio.CancelledError:
            self._finish_timing(generation, 'interrupted')
            raise
        except Exception as exc:
            if self._current(generation):
                self._finish_timing(generation, 'failed')
                self._invalidate()
                self.emit('error', text=str(exc))
        finally:
            if not self._current(generation):
                self._cleanup(generation)

    async def close(self):
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close(), name='photo session cleanup')
            self._close_task.add_done_callback(self._consume_result)
        # Request cancellation must not cancel worker draining or file cleanup.
        await asyncio.shield(self._close_task)

    async def _close(self):
        if self._close_lock is None:
            self._close_lock = asyncio.Lock()
        async with self._close_lock:
            if self.closed:
                return
            self.closed = True
            self.interrupt()
            for track in tuple(self.input_tracks):
                track.stop()
            for task in tuple(self._transport_tasks):
                task.cancel()
            if self.video_track is not None:
                await self.video_track.aclose()
                self.video_track = None
            if self.audio_track is not None:
                self.audio_track.stop()
                self.audio_track = None
            if self.peer is not None:
                await self.peer.close()
                self.peer = None
            current = asyncio.current_task()
            pending = [task for task in (*self._tasks, *self._transport_tasks) if task is not current]
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            while self._model_tasks:
                await asyncio.gather(*tuple(self._model_tasks), return_exceptions=True)
            self.timeline.close()
            for generation in tuple(self._owned_paths):
                self._cleanup(generation, defer_busy=False)
            self.input_tracks.clear()
            self.channel = None
