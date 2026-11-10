"""Photo dialogue with sentence jobs and a PCM-controlled playback timeline."""
from __future__ import annotations

import asyncio
from contextlib import aclosing
import json
from pathlib import Path
import stat
import time
from backend import clock
import uuid

import numpy as np

from backend.dialogue.client import Dialogue
from backend.emotion import normalize_emotion, parse_reply, Reply
from backend.media import write_wav
from backend.model_jobs import sentence_paths, session_folder
from backend.photo_profiles import source_path
from backend.portrait.client import PortraitClient
from backend.storage import atomic_json, safe_id
from backend.style import merge_settings, settings_of
from backend.transport.photo_timeline import PhotoTimeline, SentenceClip, StreamingSentence
from backend.voice.client import VoiceClient


SENTENCE_END = '。！？!?…'
SENTENCE_TAIL = '”’）)」』】"\''


def split_first_clause(sentences, limit):
    """Keep the opening segment short so playback can start sooner.

    Only the first sentence is cut, and only at an existing punctuation mark,
    so the spoken text keeps its original words and order.
    """
    if not sentences or limit is None or limit <= 0:
        return sentences
    first = sentences[0]
    if len(first) <= limit:
        return sentences
    for index, char in enumerate(first[:limit]):
        if char in '，,、；;：:':
            head, tail = first[:index + 1], first[index + 1:]
            if len(head.strip('，,、；;：: ')) >= 4 and len(tail.strip()) >= 2:
                return [head, tail, *sentences[1:]]
            break
    return sentences


def split_sentences(text, *, final=True):
    """Cut a reply at sentence punctuation without dropping any character.

    With ``final`` false the unfinished tail is returned instead of being
    spoken, so a streamed reply can be cut while the model is still writing.
    """
    sentences, start, index, consumed = [], 0, 0, 0
    length = len(text)
    while index < length:
        char = text[index]
        if char in SENTENCE_END:
            end = index + 1
            while end < length and text[end] in SENTENCE_END + SENTENCE_TAIL:
                end += 1
            if not final and end >= length:
                # More punctuation may still arrive; keep the tail for the next chunk.
                break
            segment = text[start:end].strip()
            if segment.strip(SENTENCE_END + SENTENCE_TAIL):
                sentences.append(segment)
            start = index = end
            consumed = end
        elif char == '\n':
            segment = text[start:index].strip()
            if segment.strip(SENTENCE_END + SENTENCE_TAIL):
                sentences.append(segment)
            index += 1
            start = consumed = index
        else:
            index += 1
    if final:
        segment = text[start:].strip()
        if segment:
            sentences.append(segment)
        consumed = length
    return sentences, text[consumed:]


class SentenceSplitter:
    """Publish each sentence as soon as its closing punctuation arrives."""

    def __init__(self, limit=0):
        self.limit = limit if isinstance(limit, int) and limit > 0 else 0
        self.pending = ''
        self.opened = False

    def feed(self, delta):
        self.pending += delta
        sentences, self.pending = split_sentences(self.pending, final=False)
        if self.opened:
            return sentences
        head, tail = self._clause(self.pending)
        if head is not None:
            self.opened = True
            self.pending = tail
            return [head, *sentences]
        if sentences:
            self.opened = True
            return split_first_clause(sentences, self.limit)
        return sentences

    def flush(self):
        sentences, self.pending = split_sentences(self.pending, final=True)
        if sentences and not self.opened:
            self.opened = True
            sentences = split_first_clause(sentences, self.limit)
        return sentences

    def _clause(self, text):
        """Cut the opening clause at a comma before its sentence is finished."""
        if self.limit <= 0:
            return None, text
        for index, char in enumerate(text[:self.limit]):
            if char in '，,、；;：:':
                head, tail = text[:index + 1], text[index + 1:]
                if len(head.strip('，,、；;：: ')) < 4 or len(tail.strip()) < 2:
                    return None, text
                return head, tail
        return None, text


class ReplyFeed:
    """Sentences and emotion of a reply that is still being written."""

    def __init__(self, limit=0):
        self.items = []
        self.emotion = None
        self.reply = None
        self.error = None
        self.splitter = SentenceSplitter(limit)
        self.emotion_ready = asyncio.Event()
        self.changed = asyncio.Event()
        self.done = asyncio.Event()

    def add(self, sentences):
        if not sentences:
            return
        self.items.extend(sentences)
        self.changed.set()

    async def get(self, index):
        while len(self.items) <= index:
            if self.error is not None:
                raise self.error
            if self.done.is_set():
                return None
            self.changed.clear()
            if len(self.items) > index:
                break
            await self.changed.wait()
        return self.items[index]

    def finish(self, error=None, reply=None):
        if reply is not None:
            self.reply = reply
            if self.emotion is None:
                self.emotion = reply.emotion
        self.error = error
        self.done.set()
        self.emotion_ready.set()
        self.changed.set()


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
        self._stream_lease=None;self._delivery_mode='buffered';self._fallback_reason=None
        self._turn_streams=[];self._stream_clips={};self._stream_active_folders=set()
        self._stream_cleanup_errors=[]
        self._stream_events=[]
        self._memory_waiter=None;self._memory_failures={}
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
                'delivery_mode':self._delivery_mode,'stream_fallback_reason':self._fallback_reason,
                'stream_underruns':sum(clip.underruns for clip in self._turn_streams)+self.timeline.stream_boundary_underruns,
                'stream_buffer_ms':sum(clip.buffered_ms for clip in self._turn_streams)+self.timeline.stream_boundary_buffer_ms,
                'paired_buffer_seconds':sum(max(0,clip.available_samples-clip.cursor)/48000
                    for clip in self._turn_streams if not clip.closed),
                'stream_cleanup_errors':list(self._stream_cleanup_errors),
                'stream_buffer_events':list(self._stream_events),
                'seconds': clock.monotonic() - self.started}

    def _begin_timing(self, generation, input_kind, started):
        if not self._current(generation):
            return
        self._timing_started = started
        self._stream_events=[]
        self._timing = dict.fromkeys(('asr_ms', 'llm_ms', 'tts_first_ms',
            'portrait_first_ms', 'first_enqueue_ms', 'total_ms', 'tts_first_chunk_ms',
            'portrait_first_chunk_ms','first_pair_ready_ms','playback_start_ms',
            'llm_first_ms','llm_first_sentence_ms'))
        self._timing.update(generation=generation, input_kind=input_kind, status='running')

    def _phase_timing(self, generation, phase, started):
        if (self._timing is not None and self._timing['generation'] == generation
                and self._timing['status'] == 'running'):
            self._timing[phase] = max(0., (clock.monotonic() - started) * 1000)

    def _first_timing(self, generation, phase, started):
        if (self._timing is not None and self._timing['generation'] == generation
                and self._timing['status'] == 'running' and self._timing.get(phase) is None):
            self._phase_timing(generation, phase, started)

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
        self._cleanup_streams()

    def _cleanup_streams(self,strict=False):
        for folder,clip in list(self._stream_clips.items()):
            if folder in self._stream_active_folders:continue
            protected={entry[0] for entry in clip.video.entries} if not clip.closed else set()
            try:
                if folder.resolve()!=folder:raise ValueError('流式清理目录包含目录链接')
                if folder.exists():
                    for path in folder.iterdir():
                        if path not in protected and not path.is_dir():path.unlink(missing_ok=True)
                    if not any(folder.iterdir()):folder.rmdir()
                if not folder.exists():self._stream_clips.pop(folder,None)
            except (OSError,ValueError) as error:
                if strict:raise
                self._stream_cleanup_errors=(self._stream_cleanup_errors+[str(error)])[-8:]

    async def _ensure_stream_lease(self,generation):
        if self._memory_waiter is not None:await self._memory_waiter.wait()
        if self._stream_lease is not None:return
        from backend.gpu import SessionGpuLease,stream_admission
        concurrent,reason=await asyncio.to_thread(stream_admission,self.root)
        def acquire():return SessionGpuLease(self.root,self.id,concurrent=concurrent).acquire()
        self._stream_lease=await self._model(asyncio.to_thread(acquire),generation,discard=lambda lease:lease.release())
        self._delivery_mode='streaming_parallel' if concurrent else 'streaming_serial'
        self._fallback_reason=reason

    def _first_stream_timing(self,generation,key,started):
        if self._timing is not None and self._timing['generation']==generation and self._timing.get(key) is None:
            self._phase_timing(generation,key,started)

    def _stream_state(self,generation,kind,index=None,clip=None):
        if not self._current(generation):return
        event=dict(generation=generation,kind=kind,sentence_index=index,
            at_ms=(clock.monotonic()-self._timing_started)*1000)
        if clip is not None:
            event.update(cursor=clip.cursor,audio_samples=len(clip.pcm),paired_samples=clip.available_samples,
                frames=clip.frames,audio_finished=clip.audio_finished,video_finished=clip.video_finished)
        self._stream_events=(self._stream_events+[event])[-64:]

    async def _pump_reply(self,feed,text,generation,started):
        """Publish sentences while the dialogue model is still writing."""
        def record(reply):
            if not self._current(generation):return
            self.history.extend([{'role':'user','content':text},
                                 {'role':'assistant','content':reply.text}])
            self.history=self.history[-12:]
            self.emit('transcript',role='assistant',text=reply.text)

        def publish():
            if not feed.items:return
            self._first_timing(generation,'llm_first_sentence_ms',started)
            if self._current(generation):
                self.emit('reply_partial',text=''.join(feed.items))

        try:
            tone=settings_of(self.manifest)['tone']
            stream=getattr(self.dialogue,'stream',None)
            if stream is None or not self.config.get('runtime',{}).get('dialogue_streaming',True):
                reply=parse_reply(await self.dialogue.answer(text,list(self.history),tone))
                feed.add(feed.splitter.feed(reply.text));feed.add(feed.splitter.flush())
                self._phase_timing(generation,'llm_ms',started)
                publish();record(reply);feed.finish(reply=reply)
                return
            async with aclosing(stream(text,list(self.history),tone)) as events:
                async for event in events:
                    kind=event.get('type')
                    if kind=='emotion':
                        feed.emotion=normalize_emotion(event.get('emotion'))
                        feed.emotion_ready.set()
                    elif kind=='text':
                        delta=event.get('delta') or ''
                        if not delta:continue
                        self._first_timing(generation,'llm_first_ms',started)
                        before=len(feed.items)
                        feed.add(feed.splitter.feed(delta))
                        if len(feed.items)>before:publish()
                    elif kind=='done':
                        reply=event.get('reply')
                        if not isinstance(reply,Reply):reply=parse_reply(reply)
                        feed.add(feed.splitter.flush())
                        self._phase_timing(generation,'llm_ms',started)
                        publish();record(reply);feed.finish(reply=reply)
                        return
            raise RuntimeError('对话流在完成前断开')
        except asyncio.CancelledError:
            feed.finish(error=RuntimeError('回答已停止'))
            raise
        except Exception as error:
            feed.finish(error=error)
            raise
        finally:
            feed.emotion_ready.set()

    def _spawn(self,coroutine):
        task=asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        task.add_done_callback(self._consume_result)
        return task

    async def _stream_sentence(self,sentence,index,generation,feed,voice_done):
        from backend.stream_jobs import stream_folder
        from backend.stream_audio import StreamingResampler
        folder=await asyncio.to_thread(stream_folder,self.root,self.id,generation,index)
        def state(kind):
            if not self._current(generation):return
            self._stream_state(generation,kind,index,clip)
            if kind=='started' and index==0:self._first_stream_timing(generation,'playback_start_ms',self._timing_started)
            self.emit('status',text='正在缓冲声音和画面' if kind=='buffering' else '正在播放')
        def start_gate():
            if index:return True
            if feed.done.is_set():
                return len(feed.items)==1 or (len(self._turn_streams)>1 and self._turn_streams[1]._enough_buffer())
            return len(self._turn_streams)>1 and self._turn_streams[1]._enough_buffer()
        clip=StreamingSentence(state,self._cleanup_streams,start_gate)
        self._stream_clips[folder]=clip;self._stream_active_folders.add(folder);self._turn_streams.append(clip)
        audio_ready=asyncio.Event();tasks=[]
        if not self.timeline.append(generation,clip):
            voice_done.set();self._stream_active_folders.discard(folder);self._cleanup_streams();return
        await feed.emotion_ready.wait()
        emotion=feed.emotion or 'neutral'
        def paired():
            if index==0 and clip.available_samples:
                self._first_stream_timing(generation,'first_pair_ready_ms',self._timing_started)
                self._first_stream_timing(generation,'first_enqueue_ms',self._timing_started)
        async def voice():
            started=clock.monotonic();resampler=None
            try:
                async with aclosing(self.voice.stream(sentence.strip(),self.avatar_dir,self.manifest['voice'],
                    root=self.root,session_id=self.id,generation=generation,index=index,
                    token=self._stream_lease.token,emotion=emotion)) as events:
                    async for event in events:
                        if event['type']=='start':resampler=StreamingResampler(event['rate'],48000)
                        elif event['type']=='chunk':
                            pcm=await asyncio.to_thread(resampler.push,event['pcm'])
                            if len(pcm):self.timeline.update_stream(generation,clip,'append_audio',pcm)
                            if index==0:self._first_stream_timing(generation,'tts_first_chunk_ms',started)
                            paired();audio_ready.set()
                        elif event['type']=='complete':
                            tail=await asyncio.to_thread(resampler.finish)
                            if len(tail):self.timeline.update_stream(generation,clip,'append_audio',tail)
                            self.timeline.update_stream(generation,clip,'finish_audio')
                            if index==0:
                                self._phase_timing(generation,'tts_first_ms',started)
                                if self._current(generation):self._timing['voice_worker']=event.get('report',{})
            finally:voice_done.set();audio_ready.set()
        async def portrait():
            await audio_ready.wait()
            if voice_task.done() and not voice_task.cancelled() and voice_task.exception():raise voice_task.exception()
            started=clock.monotonic()
            async with aclosing(self.portrait.stream(self.manifest['id'],self.id,generation,index,emotion,
                expression=settings_of(self.manifest)['expression'],root=self.root,
                token=self._stream_lease.token)) as events:
                async for event in events:
                    if event['type']=='start':self.timeline.update_stream(generation,clip,'start_video')
                    elif event['type']=='chunk':
                        self.timeline.update_stream(generation,clip,'append_video',event['video'],event['frame_start'],event['frames'])
                        if index==0:self._first_stream_timing(generation,'portrait_first_chunk_ms',started)
                        paired()
                    elif event['type']=='complete':
                        if self._current(generation) and event.get('report',{}).get('frames')!=clip.frames:
                            raise ValueError('画面完成事件的总帧数不匹配')
                        self.timeline.update_stream(generation,clip,'finish_video')
                        if index==0:
                            self._phase_timing(generation,'portrait_first_ms',started)
                            if self._current(generation):self._timing['portrait_worker']=event.get('report',{})
        voice_task=asyncio.create_task(voice());tasks.append(voice_task)
        tasks.append(asyncio.create_task(portrait()))
        try:
            await asyncio.gather(*tasks)
            if self._current(generation):self._stats['sentences']+=1
        except Exception as error:
            if self._current(generation):
                from backend.errors import StreamMemoryUnavailable
                retry=isinstance(error,StreamMemoryUnavailable) and not any(item.started for item in self._turn_streams)
                if not retry:self._finish_timing(generation,'failed')
                self._generation=self.timeline.begin_turn()
                self._save_generation()
                if self.video_track is not None:self.video_track.clear_pending()
                if isinstance(error,StreamMemoryUnavailable):
                    self._memory_waiter=asyncio.Event()
                    self._memory_failures[generation]=dict(retry=retry,generation=self._generation,
                        waiter=self._memory_waiter,error=str(error))
                    if retry:self._timing['generation']=self._generation
                if not retry:self.emit('error',text=str(error))
            raise
        finally:
            # Do not cancel an HTTP pump and then release session GPU ownership
            # while its native producer still drains. Both workers acknowledge
            # completion/error only after their generator cleanup.
            await asyncio.gather(*tasks,return_exceptions=True)
            self._stream_active_folders.discard(folder);self._cleanup_streams()

    async def _stream_reply(self,feed,generation):
        jobs=[];last_voice=None;index=0
        def state(kind):
            if self._current(generation):
                self._stream_state(generation,kind)
                self.emit('status',text='正在缓冲声音和画面' if kind=='buffering' else '正在播放')
        self.timeline.expect_streams(generation,1,state)
        try:
            while self._current(generation):
                sentence=await feed.get(index)
                if sentence is None:break
                while self._current(generation):
                    for job in jobs:
                        if job.done() and not job.cancelled() and job.exception():raise job.exception()
                    active=sum(not job.done() for job in jobs)
                    if ((last_voice is None or last_voice.is_set()) and active<2
                            and self.timeline.queue_depth<2):break
                    await asyncio.sleep(.02)
                if not self._current(generation):break
                self.timeline.expect_streams(generation,index+1,state)
                last_voice=asyncio.Event()
                jobs.append(asyncio.create_task(self._stream_sentence(sentence,index,generation,feed,last_voice)))
                index+=1
            if jobs:await asyncio.gather(*jobs)
        finally:
            await asyncio.gather(*jobs,return_exceptions=True)

    async def _stream_with_fallback(self,feed,generation):
        for attempt in range(2):
            try:
                await self._stream_reply(feed,generation)
                return generation
            except Exception:
                failure=self._memory_failures.pop(generation,None)
                if failure is None:raise
                try:
                    await asyncio.to_thread(self._stream_lease.set_concurrent,False)
                    self._delivery_mode='streaming_serial';self._fallback_reason=failure['error']
                finally:failure['waiter'].set()
                if not failure['retry'] or not self._current(failure['generation']):raise
                generation=failure['generation']
                if attempt:
                    self._finish_timing(generation,'failed');self.emit('error',text=failure['error']);raise
                self._turn_streams=[]
                for key in ('tts_first_ms','portrait_first_ms','tts_first_chunk_ms','portrait_first_chunk_ms',
                            'first_enqueue_ms','first_pair_ready_ms','playback_start_ms'):
                    self._timing[key]=None
                self._timing['stream_retries']=1
                self.emit('status',text='显存余量不足，正在使用串行方式重新准备声音和画面')

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
        self._cleanup_streams()
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
            elif kind == 'settings':
                update = {key: data[key] for key in ('tone', 'expression') if key in data}
                settings = merge_settings(self.manifest, update)
                self.manifest['settings'] = settings
                atomic_json(self.avatar_dir / 'avatar.json', self.manifest)
                self.emit('status', text='语气与表情已更新')
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
            # The reply is spoken sentence by sentence while the model writes it.
            feed = ReplyFeed(self.config.get('runtime', {}).get('first_clause_chars', 14))
            pump = self._spawn(self._pump_reply(feed, text, generation, phase_started))
            streaming=bool(self.config.get('runtime',{}).get('photo_streaming',False))
            self._turn_streams=[]
            try:
                if streaming:
                    await self._ensure_stream_lease(generation)
                    generation=await self._model(self._stream_with_fallback(feed,generation),generation)
                else:
                    index=0
                    while True:
                        sentence=await feed.get(index)
                        if sentence is None:
                            break
                        # Back pressure applies before inference, so there is no third
                        # full sentence waiting outside the playback queue.
                        while self._current(generation) and self.timeline.queue_depth >= 2:
                            await asyncio.sleep(.02)
                        if not self._current(generation):
                            return
                        await feed.emotion_ready.wait()
                        emotion = feed.emotion or 'neutral'
                        self.emit('status', text='正在生成声音和画面')
                        audio, video = sentence_paths(self.root, self.id, generation, index)
                        self._owned_paths.setdefault(generation, set()).update((audio, video))
                        phase_started = clock.monotonic()
                        pcm, rate = await self._model(self.voice.synthesize(sentence.strip(), self.avatar_dir,
                            self.manifest['voice'], emotion=emotion, session_id=self.id, generation=generation), generation)
                        if index == 0:
                            self._phase_timing(generation, 'tts_first_ms', phase_started)
                        if not self._current(generation):
                            return
                        await self._model(asyncio.to_thread(write_wav, audio, pcm, rate), generation)
                        if not self._current(generation):
                            return
                        phase_started = clock.monotonic()
                        await self._model(self.portrait.animate(self.manifest['id'], self.id, generation, index, emotion,
                            expression=settings_of(self.manifest)['expression']), generation)
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
                        index += 1
                await pump
            finally:
                if not pump.done():
                    pump.cancel()
                await asyncio.gather(pump, return_exceptions=True)
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
            errors=[]
            try:self.interrupt()
            except Exception as error:errors.append(error)
            if self._stream_lease is not None:
                try:self._stream_lease.revoke()
                except Exception as error:errors.append(error)
            for task in tuple(self._tasks):task.cancel()
            for track in tuple(self.input_tracks):
                try:track.stop()
                except Exception as error:errors.append(error)
            for task in tuple(self._transport_tasks):
                task.cancel()
            if self.video_track is not None:
                try:await self.video_track.aclose()
                except Exception as error:errors.append(error)
                self.video_track = None
            if self.audio_track is not None:
                try:self.audio_track.stop()
                except Exception as error:errors.append(error)
                self.audio_track = None
            if self.peer is not None:
                try:await self.peer.close()
                except Exception as error:errors.append(error)
                self.peer = None
            current = asyncio.current_task()
            pending = [task for task in (*self._tasks, *self._transport_tasks) if task is not current]
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            while self._model_tasks:
                await asyncio.gather(*tuple(self._model_tasks), return_exceptions=True)
            try:
                self.timeline.close()
                for generation in tuple(self._owned_paths):
                    self._cleanup(generation, defer_busy=False)
                self._cleanup_streams(strict=True)
            except Exception as error:errors.append(error)
            finally:
                if self._stream_lease is not None:
                    try:self._stream_lease.release()
                    except Exception as error:errors.append(error)
                    self._stream_lease=None
            self.input_tracks.clear()
            self.channel = None
            if errors:raise errors[0]
