from __future__ import annotations

import asyncio
import json
import math
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from backend.avatar.expressions import ExpressionMap
from backend.avatar.a2e_client import request_weights
from backend.dialogue.client import Dialogue
from backend.voice.client import VoiceClient
from backend.transport.timeline import Timeline
from backend.pipeline.media import write_wav


class Session:
    def __init__(self, root, manifest, renderer, config, render_executor=None):
        self.id = uuid.uuid4().hex
        self.root, self.manifest, self.renderer, self.config = root, manifest, renderer, config
        self.avatar_dir = root / "data/avatars" / manifest["id"]
        self.folder = root / "data/sessions" / self.id
        self.folder.mkdir(parents=True, exist_ok=True)
        self.timeline = Timeline()
        self.mapping = ExpressionMap.load(self.avatar_dir / "expression_map.npz")
        self.dialogue = Dialogue(root, config)
        self.voice = VoiceClient(config.get("voice", {}).get("base_url", "http://127.0.0.1:9880"))
        self.action = "idle"
        self.camera = dict(renderer.default_camera)
        self.camera_limits=getattr(renderer,'camera_limits',{'min_distance':.3})
        self.views=getattr(renderer,'views',{'full':dict(renderer.default_camera)})
        self.position = np.zeros(3, np.float32)
        self.started = time.monotonic()
        self.last_active = self.started
        self.recording = False
        self.recorded = []
        self.recorded_samples = 0
        self.channel = None
        self.peer = None
        self.video_track = None
        self.task = None
        self.history = []
        self.closed = False
        self._render_task = None
        self._render_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self.frames = 0
        self.render_seconds = 0.
        self.video_queue_depth=0
        self.video_frames_dropped=0
        self.render_executor=render_executor or ThreadPoolExecutor(1,thread_name_prefix='avatar-gpu')
        if render_executor is None:
            try:
                self.render_executor.submit(renderer.render,'idle',0,dict(self.camera),np.zeros(100),np.zeros(3),self.position.copy()).result()
            except BaseException:
                self.render_executor.shutdown(wait=True,cancel_futures=True)
                self.render_executor=None
                raise

    def emit(self, kind, **payload):
        if self.channel and self.channel.readyState == "open":
            self.channel.send(json.dumps({"type": kind, **payload}, ensure_ascii=False))

    def interrupt(self):
        self.timeline.interrupt()
        if self.task and not self.task.done():
            self.task.cancel()
        self.task = None
        self.emit("status", text="已停止")

    def message(self, raw):
        self.last_active = time.monotonic()
        try:
            data = json.loads(raw)
            kind = data.get("type")
            if kind == "interrupt":
                self.interrupt()
            elif kind == "action":
                if data.get("action") not in {"idle", "walk", "wave"}:
                    raise ValueError("不支持的动作")
                self.action = data["action"]
            elif kind == 'view':
                if data.get('view') not in self.views: raise ValueError('不支持的取景方式')
                self.camera=dict(self.views[data['view']])
            elif kind == "camera":
                minimum=getattr(self,'camera_limits',{'min_distance':.3})['min_distance']
                for key, low, high in (("yaw", -3600, 3600), ("pitch", -75, 75), ("distance", minimum, 8)):
                    value = float(data.get(key, self.camera[key]))
                    if not math.isfinite(value):
                        raise ValueError("相机参数无效")
                    self.camera[key] = float(np.clip(value, low, high))
            elif kind == "position":
                p = np.asarray(data["position"], np.float32)
                if p.shape != (3,) or not np.isfinite(p).all():
                    raise ValueError("位置参数无效")
                self.position = np.clip(p, -2, 2)
            elif kind == "chat_text":
                self.interrupt()
                text = str(data.get("text", "")).strip()[:500]
                if text:
                    self.task = asyncio.create_task(self.respond(text))
            elif kind == "start_recording":
                self.interrupt()
                self.recorded = []
                self.recorded_samples = 0
                self.recording = True
            elif kind == "stop_recording":
                self.recording = False
                audio = np.concatenate(self.recorded) if self.recorded else np.zeros(0, np.float32)
                self.recorded = []
                if len(audio) >= 3200:
                    self.task = asyncio.create_task(self.respond_audio(audio))
        except (ValueError, TypeError, KeyError) as exc:
            self.emit("error", text=str(exc))

    async def respond_audio(self, audio):
        try:
            self.emit("status", text="正在识别语音")
            text = await asyncio.to_thread(self.dialogue.transcribe, audio)
            if not text:
                self.emit("status", text="未检测到语音")
                return
            await self.respond(text)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.emit("error", text=str(exc))

    async def respond(self, text):
        token = self.timeline.begin_turn()
        turn_started=time.monotonic()
        try:
            self.emit("transcript", role="user", text=text)
            self.emit("status", text="正在生成回答")
            reply = await self.dialogue.answer(text, self.history)
            if not self.timeline.current(token):
                return
            self.emit("transcript", role="assistant", text=reply)
            self.history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": reply}])
            self.history = self.history[-12:]
            for index, sentence in enumerate(re.findall(r"[^。！？!?]+[。！？!?]?", reply)):
                self.emit("status", text="正在生成声音和口型")
                pcm, rate = await self.voice.synthesize(sentence, self.avatar_dir, self.manifest["voice"])
                if not self.timeline.current(token):
                    return
                path = self.folder / f"{token}_{index}.wav"
                write_wav(path, pcm, rate)
                weights = await request_weights(path.read_bytes(), timeout=180, headers={"Content-Type": "audio/wav"})
                expressions, jaws = self.mapping.apply(weights)
                while self.timeline.current(token):
                    try:
                        self.timeline.append(token, pcm, rate, expressions, jaws)
                        break
                    except BufferError:
                        await asyncio.sleep(.05)
                self.emit("status", text="正在播放")
            while self.timeline.current(token) and self.timeline.busy:
                await asyncio.sleep(.05)
            if self.timeline.current(token):
                self.emit('turn_complete',generation=token,seconds=time.monotonic()-turn_started)
                self.emit('status',text='等待输入')
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self.timeline.current(token):
                self.timeline.interrupt()
                self.emit("error", text=str(exc))

    async def render_frame(self, expression, jaw):
        async with self._render_lock:
            if self.closed:
                raise asyncio.CancelledError
            self._render_task = asyncio.get_running_loop().run_in_executor(getattr(self,'render_executor',None),
                     self.renderer.render,self.action,time.monotonic()-self.started,dict(self.camera),expression,jaw,self.position.copy())
            try:
                return await asyncio.shield(self._render_task)
            finally:
                if self._render_task.done():
                    self._render_task = None

    async def close(self):
        async with self._close_lock:
            if self.closed and self.renderer is None:
                return
            self.closed = True
            self.interrupt()
            if self.peer:
                await self.peer.close()
            video_track=getattr(self,'video_track',None)
            if video_track is not None:
                await video_track.aclose()
                self.video_track=None
            if self._render_task is not None:
                await asyncio.gather(self._render_task,return_exceptions=True)
                self._render_task = None
            unload=getattr(self.renderer,'close',None)
            executor=getattr(self,'render_executor',None)
            if callable(unload): await asyncio.get_running_loop().run_in_executor(executor,unload)
            unload=None
            if executor is not None:
                executor.shutdown(wait=True,cancel_futures=True)
                self.render_executor=None
            self.renderer = None
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            lease=getattr(self,'gpu_lease',None)
            if lease is not None:
                lease.release()
                self.gpu_lease=None
