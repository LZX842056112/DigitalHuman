"""Bounded sentence playback; the consumed PCM position selects each picture."""
from collections import deque
from math import gcd
from pathlib import Path
import threading
import time
from backend import clock
import av
import numpy as np
from scipy.signal import resample_poly


class VideoReader:
    def __init__(self,path):
        self.path=Path(path);self.container=av.open(str(path));self.stream=self.container.streams.video[0]
        self.fps=float(self.stream.average_rate or 25)
        self.duration=float(self.container.duration or 0)/av.time_base
        self.frames=self.container.decode(video=0);self.index=-1;self.last=None;self.closed=False
        self.at(0)

    def at(self,seconds):
        if self.closed:raise RuntimeError('画面已经关闭')
        index=max(0,int(seconds*self.fps))
        if index<self.index:
            self.container.seek(0);self.frames=self.container.decode(video=0);self.index=-1
        while self.index<index:
            try:frame=next(self.frames)
            except StopIteration:break
            self.last=frame.to_ndarray(format='rgb24');self.index+=1
        if self.last is None:raise ValueError('生成画面没有有效视频帧')
        return self.last

    def close(self):
        if not self.closed:self.container.close();self.closed=True;self.last=None


class IdleFrames:
    """The generated four-second loop has a fixed, bounded decode budget."""
    def __init__(self,path):
        frames=[];size=0
        container=av.open(str(path))
        try:
            self.fps=float(container.streams.video[0].average_rate or 25)
            self.duration=float(container.duration or 0)/av.time_base
            for frame in container.decode(video=0):
                if len(frames)>=100 or frame.width>1024 or frame.height>1024:
                    raise ValueError('待机视频最多100帧，每帧最大1024×1024')
                image=frame.to_ndarray(format='rgb24');size+=image.nbytes
                if size>100*1024*1024*3:
                    raise ValueError('待机视频100帧缓存超过300MiB')
                frames.append(image)
        finally:container.close()
        if not frames:raise ValueError('待机视频没有有效画面')
        self.frames=tuple(frames)

    def at(self,seconds):
        return self.frames[min(len(self.frames)-1,max(0,int(seconds*self.fps)))]

    def close(self):
        self.frames=()


class SentenceClip:
    def __init__(self,pcm,rate,video,*,delete=False,audio=None):
        pcm=np.asarray(pcm,np.float32).reshape(-1)
        if not len(pcm) or not np.isfinite(pcm).all():raise ValueError('待播声音无效')
        if rate!=48000:
            divisor=gcd(int(rate),48000)
            pcm=resample_poly(pcm,48000//divisor,int(rate)//divisor).astype(np.float32)
        self.pcm=pcm;self.video=VideoReader(video);self.delete=delete;self.audio=Path(audio) if audio else None
        self.closed=False
        if self.video.duration and self.video.duration+0.15<len(pcm)/48000:
            self.close();raise ValueError('说话画面比声音短，不能同步播放')

    def close(self):
        if self.closed:return
        self.video.close();self.pcm=np.zeros(0,np.float32);self.closed=True
        if self.delete:
            self.video.path.unlink(missing_ok=True)
            if self.audio:self.audio.unlink(missing_ok=True)


class PhotoTimeline:
    def __init__(self,idle,*,idle_video=None):
        self.idle=np.asarray(idle,np.uint8);self._queue=deque();self._current=None;self._video=None
        self._cursor=0;self._video_seconds=0.;self._pts=0;self._generation=0;self._lock=threading.RLock()
        self._decode_lock=threading.Lock();self._readers={};self._retired=set()
        self._idle_video=IdleFrames(idle_video) if idle_video is not None else None
        self._idle_started=clock.monotonic()

    def begin_turn(self):
        self.interrupt();return self._generation

    def current(self,generation):
        with self._lock:return self._generation==generation

    def _retire(self,clips):
        close=[]
        for clip in set(clips)-{None}:
            if self._readers.get(clip,0):self._retired.add(clip)
            else:close.append(clip)
        return close

    def interrupt(self):
        with self._lock:
            close=self._retire([*self._queue,self._current,self._video])
            self._generation+=1;self._queue.clear();self._current=self._video=None;self._cursor=0
        for clip in close:clip.close()

    def append(self,generation,clip):
        with self._lock:
            if self.current(generation):
                if len(self._queue)>=2:raise BufferError('待播队列已满')
                self._queue.append(clip);return True
        clip.close();return False

    def read_audio(self,count):
        close=[]
        with self._lock:
            output=np.zeros(count,np.float32);written=0
            while written<count:
                if self._current is None:
                    if self._video is not None:
                        close.extend(self._retire([self._video]));self._video=None
                    if not self._queue:break
                    self._current=self._queue.popleft();self._video=self._current;self._cursor=0
                clip=self._current
                length=min(count-written,len(clip.pcm)-self._cursor)
                output[written:written+length]=clip.pcm[self._cursor:self._cursor+length]
                self._cursor+=length;written+=length;self._video_seconds=self._cursor/48000
                if self._cursor==len(clip.pcm):self._current=None
            pts=self._pts;self._pts+=count
        for clip in close:clip.close()
        return output,pts

    def image(self):
        return self.video_frame()[0]

    def video_frame(self):
        with self._lock:
            clip=self._video;seconds=self._video_seconds;pts=self._pts
            if clip is None:
                if self._idle_video is not None:
                    image=self._idle_video.at((clock.monotonic()-self._idle_started)%(self._idle_video.duration or 4.))
                    return image,pts
                return self.idle,pts
            # Keep the reader alive while decoding outside the PCM clock lock.
            self._readers[clip]=self._readers.get(clip,0)+1
        try:
            with self._decode_lock:return clip.video.at(seconds),pts
        finally:
            close=False
            with self._lock:
                self._readers[clip]-=1
                if not self._readers[clip]:
                    del self._readers[clip]
                    if clip in self._retired:self._retired.remove(clip);close=True
            if close:clip.close()

    @property
    def busy(self):
        with self._lock:return self._current is not None or bool(self._queue)

    @property
    def queue_depth(self):
        with self._lock:return len(self._queue)

    def close(self):
        self.interrupt()
        if self._idle_video is not None:self._idle_video.close();self._idle_video=None
