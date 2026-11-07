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


class ChunkVideoReader:
    def __init__(self):
        self.entries=();self.reader=None;self.active=None;self.closed=False

    def append(self,path,start,count):
        if self.closed:raise ValueError('画面流已关闭')
        self.entries=(*self.entries,(Path(path),start,count))

    def at(self,seconds):
        if self.closed or not self.entries:raise ValueError('画面流没有有效分块')
        frame=min(int(seconds*25),self.entries[-1][1]+self.entries[-1][2]-1)
        entry=next((item for item in self.entries if item[1]<=frame<item[1]+item[2]),None)
        if entry is None:raise ValueError('画面流缺少当前帧')
        if self.active!=entry:
            if self.reader:self.reader.close()
            self.reader=VideoReader(entry[0]);self.active=entry
        return self.reader.at((frame-entry[1])/25)

    def close(self):
        if self.closed:return
        self.closed=True
        if self.reader:self.reader.close()
        for path,_,_ in self.entries:path.unlink(missing_ok=True)
        self.entries=()


class StreamingSentence:
    """One bounded sentence whose PCM and sealed video arrive independently."""
    def __init__(self,on_state=None,on_close=None,start_gate=None):
        self.pcm=np.zeros(0,np.float32);self.video=ChunkVideoReader();self.cursor=0
        self.audio_finished=self.video_finished=self.closed=self.started=self.paused=False
        self.final_audio_samples=None;self.created_at=clock.monotonic();self.video_started_at=None
        self.frames=0;self.underruns=0;self.buffered_ms=0.;self.pause_started=None
        self.on_state=on_state;self.on_close=on_close
        self.start_gate=start_gate
        self.chunk_rates=deque(maxlen=3);self.last_chunk_at=clock.monotonic()
        self.audio_rates=deque(maxlen=3);self.last_audio_at=self.last_chunk_at;self.last_audio_seconds=0.

    @property
    def finished(self):return self.audio_finished and self.video_finished

    @property
    def available_samples(self):return min(len(self.pcm),self.frames*1920)

    def append_audio(self,pcm):
        pcm=np.asarray(pcm,np.float32).reshape(-1)
        if self.closed or self.audio_finished or not len(pcm) or not np.isfinite(pcm).all():
            raise ValueError('声音流分块无效或已结束')
        now=clock.monotonic();seconds=len(pcm)/48000
        self.audio_rates.append(seconds/max(now-self.last_audio_at,1e-6))
        self.last_audio_at=now;self.last_audio_seconds=seconds
        self.pcm=np.concatenate([self.pcm,pcm])

    def append_video(self,path,start,count):
        if (self.closed or self.video_finished or type(start) is not int or start!=self.frames
                or type(count) is not int or not 1<=count<=25):
            raise ValueError('画面流分块重复、缺失或乱序')
        now=clock.monotonic()
        self.chunk_rates.append((count/25)/max(now-self.last_chunk_at,1e-6));self.last_chunk_at=now
        self.video.append(path,start,count);self.frames+=count

    def start_video(self):
        if self.video_started_at is not None or self.frames:raise ValueError('画面流重复开始')
        self.video_started_at=clock.monotonic();self.last_chunk_at=self.video_started_at

    def _validate_end(self):
        if self.finished and (not self.final_audio_samples or self.frames!=(self.final_audio_samples+1919)//1920):
            raise ValueError('画面流与声音的末尾不匹配')

    def finish_audio(self):
        if not len(self.pcm):raise ValueError('声音流没有有效采样')
        self.final_audio_samples=len(self.pcm)
        self.audio_finished=True;self._validate_end()

    def finish_video(self):
        self.video_finished=True;self._validate_end()

    def _enough_buffer(self):
        available=(self.available_samples-self.cursor)/48000
        if self.finished:return available>0
        # A native motion window can publish several fast frame blocks, then
        # stop for its next diffusion pass. Retain its amortized cost too.
        began=self.created_at if self.video_started_at is None else self.video_started_at
        average=(self.frames/25)/max(clock.monotonic()-began,1e-6)
        rate=min(min(self.chunk_rates,default=0.),average)
        if self.audio_finished:
            remaining=(len(self.pcm)-self.cursor)/48000
            target=min(remaining,max(1.,remaining*(1-min(rate,1.))+1.))
            return available>=target
        audio_rate=min(*self.audio_rates,
            self.last_audio_seconds/max(clock.monotonic()-self.last_audio_at,1e-6)) if self.audio_rates else 0.
        return len(self.chunk_rates)==3 and min(rate,audio_rate)>=1.1 and available>=2.

    def can_read(self,count):
        if not self.started and self.start_gate is not None and not self.start_gate():return False
        required=min(count,len(self.pcm)-self.cursor) if self.audio_finished else count
        available=self.available_samples-self.cursor
        if (not self.started or self.paused) and self._enough_buffer() and available>=required>0:
            if self.paused:self.buffered_ms+=(clock.monotonic()-self.pause_started)*1000
            first=not self.started;self.started=True;self.paused=False
            if self.on_state:self.on_state('started' if first else 'resumed')
        if self.started and not self.paused and available>=required>0:return True
        if self.started and not self.paused and not self.finished:
            self.paused=True;self.pause_started=clock.monotonic();self.underruns+=1
            if self.on_state:self.on_state('buffering')
        return False

    def close(self):
        if self.closed:return
        if self.paused:
            self.buffered_ms+=(clock.monotonic()-self.pause_started)*1000;self.paused=False
        self.video.close();self.closed=True;self.pcm=np.zeros(0,np.float32)
        if self.on_close:self.on_close()


class PhotoTimeline:
    def __init__(self,idle,*,idle_video=None):
        self.idle=np.asarray(idle,np.uint8);self._queue=deque();self._current=None;self._video=None
        self._cursor=0;self._video_seconds=0.;self._pts=0;self._generation=0;self._lock=threading.RLock()
        self._decode_lock=threading.Lock();self._readers={};self._retired=set()
        self._idle_video=IdleFrames(idle_video) if idle_video is not None else None
        self._idle_started=clock.monotonic()
        self._stream_expected=self._streams_consumed=0;self._speech_started=False
        self.stream_boundary_underruns=0;self._boundary_started=None;self._boundary_ms=0.;self._boundary_callback=None

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
            self._stream_expected=self._streams_consumed=0;self._speech_started=False
            self.stream_boundary_underruns=0;self._boundary_started=None;self._boundary_ms=0.;self._boundary_callback=None
        for clip in close:clip.close()

    def expect_streams(self,generation,count,on_state=None):
        with self._lock:
            if not self.current(generation):return False
            self._stream_expected=count;self._boundary_callback=on_state
            return True

    def _boundary(self,waiting):
        if waiting and self._speech_started and self._stream_expected>self._streams_consumed:
            if self._boundary_started is None:
                self._boundary_started=clock.monotonic();self.stream_boundary_underruns+=1
                if self._boundary_callback:self._boundary_callback('buffering')
        elif self._boundary_started is not None:
            self._boundary_ms+=(clock.monotonic()-self._boundary_started)*1000;self._boundary_started=None
            if self._boundary_callback:self._boundary_callback('resumed')

    @property
    def stream_boundary_buffer_ms(self):
        return self._boundary_ms+(0. if self._boundary_started is None else (clock.monotonic()-self._boundary_started)*1000)

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
                    if not self._queue:
                        self._boundary(True);break
                    self._current=self._queue.popleft();self._cursor=0
                    if not isinstance(self._current,StreamingSentence):self._video=self._current
                clip=self._current
                if isinstance(clip,StreamingSentence):
                    if clip.audio_finished and self._cursor==len(clip.pcm):
                        self._current=None;self._streams_consumed+=1;continue
                    if not clip.can_read(count-written):
                        if not clip.started:self._boundary(True)
                        break
                    self._speech_started=True;self._boundary(False)
                    self._video=clip
                length=min(count-written,len(clip.pcm)-self._cursor)
                output[written:written+length]=clip.pcm[self._cursor:self._cursor+length]
                self._cursor+=length;written+=length;self._video_seconds=self._cursor/48000
                if isinstance(clip,StreamingSentence):clip.cursor=self._cursor
                if self._cursor==len(clip.pcm) and (not isinstance(clip,StreamingSentence) or clip.audio_finished):
                    self._current=None
                    if isinstance(clip,StreamingSentence):self._streams_consumed+=1
            pts=self._pts;self._pts+=count
        for clip in close:clip.close()
        return output,pts

    def update_stream(self,generation,clip,method,*args):
        with self._lock:
            if not self.current(generation) or clip.closed and method!='finish_video':return False
            getattr(clip,method)(*args)
            return True

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
