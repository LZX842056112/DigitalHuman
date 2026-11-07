"""Continuous resampling and one fixed speech gain across native TTS blocks."""
from fractions import Fraction
import math

import av
import numpy as np
from scipy.ndimage import maximum_filter1d


class StreamingResampler:
    def __init__(self,input_rate,output_rate):
        if input_rate<=0 or output_rate<=0:raise ValueError('采样率无效')
        self.input_rate=int(input_rate);self.output_rate=int(output_rate);self.samples=0
        self.resampler=av.AudioResampler(format='flt',layout='mono',rate=self.output_rate)
        self.finished=False

    def _convert(self,frame):
        frames=self.resampler.resample(frame)
        return np.concatenate([item.to_ndarray().reshape(-1) for item in frames]).astype(np.float32) if frames else np.zeros(0,np.float32)

    def push(self,pcm):
        pcm=np.asarray(pcm,np.float32).reshape(-1)
        if self.finished or not np.isfinite(pcm).all():raise ValueError('重采样声音无效或已结束')
        if not len(pcm):return np.zeros(0,np.float32)
        frame=av.AudioFrame.from_ndarray(pcm[None],format='flt',layout='mono')
        frame.sample_rate=self.input_rate;frame.pts=self.samples;frame.time_base=Fraction(1,self.input_rate)
        self.samples+=len(pcm)
        return self._convert(frame)

    def finish(self):
        if self.finished:return np.zeros(0,np.float32)
        self.finished=True
        return self._convert(None)


class SpeechNormalizer:
    def __init__(self,rate,target_rms=.1,peak_limit=.891):
        if rate<=0:raise ValueError('采样率无效')
        self.rate=rate;self.target_rms=target_rms;self.peak_limit=peak_limit
        self.lookahead=max(2,2*(int(rate*.005)//2))
        self.release=1-math.exp(-1/(rate*.05));self.limiter_gain=1.
        self.pending=np.zeros(0,np.float32);self.gain=None;self.finished=False
        self.limited_samples=0

    def _process(self,final=False):
        if self.gain is None:
            voiced=np.flatnonzero(np.abs(self.pending)>1e-5)
            if not len(voiced):
                if final:
                    result=self.pending;self.pending=np.zeros(0,np.float32)
                    return result
                return np.zeros(0,np.float32)
            start=int(voiced[0]);end=min(len(self.pending),start+self.rate//2)
            if not final and end-start<self.rate//2:return np.zeros(0,np.float32)
            prefix=self.pending[start:end]
            peak=float(np.max(np.abs(prefix)));active=prefix[np.abs(prefix)>peak*.05].astype(np.float64)
            self.gain=min(self.target_rms/float(np.sqrt(np.mean(active**2))),.5/peak)
        count=len(self.pending) if final else max(0,len(self.pending)-self.lookahead)
        if not count:return np.zeros(0,np.float32)
        amplified=self.pending*self.gain
        peaks=maximum_filter1d(np.abs(amplified),size=self.lookahead+1,
            origin=-(self.lookahead//2),mode='constant',cval=0)
        output=np.empty(count,np.float32)
        for index in range(count):
            target=min(1.,self.peak_limit/max(float(peaks[index]),1e-12))
            self.limiter_gain=min(target,self.limiter_gain+(1-self.limiter_gain)*self.release)
            output[index]=amplified[index]*self.limiter_gain
            self.limited_samples+=self.limiter_gain<.999
        self.pending=self.pending[count:].copy()
        return np.clip(output,-self.peak_limit,self.peak_limit)

    def push(self,pcm):
        pcm=np.asarray(pcm,np.float32).reshape(-1)
        if self.finished or not np.isfinite(pcm).all():raise ValueError('声音数据无效或已结束')
        self.pending=np.concatenate([self.pending,pcm])
        return self._process()

    def finish(self):
        if self.finished:return np.zeros(0,np.float32)
        self.finished=True
        return self._process(final=True)
