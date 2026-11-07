"""Stable prefixes of the pinned HuBERT windows and Ditto overlap fusion."""
import numpy as np


class FeatureStream:
    def __init__(self,infer):
        self.infer=infer;self.pending=np.zeros(2000,np.float32)
        self.samples=0;self.frames=0;self.finished=False

    def _take(self,target=None):
        values=[]
        while len(self.pending)>=6480 and (target is None or self.frames<target):
            features=np.asarray(self.infer(self.pending[:6480]),np.float32)
            if features.shape!=(5,1024) or not np.isfinite(features).all():raise ValueError('流式声音特征无效')
            if target is not None:features=features[:target-self.frames]
            values.append(features);self.frames+=len(features);self.pending=self.pending[3200:]
        return np.concatenate(values) if values else np.zeros((0,1024),np.float32)

    def push(self,pcm):
        if self.finished:raise ValueError('声音特征流已结束')
        pcm=np.asarray(pcm,np.float32).reshape(-1)
        if not np.isfinite(pcm).all():raise ValueError('声音特征输入无效')
        self.samples+=len(pcm);self.pending=np.concatenate([self.pending,pcm])
        return self._take()

    def finish(self):
        if self.finished:return np.zeros((0,1024),np.float32)
        self.finished=True;self.pending=np.concatenate([self.pending,np.zeros(6480,np.float32)])
        return self._take((self.samples+639)//640)


class MotionStream:
    def __init__(self,model,condition):
        self.model=model;self.condition=condition;self.features=None;self.sequence=None
        self.position=0;self.emitted=0;self.finished=False

    def _generate(self,final=False):
        if self.features is None:return []
        while self.position<len(self.features):
            values=self.features[self.position:self.position+self.model.seq_frames]
            if len(values)<self.model.seq_frames:
                if not final:break
                values=np.concatenate([values,np.repeat(values[-1:],self.model.seq_frames-len(values),axis=0)])
            self.sequence=self.model(self.condition(values,self.position)[None],self.sequence)
            self.position+=self.model.valid_clip_len
        if self.sequence is None:return []
        # Future overlap fusion and smoothing can revise the tail. Retain
        # overlap+two smoothing windows, then apply the original final smoother
        # exactly once to each prefix that can no longer be revised.
        end=len(self.features) if final else min(len(self.features),self.sequence.shape[1]-self.model.overlap_v2-6)
        end=max(self.emitted,end)
        if end==self.emitted:return []
        smoothed=self.model._smo(self.sequence[:,:len(self.features)].copy(),0,end)
        result=self.model.cvt_fmt(smoothed[:,self.emitted:end]);self.emitted=end
        return result

    def push(self,features):
        if self.finished:raise ValueError('动作流已结束')
        if not len(features):return []
        self.features=features.copy() if self.features is None else np.concatenate([self.features,features])
        return self._generate()

    def finish(self):
        if self.finished:return []
        self.finished=True
        if self.features is None or not len(self.features):raise ValueError('声音没有产生动作特征')
        return self._generate(final=True)
