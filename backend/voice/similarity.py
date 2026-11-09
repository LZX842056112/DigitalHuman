"""Measure how much a generated clip sounds like the reference speaker.

CosyVoice's own campplus encoder is used, with the same 80-bin Kaldi filterbank
the cloner feeds it, so the number matches what the model actually conditions
on. Two clips of one speaker land around 0.75-0.95; different speakers usually
stay below 0.5.
"""
from pathlib import Path
import numpy as np

_MAX_SECONDS=10


def model_path(root):
    from backend.storage import read_json
    destination=read_json(Path(root)/'configs/photo-models.json')['models']['cosyvoice']['destination']
    return Path(root)/destination/'campplus.onnx'


class SpeakerEmbedder:
    def __init__(self,path):
        import onnxruntime
        self.torch,self.torchaudio=_backend()
        options=onnxruntime.SessionOptions()
        options.graph_optimization_level=onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads=2
        self.session=onnxruntime.InferenceSession(str(path),sess_options=options,
            providers=['CPUExecutionProvider'])
        self.input=self.session.get_inputs()[0].name

    def embedding(self,pcm,rate):
        speech=self.torch.from_numpy(np.asarray(pcm,np.float32).reshape(1,-1))
        if rate!=16000:speech=self.torchaudio.functional.resample(speech,rate,16000)
        if speech.shape[1]<1600:raise ValueError('声音太短，无法比较')
        speech=speech[:,:16000*_MAX_SECONDS]
        feat=self.torchaudio.compliance.kaldi.fbank(speech,num_mel_bins=80,dither=0,
            sample_frequency=16000)
        feat=feat-feat.mean(dim=0,keepdim=True)
        value=self.session.run(None,{self.input:feat.unsqueeze(0).numpy()})[0]
        vector=np.asarray(value,np.float32).reshape(-1)
        return vector/max(float(np.linalg.norm(vector)),1e-8)


def _backend():
    """Import the optional CPU extras once, so a missing one disables scoring."""
    import torch
    import torchaudio
    return torch,torchaudio


def load(root):
    """Return an embedder, or None when the optional CPU extras are absent."""
    try:
        path=model_path(root)
        if not path.is_file():return None
        return SpeakerEmbedder(path)
    except (ImportError,OSError,ValueError,KeyError):
        return None


def cosine(left,right):
    return float(np.dot(left,right))
