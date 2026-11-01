"""Local CosyVoice 3 cloning with one reference/emotion cache per voice."""
import gc
import hashlib
import io
import sys
from backend import clock
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
import yaml
from backend.emotion import normalize_emotion,voice_instruction
from backend.gpu import gpu_operation
from backend.portrait.assets import verified_model
from backend.storage import read_json,sha256
from backend.voice.client import normalize_speech


def voice_cache_key(reference_digest,emotion,revision):
    return hashlib.sha256((reference_digest+'\0'+voice_instruction(emotion)+'\0'+revision).encode()).hexdigest()


def inference_yaml(text):
    document=yaml.compose(text)
    document.value=[(key,value) for key,value in document.value
        if not value.tag.startswith('!name:cosyvoice.dataset.processor.')
        and not key.value.startswith(('data_pipeline','train_conf'))]
    return yaml.serialize(document)


@contextmanager
def local_text_frontend(root):
    from wetext import wetext
    folder=verified_model(root,'text_normalizer')
    original=wetext.snapshot_download
    def local_snapshot(repo_id,*args,**kwargs):
        if repo_id!='pengzhendong/wetext':raise ValueError('未安装的文本处理模型')
        return str(folder)
    wetext.snapshot_download=local_snapshot
    try:yield
    finally:wetext.snapshot_download=original


class VoiceEngine:
    def __init__(self,root):
        self.root=Path(root);self.voice=None

    def _load(self):
        if self.voice is not None:return
        folder=verified_model(self.root,'cosyvoice')
        repo=self.root/'third_party/cosyvoice'
        for path in [repo,repo/'third_party/Matcha-TTS']:
            if str(path) not in sys.path:sys.path.insert(0,str(path))
        from cosyvoice.cli import cosyvoice
        original=cosyvoice.load_hyperpyyaml
        def load_inference(stream,**kwargs):
            return original(io.StringIO(inference_yaml(stream.read())),**kwargs)
        cosyvoice.load_hyperpyyaml=load_inference
        try:
            with local_text_frontend(self.root):
                self.voice=cosyvoice.AutoModel(model_dir=str(folder),load_trt=False,load_vllm=False,fp16=True)
        finally:cosyvoice.load_hyperpyyaml=original
        for network in self._networks():network.eval().requires_grad_(False)
        self.voice.model.llm.half()
        self.revision=read_json(self.root/'configs/photo-models.json')['models']['cosyvoice']['revision']

    def _networks(self):
        if self.voice is None:return []
        model=self.voice.model
        return [model.llm,model.flow,model.hift]

    def offload(self):
        for network in self._networks():network.cpu()
        if self.voice is not None:
            for key,values in self.voice.frontend.spk2info.items():
                self.voice.frontend.spk2info[key]={name:value.cpu() if isinstance(value,torch.Tensor) else value
                                                for name,value in values.items()}
        gc.collect();torch.cuda.empty_cache()

    def synthesize(self,text,reference,emotion='neutral',cache_folder=None,cancelled=None):
        if not isinstance(text,str) or not text.strip():raise ValueError('回答文字为空')
        emotion=normalize_emotion(emotion);reference=Path(reference);started=clock.monotonic()
        with gpu_operation(self.root,cancelled),torch.inference_mode():
            try:
                self._load()
                for network in self._networks():network.cuda()
                torch.cuda.reset_peak_memory_stats()
                key=voice_cache_key(sha256(reference),emotion,self.revision)
                cache=Path(cache_folder or reference.parent)/('prompt-'+key+'.safetensors')
                from safetensors.torch import save_file,load_file
                if key not in self.voice.frontend.spk2info:
                    if cache.is_file():self.voice.frontend.spk2info[key]=load_file(str(cache))
                    else:
                        self.voice.add_zero_shot_spk(voice_instruction(emotion),str(reference),key)
                        values={name:value.detach().cpu().contiguous().clone()
                                for name,value in self.voice.frontend.spk2info[key].items()}
                        cache.parent.mkdir(parents=True,exist_ok=True);save_file(values,str(cache))
                chunks=[]
                for chunk in self.voice.inference_instruct2(text,voice_instruction(emotion),str(reference),
                                                            zero_shot_spk_id=key,stream=False):
                    if cancelled and cancelled():raise InterruptedError('当前回答已停止')
                    chunks.append(chunk['tts_speech'].detach().float().cpu().numpy().reshape(-1))
                if not chunks:raise ValueError('声音模型没有生成语音')
                pcm=normalize_speech(np.concatenate(chunks));rate=self.voice.sample_rate
                pcm=np.pad(pcm,(0,int(rate*.24)))
                report={'emotion':emotion,'seconds':len(pcm)/rate,'generation_seconds':clock.monotonic()-started,
                        'peak_vram_mib':torch.cuda.max_memory_allocated()/1024**2,'reference_sha256':sha256(reference)}
                return pcm,rate,report
            finally:self.offload()
