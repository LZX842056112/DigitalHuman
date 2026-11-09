"""Local CosyVoice 3 cloning with one reference/emotion cache per voice."""
import gc
import hashlib
import io
import sys
import threading
from backend import clock
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
import yaml
from backend.emotion import normalize_emotion,voice_instruction
from backend.gpu import gpu_operation,session_operation,check_stream_headroom
from backend.errors import StreamMemoryUnavailable
from backend.portrait.assets import verified_model
from backend.storage import read_json,sha256
from backend.voice.client import normalize_speech
from backend.stage_timing import StageTiming


def voice_cache_key(reference_digest,emotion,revision,prompt_text='',mode='instruct'):
    parts=(reference_digest,voice_instruction(emotion),revision,mode,prompt_text or '')
    return hashlib.sha256('\0'.join(parts).encode()).hexdigest()


def zero_shot_prompt_text(text):
    """CosyVoice3 wants one shared prefix plus the reference transcript."""
    return 'You are a helpful assistant.<|endofprompt|>'+text.strip()


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

    def warmup(self):
        self._load();self.offload()

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

    def _speaker(self,reference,emotion,cache_folder,prompt_text='',mode='instruct'):
        digest=sha256(reference);identity=(digest,self.revision)
        if getattr(self,'_speaker_identity',None)!=identity:
            self.voice.frontend.spk2info.clear();self._speaker_identity=identity
        key=voice_cache_key(digest,emotion,self.revision,prompt_text,mode)
        cache=Path(cache_folder or reference.parent)/('prompt-'+key+'.safetensors')
        from safetensors.torch import save_file,load_file
        if key not in self.voice.frontend.spk2info:
            if cache.is_file():self.voice.frontend.spk2info[key]=load_file(str(cache))
            else:
                seed=(zero_shot_prompt_text(prompt_text)
                      if mode=='zero_shot' and prompt_text else voice_instruction(emotion))
                self.voice.add_zero_shot_spk(seed,str(reference),key)
                values={name:value.detach().cpu().contiguous().clone()
                        for name,value in self.voice.frontend.spk2info[key].items()}
                cache.parent.mkdir(parents=True,exist_ok=True);save_file(values,str(cache))
        return key

    def _protect_native_jobs(self):
        model=self.voice.model
        if not hasattr(model,'llm_job') or getattr(model,'_digitalhuman_guarded',False):return
        original=model.llm_job
        model._digitalhuman_threads={};model._digitalhuman_errors={};model._digitalhuman_guarded=True
        def guarded(*args,**kwargs):
            identity=kwargs.get('uuid',args[-1] if args else None)
            with model.lock:model._digitalhuman_threads[identity]=threading.current_thread()
            try:original(*args,**kwargs)
            except BaseException as error:
                with model.lock:model._digitalhuman_errors[identity]=error
            finally:
                with model.lock:
                    if identity in model.llm_end_dict:model.llm_end_dict[identity]=True
        model.llm_job=guarded

    def _native_chunks(self,text,emotion,reference,key,*,stream,cancelled=None,mode='instruct'):
        self._protect_native_jobs();model=self.voice.model
        threads=getattr(model,'_digitalhuman_threads',{});errors=getattr(model,'_digitalhuman_errors',{})
        before=set(threads);stopping=False
        llm=getattr(model,'llm',None);original_infer=getattr(llm,'inference',None)
        if original_infer is not None:
            def infer(*args,**kwargs):
                tokens=iter(original_infer(*args,**kwargs))
                try:
                    for token in tokens:
                        if stopping or (cancelled and cancelled()):break
                        yield token
                finally:
                    close=getattr(tokens,'close',None)
                    if close:close()
            llm.inference=infer
        # zero_shot feeds the reference speech tokens to the LLM as well, so the
        # answer inherits the speaker's rate and melody. instruct2 drops them and
        # trades that fidelity for a style instruction such as "sad".
        if mode=='zero_shot':
            native=iter(self.voice.inference_zero_shot(text,'',str(reference),
                zero_shot_spk_id=key,stream=stream))
        else:
            native=iter(self.voice.inference_instruct2(text,voice_instruction(emotion),str(reference),
                zero_shot_spk_id=key,stream=stream))
        def check_error():
            for identity in set(errors)-before:
                if isinstance(errors[identity],torch.cuda.OutOfMemoryError):
                    raise StreamMemoryUnavailable('声音生成时显存不足') from errors[identity]
                raise RuntimeError('声音模型推理失败：'+str(errors[identity])[:400]) from errors[identity]
        try:
            while True:
                try:chunk=next(native)
                except StopIteration:break
                except BaseException:
                    check_error();raise
                check_error();yield chunk
            check_error()
        finally:
            original_error=sys.exc_info()[1];drain_error=None;stopping=True
            try:
                for _ in native:pass
            except BaseException as error:drain_error=error
            finally:
                identities=set(threads)-before
                for identity in identities:
                    if threads[identity] is not threading.current_thread():threads[identity].join()
                with getattr(model,'lock',threading.Lock()):
                    for name in ('tts_speech_token_dict','llm_end_dict','hift_cache_dict','flow_cache_dict'):
                        values=getattr(model,name,{})
                        for identity in identities:values.pop(identity,None)
                    for identity in identities:threads.pop(identity,None);errors.pop(identity,None)
                if original_infer is not None:llm.inference=original_infer
            if drain_error is not None and original_error is None:raise drain_error

    def stream_synthesize(self,text,reference,emotion='neutral',cache_folder=None,cancelled=None,
                          *,session_id,token,prompt_text='',mode='instruct'):
        from backend.stream_audio import SpeechNormalizer
        if not isinstance(text,str) or not text.strip():raise ValueError('回答文字为空')
        emotion=normalize_emotion(emotion);reference=Path(reference)
        timing=StageTiming();started=clock.monotonic();native=None;total=0;raw_peak=0.
        waiting=clock.monotonic()
        with session_operation(self.root,session_id,token,cancelled),torch.inference_mode():
            timing.values['gpu_wait_ms']=(clock.monotonic()-waiting)*1000
            try:
                with timing.measure('load_ms'):self._load()
                if cancelled and cancelled():raise InterruptedError('当前回答已停止')
                check_stream_headroom(self.root,session_id,token)
                with timing.measure('gpu_transfer_ms'):
                    networks=self._networks()
                    for network in networks:network.cuda()
                if networks:torch.cuda.reset_peak_memory_stats()
                with timing.measure('reference_ms'):key=self._speaker(reference,emotion,cache_folder,prompt_text,mode)
                self.voice.model.token_hop_len=25
                normalizer=SpeechNormalizer(self.voice.sample_rate)
                native=self._native_chunks(text,emotion,reference,key,stream=True,cancelled=cancelled,mode=mode)
                stopped=False
                while True:
                    with timing.measure('inference_ms'):
                        try:chunk=next(native)
                        except StopIteration:break
                    stopped=stopped or bool(cancelled and cancelled())
                    if stopped:continue
                    check_stream_headroom(self.root,session_id,token)
                    with timing.measure('normalize_ms'):
                        raw=chunk['tts_speech'].detach().float().cpu().numpy().reshape(-1)
                        if len(raw):raw_peak=max(raw_peak,float(np.max(np.abs(raw))))
                        pcm=normalizer.push(raw)
                    if len(pcm):
                        total+=len(pcm)
                        yield pcm
                if stopped:raise InterruptedError('当前回答已停止')
                if raw_peak<1e-4:raise RuntimeError('声音模型没有生成有效语音')
                tail=np.concatenate([normalizer.finish(),np.zeros(int(self.voice.sample_rate*.24),np.float32)])
                total+=len(tail)
                yield tail
                self.last_stream_report=dict(seconds=total/self.voice.sample_rate,rate=self.voice.sample_rate,
                    generation_seconds=clock.monotonic()-started,timings=timing.values,timing_spans=timing.spans,
                    limited_samples=normalizer.limited_samples,clone_mode=mode,prompt_chars=len(prompt_text),
                    peak_vram_mib=torch.cuda.max_memory_allocated()/1024**2 if networks else 0.)
            except torch.cuda.OutOfMemoryError as error:
                raise StreamMemoryUnavailable('声音生成时显存不足') from error
            finally:
                # Generator.close/HTTP disconnect must not move weights while
                # the native background LLM is still using its CUDA context.
                if native is not None:
                    try:
                        with timing.measure('drain_ms'):native.close()
                    finally:
                        with timing.measure('offload_ms'):self.offload()
                else:
                    with timing.measure('offload_ms'):self.offload()

    def synthesize(self,text,reference,emotion='neutral',cache_folder=None,cancelled=None,
                   prompt_text='',mode='instruct'):
        if not isinstance(text,str) or not text.strip():raise ValueError('回答文字为空')
        emotion=normalize_emotion(emotion);reference=Path(reference);started=clock.monotonic()
        with gpu_operation(self.root,cancelled),torch.inference_mode():
            native=None
            try:
                self._load()
                for network in self._networks():network.cuda()
                torch.cuda.reset_peak_memory_stats()
                key=self._speaker(reference,emotion,cache_folder,prompt_text,mode)
                chunks=[]
                native=self._native_chunks(text,emotion,reference,key,stream=False,cancelled=cancelled,mode=mode)
                for chunk in native:
                    if cancelled and cancelled():raise InterruptedError('当前回答已停止')
                    chunks.append(chunk['tts_speech'].detach().float().cpu().numpy().reshape(-1))
                if not chunks:raise ValueError('声音模型没有生成语音')
                pcm=normalize_speech(np.concatenate(chunks));rate=self.voice.sample_rate
                pcm=np.pad(pcm,(0,int(rate*.24)))
                report={'emotion':emotion,'seconds':len(pcm)/rate,'generation_seconds':clock.monotonic()-started,
                        'clone_mode':mode,'prompt_chars':len(prompt_text),
                        'peak_vram_mib':torch.cuda.max_memory_allocated()/1024**2,'reference_sha256':sha256(reference)}
                return pcm,rate,report
            finally:
                try:
                    if native is not None:native.close()
                finally:self.offload()
