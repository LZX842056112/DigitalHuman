from types import SimpleNamespace
import threading

import numpy as np
import pytest


def fake_engine(root):
    from backend.voice.engine import VoiceEngine
    from backend.gpu import SessionGpuLease
    import torch
    engine=VoiceEngine(root);events=[]
    model=SimpleNamespace(token_hop_len=100)
    def inference(*args,**kwargs):
        events.append(('start',kwargs,model.token_hop_len))
        try:
            for index in range(3):
                events.append(('native',index))
                yield {'tts_speech':torch.full((1,24000),.05)}
        finally:events.append(('drained',))
    engine.voice=SimpleNamespace(model=model,sample_rate=24000,inference_instruct2=inference)
    engine._load=lambda:None;engine._networks=lambda:[]
    engine._speaker=lambda *args:'speaker'
    engine.offload=lambda:events.append(('offload',))
    owner=SessionGpuLease(root,'session').acquire()
    return engine,events,owner


def test_native_voice_publishes_before_sentence_is_finished_and_resets_hop(tmp_path):
    engine,events,owner=fake_engine(tmp_path)
    try:
        stream=engine.stream_synthesize('你好',tmp_path/'reference.wav',session_id='session',token=owner.token)
        first=next(stream)
        assert len(first)>0 and ('native',2) not in events
        assert events[0][1]['stream'] is True and events[0][2]==25
        pcm=np.concatenate([first,*stream])
        assert len(pcm)==3*24000+int(24000*.24)
        assert np.max(np.abs(pcm))<=.891
        assert events[-2:]==[('drained',),('offload',)]
        assert engine.last_stream_report['seconds']==pytest.approx(3.24)
    finally:owner.release()


def test_closing_voice_stream_drains_native_before_offload(tmp_path):
    engine,events,owner=fake_engine(tmp_path)
    try:
        stream=engine.stream_synthesize('你好',tmp_path/'reference.wav',session_id='session',token=owner.token)
        next(stream);stream.close()
        assert ('native',2) in events
        assert events[-2:]==[('drained',),('offload',)]
    finally:owner.release()


def test_native_llm_failure_signals_end_instead_of_hanging_tts(tmp_path):
    from backend.voice.engine import VoiceEngine
    engine=VoiceEngine(tmp_path)
    def broken(*args):raise RuntimeError('LLM推理失败')
    model=SimpleNamespace(llm_job=broken,lock=threading.Lock(),llm_end_dict={'job':False})
    engine.voice=SimpleNamespace(model=model)
    engine._protect_native_jobs()
    thread=threading.Thread(target=model.llm_job,args=('job',));thread.start();thread.join(1)
    assert not thread.is_alive() and model.llm_end_dict['job'] is True
    assert 'LLM推理失败' in str(model._digitalhuman_errors['job'])


def test_flow_error_waits_for_native_llm_thread_before_clearing_caches(tmp_path):
    from backend.voice.engine import VoiceEngine
    engine=VoiceEngine(tmp_path);entered=threading.Event();release=threading.Event();finished=threading.Event()
    def llm(*args):entered.set();release.wait(3)
    model=SimpleNamespace(llm_job=llm,lock=threading.Lock(),llm_end_dict={},
        hift_cache_dict={},tts_speech_token_dict={})
    def native(*args,**kwargs):
        model.llm_end_dict['job']=False;model.hift_cache_dict['job']={'active':True}
        thread=threading.Thread(target=model.llm_job,args=('job',));thread.start()
        assert entered.wait(1)
        raise RuntimeError('Flow推理失败')
        yield
    engine.voice=SimpleNamespace(model=model,inference_instruct2=native)
    errors=[]
    def consume():
        try:list(engine._native_chunks('你好','neutral',tmp_path/'ref.wav','speaker',stream=True))
        except RuntimeError as error:errors.append(str(error))
        finally:finished.set()
    thread=threading.Thread(target=consume);thread.start()
    try:
        assert entered.wait(1)
        assert not finished.wait(.03) and model.hift_cache_dict
        release.set();thread.join(2)
        assert finished.is_set() and errors==['Flow推理失败']
        assert not model.hift_cache_dict and not model.llm_end_dict
    finally:release.set();thread.join(3)


def test_silent_native_output_is_rejected_before_publication(tmp_path):
    import torch
    engine,events,owner=fake_engine(tmp_path)
    engine.voice.inference_instruct2=lambda *args,**kwargs:iter([{'tts_speech':torch.zeros((1,24000))}])
    try:
        with pytest.raises(RuntimeError,match='语音'):
            list(engine.stream_synthesize('你好',tmp_path/'ref.wav',session_id='session',token=owner.token))
    finally:owner.release()


def test_speaker_memory_cache_keeps_only_current_reference_and_model(tmp_path):
    import torch
    from backend.voice.engine import VoiceEngine
    class Frontend:
        def __init__(self):self.spk2info={};self.calls=0
        def add_zero_shot_spk(self,instruction,reference,key):
            self.calls+=1;self.spk2info[key]={'embedding':torch.ones(2)}
    engine=VoiceEngine(tmp_path);frontend=Frontend()
    engine.voice=SimpleNamespace(frontend=frontend,add_zero_shot_spk=frontend.add_zero_shot_spk);engine.revision='model-one'
    first=tmp_path/'first.wav';second=tmp_path/'second.wav'
    first.write_bytes(b'first reference');second.write_bytes(b'second reference')
    neutral=engine._speaker(first,'neutral',tmp_path)
    happy=engine._speaker(first,'happy',tmp_path)
    assert set(frontend.spk2info)=={neutral,happy}
    other=engine._speaker(second,'neutral',tmp_path)
    assert set(frontend.spk2info)=={other}
    engine.revision='model-two'
    revised=engine._speaker(second,'neutral',tmp_path)
    assert set(frontend.spk2info)=={revised}
    assert len(list(tmp_path.glob('prompt-*.safetensors')))==4
    assert first.read_bytes()==b'first reference'
