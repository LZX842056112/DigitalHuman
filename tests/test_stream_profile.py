import pytest


def test_memory_profile_requires_every_sample_and_both_workers(monkeypatch,tmp_path):
    from scripts.profile_streaming import admission_record
    monkeypatch.setattr('scripts.profile_streaming.stream_code_key',lambda root:'source-key')
    case={'delivery_mode':'streaming_parallel','last_turn_timing':{'status':'completed',
        'voice_worker':{'peak_vram_mib':2500,'timing_spans':{'inference_ms':[1.,4.]}},
        'portrait_worker':{'peak_vram_mib':1500,'timing_spans':{'motion_ms':[3.,6.]}}}}
    samples=[{'used_mib':1000,'free_mib':7192},{'used_mib':5000,'free_mib':3192}]
    result=admission_record(tmp_path,'GPU-one',1000,samples,[case])
    assert result['status']=='passed' and result['peak_delta_mib']>=4500
    assert result['code_key']=='source-key'
    with pytest.raises(RuntimeError):admission_record(tmp_path,'GPU-one',1000,[],[case])
    with pytest.raises(RuntimeError):admission_record(tmp_path,'GPU-one',1000,samples+[{'used_mib':7500,'free_mib':692}],[case])
    with pytest.raises(RuntimeError):admission_record(tmp_path,'GPU-one',1000,samples,[{**case,'delivery_mode':'streaming_serial'}])
    with pytest.raises(RuntimeError):admission_record(tmp_path,'GPU-one',1000,samples,[{'delivery_mode':'streaming_parallel'}])
    import copy
    serial=copy.deepcopy(case)
    serial['last_turn_timing']['portrait_worker']['timing_spans']['motion_ms']=[5.,8.]
    with pytest.raises(RuntimeError):admission_record(tmp_path,'GPU-one',1000,samples,[serial])
