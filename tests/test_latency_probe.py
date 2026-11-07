"""Latency probes require real non-silent reception and complete sample sets."""
import numpy as np
import pytest


def test_stream_continuity_gate_rejects_hidden_sentence_gaps():
    from scripts.acceptance import check_stream_continuity
    check_stream_continuity({})
    check_stream_continuity({'delivery_mode':'streaming_parallel','stream_underruns':0})
    with pytest.raises(RuntimeError):check_stream_continuity({'delivery_mode':'streaming_serial'})
    with pytest.raises(RuntimeError):check_stream_continuity({'delivery_mode':'streaming_parallel','stream_underruns':1})
    with pytest.raises(RuntimeError):check_stream_continuity({'delivery_mode':'streaming_serial','stream_underruns':0,
                                                           'stream_cleanup_errors':['文件未回收']})


def test_speech_meter_ignores_idle_and_keeps_first_non_silent_frame(monkeypatch):
    from scripts import benchmark_latency
    now=[100.]
    monkeypatch.setattr(benchmark_latency.clock,'monotonic',lambda:now[0])
    meter=benchmark_latency.SpeechMeter()
    meter.observe(np.array([300],np.int16))
    assert meter.first_ms is None
    meter.arm(100.)
    meter.observe(np.zeros(960,np.int16))
    assert meter.first_ms is None
    now[0]=100.3;meter.observe(np.array([300],np.int16))
    assert meter.first_ms==pytest.approx(300)
    now[0]=101.;meter.observe(np.array([-32768],np.int16))
    assert meter.first_ms==pytest.approx(300)
    meter.arm(102.)
    now[0]=102.1;meter.observe(np.zeros(960,np.int16))
    assert meter.first_ms is None


def test_warm_summary_uses_nearest_rank_and_excludes_cold_by_label():
    from scripts.benchmark_latency import summarize_warm
    rows=[{'phase':'warm','status':'completed','first_non_silent_receive_ms':value}
          for value in range(1,21)]
    rows.append({'phase':'cold','status':'completed','first_non_silent_receive_ms':1000})
    summary=summarize_warm(rows)
    assert summary=={'samples':20,'failed':0,'median_ms':10.5,'p95_ms':19.}


def test_failed_sample_cannot_produce_a_passing_partial_latency_summary():
    from scripts.benchmark_latency import summarize_warm
    summary=summarize_warm([
        {'phase':'warm','status':'completed','first_non_silent_receive_ms':10},
        {'phase':'warm','status':'failed','first_non_silent_receive_ms':None},
    ])
    assert summary=={'samples':2,'failed':1,'median_ms':None,'p95_ms':None}


def test_quiet_period_requires_fresh_continuous_audio_and_resets_for_late_speech(monkeypatch):
    from scripts import benchmark_latency
    now=[100.]
    monkeypatch.setattr(benchmark_latency.clock,'monotonic',lambda:now[0])
    meter=benchmark_latency.SpeechMeter();meter.observe(np.zeros(960,np.int16))
    now[0]=101.
    assert meter.quiet_for==0.,'Missing packets cannot establish silence'
    for index in range(27):
        now[0]=101.+index*.02;meter.observe(np.zeros(960,np.int16))
    assert meter.quiet_for>=.5
    now[0]+=.02;meter.observe(np.array([300],np.int16))
    assert meter.quiet_for==0.,'Late old speech must restart the quiet interval'


@pytest.mark.asyncio
async def test_quiet_wait_has_a_deadline_when_audio_stalls(monkeypatch):
    from scripts import benchmark_latency
    now=[100.]
    monkeypatch.setattr(benchmark_latency.clock,'monotonic',lambda:now[0])
    async def sleep(seconds):now[0]+=seconds
    monkeypatch.setattr(benchmark_latency.asyncio,'sleep',sleep)
    meter=benchmark_latency.SpeechMeter();meter.observe(np.zeros(960,np.int16))
    with pytest.raises(TimeoutError,match='audio|静音'):
        await benchmark_latency.wait_for_quiet(meter,timeout=.2)


@pytest.mark.parametrize('damage',['hash','question','path'])
def test_reused_audio_fixtures_require_same_questions_and_verified_files(tmp_path,damage):
    import hashlib
    import json
    from scripts import benchmark_latency
    rows=[]
    for index,question in enumerate(benchmark_latency.QUESTIONS):
        path=tmp_path/f'{index:02}.wav';path.write_bytes(b'fixed audio '+bytes([index]))
        rows.append({'question':question,'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    metadata=tmp_path/'manifest.json';metadata.write_text(json.dumps(rows))
    loaded=benchmark_latency.load_audio_fixtures(tmp_path)
    assert loaded[0]['file']==str((tmp_path/'00.wav').resolve())
    if damage=='hash':(tmp_path/'00.wav').write_bytes(b'different waveform')
    elif damage=='question':rows[0]['question']='different question';metadata.write_text(json.dumps(rows))
    else:rows[0]['file']='../outside.wav';metadata.write_text(json.dumps(rows))
    with pytest.raises(ValueError):benchmark_latency.load_audio_fixtures(tmp_path)
