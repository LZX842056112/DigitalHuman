import numpy as np


def test_playback_busy_ends_when_last_audio_sample_is_consumed():
    from backend.transport.timeline import Timeline
    clock=Timeline()
    assert not clock.busy
    token=clock.begin_turn()
    clock.append(token,np.ones(1600,np.float32),16000,np.zeros((3,100)),np.zeros((3,3)))
    assert clock.busy
    clock.read_audio(4800)
    assert not clock.busy
