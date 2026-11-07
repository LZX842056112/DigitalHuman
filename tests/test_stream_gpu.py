import pytest


def test_session_gpu_lease_only_authorizes_its_workers(tmp_path):
    from backend.gpu import GpuLease,SessionGpuLease,session_operation
    from backend.errors import NotReady
    owner=SessionGpuLease(tmp_path,'session').acquire()
    try:
        with pytest.raises(NotReady):GpuLease(tmp_path).acquire()
        with session_operation(tmp_path,'session',owner.token):pass
        with pytest.raises(NotReady):
            with session_operation(tmp_path,'other',owner.token):pass
        with pytest.raises(NotReady):
            with session_operation(tmp_path,'session','wrong'):pass
    finally:owner.release()
    with pytest.raises(NotReady):
        with session_operation(tmp_path,'session',owner.token):pass
    GpuLease(tmp_path).acquire().release()


def test_dead_or_reused_owner_is_not_authorized(tmp_path,monkeypatch):
    from backend.gpu import SessionGpuLease,session_operation
    from backend.errors import NotReady
    owner=SessionGpuLease(tmp_path,'session').acquire()
    try:
        monkeypatch.setattr('backend.gpu.process_signature',lambda pid:'another-process')
        with pytest.raises(NotReady):
            with session_operation(tmp_path,'session',owner.token):pass
    finally:owner.release()


def test_parallel_admission_requires_measured_peak_and_one_gib_margin():
    from backend.gpu import enough_stream_memory
    assert not enough_stream_memory(8*1024,None)
    assert not enough_stream_memory(4023,3000)
    assert enough_stream_memory(4024,3000)


def test_worker_keeps_gpu_locked_after_session_owner_exits(tmp_path):
    from backend.gpu import GpuLease,SessionGpuLease,session_operation
    from backend.errors import NotReady
    owner=SessionGpuLease(tmp_path,'session').acquire()
    try:
        with session_operation(tmp_path,'session',owner.token):
            owner.release()
            with pytest.raises(NotReady):GpuLease(tmp_path).acquire()
        GpuLease(tmp_path).acquire().release()
    finally:owner.release()


def test_running_parallel_job_refuses_to_start_a_stage_below_margin(tmp_path,monkeypatch):
    from backend.gpu import SessionGpuLease,check_stream_headroom
    from backend.errors import StreamMemoryUnavailable
    owner=SessionGpuLease(tmp_path,'session',concurrent=True).acquire()
    try:
        monkeypatch.setattr('backend.gpu.free_vram_mib',lambda:1023)
        with pytest.raises(StreamMemoryUnavailable):check_stream_headroom(tmp_path,'session',owner.token)
        monkeypatch.setattr('backend.gpu.free_vram_mib',lambda:1024)
        check_stream_headroom(tmp_path,'session',owner.token)
    finally:owner.release()


def test_stream_cancellation_is_sticky_and_bounds_filesystem_polling(tmp_path,monkeypatch):
    from backend.gpu import SessionGpuLease,StreamCancellation
    from backend.storage import atomic_json
    now=[0.];monkeypatch.setattr('backend.clock.monotonic',lambda:now[0])
    marker=tmp_path/'data/sessions/session/generation.json'
    atomic_json(marker,{'generation':1,'closed':False})
    owner=SessionGpuLease(tmp_path,'session').acquire()
    try:
        cancelled=StreamCancellation(tmp_path,'session',1,owner.token,lambda:False)
        assert not cancelled()
        atomic_json(marker,{'generation':2,'closed':False})
        now[0]=.04;assert not cancelled()
        now[0]=.051;assert cancelled()
        atomic_json(marker,{'generation':1,'closed':False})
        now[0]=.2;assert cancelled()
    finally:owner.release()


def test_buffered_jobs_use_same_bounded_marker_polling_without_a_shared_capability(tmp_path):
    from backend.gpu import StreamCancellation
    from backend.storage import atomic_json
    marker=tmp_path/'data/sessions/session/generation.json'
    atomic_json(marker,{'generation':1,'closed':False})
    assert not StreamCancellation(tmp_path,'session',1,None,lambda:False)()
    assert not StreamCancellation(tmp_path,None,None,None,lambda:False)()
