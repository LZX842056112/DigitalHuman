from pathlib import Path

import numpy as np
import pytest


def test_stream_paths_are_local_project_scoped_and_reject_links(tmp_path, monkeypatch):
    from backend.stream_jobs import stream_folder
    monkeypatch.setattr('tempfile.gettempdir', lambda: str(tmp_path/'native'))
    first=stream_folder(tmp_path/'one','session',2,0)
    assert first.is_relative_to(tmp_path/'native')
    assert first != stream_folder(tmp_path/'two','session',2,0)
    with pytest.raises(ValueError):stream_folder(tmp_path,'../outside',2,0)
    with pytest.raises(ValueError):stream_folder(tmp_path,'session',-1,0)
    first.mkdir(parents=True)
    linked=first.parent/'3_0'
    linked.symlink_to(first, target_is_directory=True)
    with pytest.raises(ValueError):stream_folder(tmp_path/'one','session',3,0)


def test_audio_blocks_are_atomic_contiguous_and_verified(tmp_path):
    from backend.stream_jobs import AudioSpool
    spool=AudioSpool(tmp_path,24000)
    event=spool.append(np.arange(300,dtype=np.float32)/300)
    assert event['chunk_index']==0 and event['sample_start']==0
    np.testing.assert_array_equal(spool.read(event),np.arange(300,dtype=np.float32)/300)
    with pytest.raises(ValueError):spool.read({**event,'sample_start':1})
    with pytest.raises(ValueError):spool.append(np.array([np.nan],np.float32))
    spool.finish()
    assert spool.state()['finished'] and spool.state()['samples']==300
    with pytest.raises(ValueError):spool.append(np.ones(3,np.float32))
    assert not list(tmp_path.glob('.*.tmp*'))


def test_stream_disk_guard_checks_host_and_temporary_volume(tmp_path,monkeypatch):
    from backend.stream_jobs import check_space
    from collections import namedtuple
    usage=namedtuple('usage','total used free')
    monkeypatch.setattr('shutil.disk_usage',lambda path:usage(100,90,9*1024**3))
    with pytest.raises(OSError,match='10'):check_space(tmp_path,tmp_path)


def test_native_source_snapshot_still_checks_mounted_user_data_volume(tmp_path,monkeypatch):
    from backend.stream_jobs import check_space
    from collections import namedtuple
    usage=namedtuple('usage','total used free')
    snapshot=tmp_path/'source';snapshot.mkdir()
    data=tmp_path/'mounted';data.mkdir();(snapshot/'data').symlink_to(data,target_is_directory=True)
    def disk(path):return usage(100,0,(9 if Path(path).resolve()==data else 20)*1024**3)
    monkeypatch.setattr('shutil.disk_usage',disk)
    with pytest.raises(OSError,match='10'):check_space(snapshot,tmp_path)

