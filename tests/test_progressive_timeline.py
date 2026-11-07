import numpy as np
import pytest

from test_photo_session import movie


def test_stream_waits_for_paired_buffer_and_keeps_every_sample(tmp_path):
    from backend.transport.photo_timeline import PhotoTimeline,StreamingSentence
    timeline=PhotoTimeline(np.zeros((16,16,3),np.uint8))
    generation=timeline.begin_turn();clip=StreamingSentence()
    clip.append_audio(np.concatenate([np.full(48000,.1,np.float32),np.full(48000,.2,np.float32)]))
    first=movie(tmp_path/'first.mkv',20);second=movie(tmp_path/'second.mkv',40)
    clip.append_video(first,0,25);timeline.append(generation,clip)
    assert not np.any(timeline.read_audio(960)[0])
    assert clip.cursor==0 and clip.underruns==0
    clip.append_video(second,25,25);clip.finish_audio();clip.finish_video()
    parts=[]
    for index in range(100):
        parts.append(timeline.read_audio(960)[0])
        if index in (0,50):assert timeline.image()[0,0,0]==(20 if index==0 else 40)
    np.testing.assert_array_equal(np.concatenate(parts),clip.pcm)
    timeline.close()
    assert not first.exists() and not second.exists()


def test_late_video_pauses_both_content_cursors_and_resumes(tmp_path,monkeypatch):
    from backend.transport.photo_timeline import PhotoTimeline,StreamingSentence
    now=[0.]
    monkeypatch.setattr('backend.clock.monotonic',lambda:now[0])
    clip=StreamingSentence();clip.append_audio(np.full(192000,.1,np.float32))
    for index in range(3):
        now[0]+=.8
        clip.append_video(movie(tmp_path/f'{index}.mkv',20+index),index*25,25)
    timeline=PhotoTimeline(np.zeros((16,16,3),np.uint8));generation=timeline.begin_turn()
    timeline.append(generation,clip)
    for _ in range(150):assert np.all(timeline.read_audio(960)[0]>.09)
    cursor=clip.cursor;picture=timeline.image().copy()
    for _ in range(5):
        now[0]+=.02
        assert not np.any(timeline.read_audio(960)[0])
        np.testing.assert_array_equal(timeline.image(),picture)
    assert clip.cursor==cursor and clip.underruns==1
    clip.append_video(movie(tmp_path/'last.mkv',25),75,25)
    clip.finish_audio();clip.finish_video()
    for _ in range(50):assert np.all(timeline.read_audio(960)[0]>.09)
    assert clip.cursor==192000
    timeline.close()


def test_stream_rejects_video_gaps_duplicates_and_missing_tail(tmp_path):
    from backend.transport.photo_timeline import StreamingSentence
    clip=StreamingSentence();clip.append_audio(np.ones(96000,np.float32))
    clip.append_video(movie(tmp_path/'first.mkv'),0,25)
    with pytest.raises(ValueError):clip.append_video(tmp_path/'bad.mkv',26,25)
    with pytest.raises(ValueError):clip.append_video(tmp_path/'bad.mkv',0,25)
    clip.finish_audio()
    with pytest.raises(ValueError):clip.finish_video()
    clip.close()


def test_waiting_between_sentences_is_counted_as_a_real_buffer_underrun(tmp_path):
    from backend.transport.photo_timeline import StreamingSentence,PhotoTimeline
    timeline=PhotoTimeline(np.zeros((16,16,3),np.uint8));generation=timeline.begin_turn()
    timeline.expect_streams(generation,2)
    def complete(index):
        clip=StreamingSentence();clip.append_audio(np.full(48000,.1,np.float32))
        clip.append_video(movie(tmp_path/f'{index}.mkv'),0,25)
        clip.finish_audio();clip.finish_video();return clip
    timeline.append(generation,complete(0))
    for _ in range(50):timeline.read_audio(960)
    for _ in range(5):assert not np.any(timeline.read_audio(960)[0])
    assert timeline.stream_boundary_underruns==1
    timeline.append(generation,complete(1))
    for _ in range(50):assert np.any(timeline.read_audio(960)[0])
    for _ in range(5):timeline.read_audio(960)
    assert timeline.stream_boundary_underruns==1
    timeline.close()


def test_first_sentence_can_wait_for_following_sentence_buffer(tmp_path):
    from backend.transport.photo_timeline import StreamingSentence,PhotoTimeline
    allow=[False]
    clip=StreamingSentence(start_gate=lambda:allow[0]);clip.append_audio(np.full(48000,.1,np.float32))
    clip.append_video(movie(tmp_path/'first.mkv'),0,25);clip.finish_audio();clip.finish_video()
    timeline=PhotoTimeline(np.zeros((16,16,3),np.uint8));generation=timeline.begin_turn()
    timeline.append(generation,clip)
    assert not np.any(timeline.read_audio(960)[0]) and clip.cursor==0
    allow[0]=True
    assert np.any(timeline.read_audio(960)[0]) and clip.cursor==960
    timeline.close()


def test_fast_video_burst_cannot_hide_slow_audio_production(tmp_path,monkeypatch):
    from backend.transport.photo_timeline import StreamingSentence
    now=[0.]
    monkeypatch.setattr('backend.clock.monotonic',lambda:now[0])
    clip=StreamingSentence()
    for _ in range(3):
        now[0]+=5
        clip.append_audio(np.full(96000,.1,np.float32))
    for index in range(6):
        now[0]+=.05;clip.append_video(tmp_path/f'{index}.mkv',index*25,25)
    assert clip.available_samples==6*48000
    assert not clip.can_read(960),'未知总长的慢声音流必须继续缓冲'
    clip.close()


def test_fast_blocks_inside_one_motion_window_do_not_hide_inference_gaps(tmp_path,monkeypatch):
    from backend.transport.photo_timeline import StreamingSentence
    now=[0.]
    monkeypatch.setattr('backend.clock.monotonic',lambda:now[0])
    clip=StreamingSentence();clip.append_audio(np.full(318720,.1,np.float32));clip.finish_audio()
    # 6.64 seconds of speech. Recent frame blocks are fast, but another
    # diffusion window must still run before the final 41 frames exist.
    for index,stamp in enumerate((10.,12.,14.,16.,18.)):
        now[0]=stamp;clip.append_video(tmp_path/f'{index}.mkv',index*25,25)
    assert not clip.can_read(960),'不能只用窗口内的三个快块预测剩余生成时间'
    clip.close()


def test_media_end_advances_without_waiting_for_native_unload_ack(tmp_path):
    from backend.transport.photo_timeline import StreamingSentence,PhotoTimeline
    timeline=PhotoTimeline(np.zeros((16,16,3),np.uint8));generation=timeline.begin_turn()
    timeline.expect_streams(generation,2)
    clips=[]
    for index in range(2):
        clip=StreamingSentence();clip.append_audio(np.full(48000,.1+index*.1,np.float32))
        clip.append_video(movie(tmp_path/f'{index}.mkv'),0,25);clip.finish_audio()
        if index:clip.finish_video()
        timeline.append(generation,clip);clips.append(clip)
    for _ in range(50):assert np.all(timeline.read_audio(960)[0]>.09)
    assert np.all(timeline.read_audio(960)[0]>.19),'首句内容已结束，不能等待GPU卸载才进入下一句'
    assert clips[0].underruns==0 and timeline.stream_boundary_underruns==0
    assert clips[0].closed and not clips[0].video_finished
    assert timeline.update_stream(generation,clips[0],'finish_video')
    assert clips[0].video_finished
    timeline.close()


def test_video_average_uses_worker_start_instead_of_previous_sentence_wait(tmp_path,monkeypatch):
    from backend.transport.photo_timeline import StreamingSentence
    now=[0.]
    monkeypatch.setattr('backend.clock.monotonic',lambda:now[0])
    clip=StreamingSentence();clip.append_audio(np.full(192000,.1,np.float32));clip.finish_audio()
    now[0]=100.;clip.start_video()
    for index in range(3):
        now[0]+=.8;clip.append_video(tmp_path/f'{index}.mkv',index*25,25)
    assert clip.can_read(960),'上一句占用画面任务的等待不能当成本句生成速度'
    clip.close()
