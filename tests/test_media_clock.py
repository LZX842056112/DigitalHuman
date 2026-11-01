import datetime


def test_media_clock_uses_elapsed_raw_time_when_adjusted_time_runs_slow(monkeypatch):
    from backend import clock
    monkeypatch.setattr(clock.time,'CLOCK_MONOTONIC_RAW',4,raising=False)
    monkeypatch.setattr(clock.time,'clock_gettime',lambda source:120.,raising=False)
    monkeypatch.setattr(clock.time,'monotonic',lambda:118.333)
    assert clock.monotonic()==120.


def test_media_clock_falls_back_on_platforms_without_raw_clock(monkeypatch):
    from backend import clock
    monkeypatch.delattr(clock.time,'CLOCK_MONOTONIC_RAW',raising=False)
    monkeypatch.setattr(clock.time,'monotonic',lambda:120.)
    assert clock.monotonic()==120.


def test_audio_and_video_rtcp_share_elapsed_time_despite_wall_clock_steps(monkeypatch):
    from aiortc import clock as rtc_clock
    from backend import clock
    elapsed=[100.];wall=[1_700_000_000.]
    monkeypatch.setattr(clock,'monotonic',lambda:elapsed[0])
    monkeypatch.setattr(clock.time,'time',lambda:wall[0])
    # Restore aiortc's process-wide hook after this isolated test.
    monkeypatch.setattr(rtc_clock,'current_datetime',rtc_clock.current_datetime)
    clock.install_rtc_clock()
    first=rtc_clock.current_ntp_time();first_ms=rtc_clock.current_ms()
    elapsed[0]+=10;wall[0]+=12
    assert (rtc_clock.current_ntp_time()-first)/(1<<32)==10
    assert rtc_clock.current_ms()-first_ms==10_000
    wall[0]-=7;elapsed[0]+=1
    assert (rtc_clock.current_ntp_time()-first)/(1<<32)==11
    assert rtc_clock.current_datetime()==datetime.datetime.fromtimestamp(1_700_000_011,datetime.timezone.utc)
