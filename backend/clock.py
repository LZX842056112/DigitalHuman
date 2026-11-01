"""Elapsed media time must not follow NTP frequency slews or wall-clock steps."""
import datetime
import time


def monotonic():
    if hasattr(time,'CLOCK_MONOTONIC_RAW'):
        return time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
    return time.monotonic()


def source_name():
    return 'CLOCK_MONOTONIC_RAW' if hasattr(time,'CLOCK_MONOTONIC_RAW') else 'monotonic'


def install_rtc_clock():
    from aiortc import clock as rtc_clock
    epoch=time.time();started=monotonic()
    def utc_now():
        return datetime.datetime.fromtimestamp(epoch+monotonic()-started,datetime.timezone.utc)
    # The API permits one local media peer. Both RTP senders and their RTCP
    # reports must share this session epoch, even if the guest wall time steps.
    # Replace the hook in this process; never edit the installed aiortc package.
    rtc_clock.current_datetime=utc_now
