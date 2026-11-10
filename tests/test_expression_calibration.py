import numpy as np
import pytest


def test_compare_frames_reports_the_changed_region():
    from scripts.expression_calibrate import compare_frames
    base = np.zeros((20, 20, 3), np.uint8)
    other = base.copy()
    other[8:12, 8:12] = 40
    result = compare_frames(base, other)
    assert result['bbox'] == [8, 8, 11, 11]
    assert result['changed_ratio'] == pytest.approx(16/400)
    assert result['mean_abs_diff'] == pytest.approx(40.)
    assert result['border_identical'] is True


def test_compare_frames_flags_a_changed_border():
    from scripts.expression_calibrate import compare_frames
    base = np.zeros((20, 20, 3), np.uint8)
    other = base.copy()
    other[0, 0] = 90
    result = compare_frames(base, other)
    assert result['border_identical'] is False


def test_identical_frames_have_no_region():
    from scripts.expression_calibrate import compare_frames
    frame = np.zeros((10, 10, 3), np.uint8)
    result = compare_frames(frame, frame.copy())
    assert result['bbox'] is None and result['changed_ratio'] == 0.


def test_summary_rejects_a_preset_that_is_indistinguishable():
    from scripts.expression_calibrate import summarize
    frame = np.zeros((20, 20, 3), np.uint8)
    assert summarize('serious', [frame], [frame.copy()])['passed'] is False


def test_summary_rejects_a_frame_count_mismatch():
    from scripts.expression_calibrate import summarize
    frame = np.zeros((20, 20, 3), np.uint8)
    assert summarize('smile', [frame], [frame, frame])['passed'] is False
