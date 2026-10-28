from scripts.acceptance import video_rate_target_met


def test_target_accepts_one_frame_boundary_uncertainty_with_full_duration_gate():
    windows=[{'fps':14.9,'seconds':10},{'fps':15.1,'seconds':10}]
    assert video_rate_target_met(15.,20.,windows)
    assert video_rate_target_met(8999/600,600,[{'fps':15.,'seconds':10}])


def test_target_rejects_sustained_slow_video_and_large_local_drop():
    assert not video_rate_target_met(14.9,600,[{'fps':14.9,'seconds':10}])
    assert not video_rate_target_met(15.,20.,[{'fps':14.8,'seconds':10},{'fps':15.2,'seconds':10}])
