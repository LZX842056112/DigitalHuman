import numpy as np


def test_face_camera_recovers_known_perspective_from_observed_pixels():
    from backend.avatar.calibrate import fit_camera
    points = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1],[0,1,1],[1,1,1]],np.float64)
    observed = np.array([[256,256],[306,256],[256,306],[306,306],[256,256],[296,256],[256,296],[296,296]],np.float64)
    intrinsic = np.array([[200,0,256],[0,200,256],[0,0,1]],np.float64)
    c2w, error = fit_camera(points,observed,intrinsic)
    np.testing.assert_allclose(c2w[:3,3],[0,0,-4],atol=1e-4)
    np.testing.assert_allclose(c2w[:3,:3],np.eye(3),atol=1e-4)
    assert error < .01
