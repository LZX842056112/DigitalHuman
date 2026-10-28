import itertools
import numpy as np
from backend.avatar import camera


def test_full_body_bounds_project_inside_the_camera_with_margin():
    low=np.array([-.6,-1.3,-.4],np.float32)
    high=np.array([.6,.55,.4],np.float32)
    fitted=camera.fit_bounds(low,high)
    c2w,intr=camera.orbit_camera(fitted['yaw'],fitted['pitch'],fitted['distance'],fitted['target'],512,512)
    corners=np.array(list(itertools.product(*zip(low,high))))
    w2c=np.linalg.inv(c2w)
    points=corners@w2c[:3,:3].T+w2c[:3,3]
    pixels=points@intr.T;pixels=pixels[:,:2]/pixels[:,2:]
    assert (points[:,2]>0).all()
    assert pixels.min()>=25 and pixels.max()<=487
    np.testing.assert_allclose(fitted['target'],(low+high)/2,atol=1e-6)
