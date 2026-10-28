import math
import numpy as np


def fit_bounds(low, high, fov=35, margin=.1):
    low,high=np.asarray(low,np.float64),np.asarray(high,np.float64)
    if low.shape!=(3,) or high.shape!=(3,) or not np.isfinite([low,high]).all() or np.any(high<low):
        raise ValueError('人物边界无效')
    half=(high-low)/2
    if not 0<=margin<1 or not 0<fov<180 or max(half[:2])<=0:
        raise ValueError('相机取景参数无效')
    distance=half[2]+max(half[:2])/math.tan(math.radians(fov)/2)/(1-margin)
    return {'yaw':0.,'pitch':0.,'target':((low+high)/2).tolist(),'distance':float(distance)}


def orbit_camera(yaw, pitch, distance, target, width, height, fov=35):
    if not all(math.isfinite(float(v)) for v in (yaw, pitch, distance)) or distance <= 0:
        raise ValueError("相机参数无效")
    yaw, pitch = math.radians(yaw), math.radians(np.clip(pitch, -80, 80))
    target = np.asarray(target, np.float32)
    position = target + distance * np.array([math.cos(pitch) * math.sin(yaw), math.sin(pitch), math.cos(pitch) * math.cos(yaw)])
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 1., 0.])
    right /= np.linalg.norm(right)
    # OpenCV camera convention: x right, y down, z forward.
    down = np.cross(forward, right)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = np.stack([right, down, forward], axis=1)
    c2w[:3, 3] = position
    focal = .5 * height / math.tan(math.radians(fov) / 2)
    intr = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], np.float32)
    return c2w, intr
