import math
import numpy as np

# SMPL-X body_pose excludes pelvis. Values are axis-angle radians.
JOINTS = {"left_hip": 0, "right_hip": 1, "left_knee": 3, "right_knee": 4,
          "left_ankle": 6, "right_ankle": 7, "left_shoulder": 15,
          "right_shoulder": 16, "left_elbow": 17, "right_elbow": 18,
          "left_wrist": 19, "right_wrist": 20}


def sample_motion(action: str, seconds: float) -> dict:
    if action not in {"idle", "walk", "wave"}:
        raise ValueError("不支持的动作")
    pose = np.zeros((21, 3), np.float32)
    root = np.zeros(3, np.float32)
    # Relax the arms from SMPL-X T-pose.
    pose[JOINTS["left_shoulder"], 2] = -.95
    pose[JOINTS["right_shoulder"], 2] = .95
    if action == "walk":
        phase = seconds * 2 * math.pi
        for side, sign in (("left", 1), ("right", -1)):
            cycle = sign * math.sin(phase)
            pose[JOINTS[f"{side}_hip"], 0] = .32 * cycle
            pose[JOINTS[f"{side}_knee"], 0] = .52 * max(0, -cycle)
            pose[JOINTS[f"{side}_ankle"], 0] = -.2 * max(0, -cycle)
            pose[JOINTS[f"{side}_shoulder"], 0] = -.18 * cycle
        root[1] = .015 * (1 - math.cos(phase * 2))
    elif action == "wave":
        pose[JOINTS["right_shoulder"], 2] = -.65
        pose[JOINTS["right_elbow"], 0] = -1.4
        pose[JOINTS["right_wrist"], 2] = .3 * math.sin(seconds * 8)
    else:
        pose[2, 0] = .01 * math.sin(seconds * 1.5)
    return {"body_pose": pose, "root_pose": np.zeros(3, np.float32), "trans": root}
