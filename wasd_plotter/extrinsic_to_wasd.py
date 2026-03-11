import numpy as np
import json
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d


def categorize_movements(npy_path, trans_thresh=0.015, rot_thresh=0.4, stride=3, sigma=1.0):
    # shape: T*4*4, w2c
    matrices = np.load(npy_path)
    num_frames = matrices.shape[0]

    smoothed = gaussian_filter1d(matrices, sigma=sigma, axis=0)

    # re-orthogonalize rotations, cause Gaussian filtering breaks SO(3)
    for i in range(num_frames):
        U, _, Vt = np.linalg.svd(smoothed[i, :3, :3])
        smoothed[i, :3, :3] = U @ Vt

    categories = []
    for i in range(0, num_frames - stride):
        M_curr = smoothed[i]  # w2c
        M_next = smoothed[i + stride]  # w2c
        # relative transformation
        rel_move = np.linalg.inv(M_curr) @ M_next

        # translation (WASD)
        t = rel_move[:3, 3]
        move_label = "nothing"
        max_t_idx = np.argmax(np.abs(t))
        if np.abs(t[max_t_idx]) > trans_thresh:
            if max_t_idx == 2:  # forward/back
                move_label = "W" if t[2] < 0 else "S"
            elif max_t_idx == 0:  # left/right
                move_label = "D" if t[0] > 0 else "A"

        # rotation (Arrow keys)
        rel_rot = R.from_matrix(rel_move[:3, :3])
        yaw, pitch, roll = rel_rot.as_euler("yxz", degrees=True)
        rot_label = "nothing"
        if max(abs(yaw), abs(pitch)) > rot_thresh:
            if abs(yaw) > abs(pitch):
                rot_label = "LEFT" if yaw > 0 else "RIGHT"
            else:
                rot_label = "UP" if pitch < 0 else "DOWN"

        categories.append({
                "frame_idx": int(i),
                "movement": move_label,
                "rotation": rot_label
        })

    return categories


if __name__ == '__main__':
    results = categorize_movements("demo-pose.npy", stride=3, sigma=1.2)
    with open("keys.json", "w") as f:
        json.dump(results, f, indent=4)
    print('done')
