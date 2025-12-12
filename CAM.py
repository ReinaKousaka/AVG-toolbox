import argparse
import numpy as np
import matplotlib
import os, cv2
import imageio.v2 as imageio
import json

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def load_params(intr_path, extr_path):
    intr = np.load(intr_path)["intrinsic"]
    # (N,4) = fx, fy, cx, cy
    extr = np.load(extr_path)["extrinsic"]  # (N,4,4) = c2w
    intr = intr[None, :].repeat(extr.shape[0], 0)
    intr = np.stack(
        (intr[:, 0, 0], intr[:, 1, 1], intr[:, 0, 2], intr[:, 1, 2]), axis=1
    )
    assert intr.shape[0] == extr.shape[0], "intrinsics/extrinsics N mismatch"
    return intr, extr


def apply_world_axis_transform(R, t, axis_perm, axis_sign):
    """
    调整世界坐标定义，适配 xyz / xzy 等。
    axis_perm: 长度3的排列，比如 [0,1,2] 或 [0,2,1]
    axis_sign: 长度3的符号，比如 [1,1,1] 或 [1,-1,1]
    """
    P = np.asarray(axis_perm, dtype=int)
    S = np.asarray(axis_sign, dtype=float)

    # 对世界坐标的行/分量做变换
    R_new = R[:, P, :] * S[:, None]
    t_new = t[:, P] * S
    return R_new, t_new


def compute_lr_dirs_in_cam(intr, img_width=None):
    """
    基于 fx, cx 构造左右边缘方向向量（相机坐标系）。
    默认假设：z 前方，x 右。仅做水平线。
    """
    IMG_SIZE = (1280, 720)
    cx = intr[:, 2]
    cx = cx * IMG_SIZE[0] / 2
    # cy = cy * IMG_SIZE[1] / 2
    fx = intr[:, 0]
    fx = fx * IMG_SIZE[0] / 2

    if img_width is None:
        # 粗略从 cx 估计图像宽度
        img_width = int(round(float(cx[0]) * 2.0))

    left_x = 0.0
    right_x = float(img_width - 1)

    # 归一化平面上点 (x_n, 0, 1)
    xL_n = (left_x - cx) / fx
    xR_n = (right_x - cx) / fx

    dirs_L = np.stack([xL_n, np.zeros_like(xL_n), np.ones_like(xL_n)], axis=1)
    dirs_R = np.stack([xR_n, np.zeros_like(xR_n), np.ones_like(xR_n)], axis=1)

    dirs_L /= np.linalg.norm(dirs_L, axis=1, keepdims=True)
    dirs_R /= np.linalg.norm(dirs_R, axis=1, keepdims=True)
    return dirs_L, dirs_R


def cam_rays_to_world(R, dirs_cam):
    """dirs_cam: (N,3), R: (N,3,3)，返回世界坐标方向 (N,3)"""
    d = (R @ dirs_cam[..., None])[..., 0]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return d


def flatten_to_2d(t, dL_world, dR_world, flatten_axes):
    """
    把相机中心 & 左右射线拍平到 2D。
    flatten_axes: 例如 [0,2] 表示用 world.x 和 world.z。
    """
    ax0, ax1 = flatten_axes
    cam_2d = t[:, [ax0, ax1]]
    L_2d = dL_world[:, [ax0, ax1]]
    R_2d = dR_world[:, [ax0, ax1]]

    L_2d /= np.linalg.norm(L_2d, axis=1, keepdims=True)
    R_2d /= np.linalg.norm(R_2d, axis=1, keepdims=True)
    return cam_2d, L_2d, R_2d


def cross2d(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def line_intersections_batch(p0, v0, P, V, require_positive=True, eps=1e-8):
    """
    一条线 (p0, v0) 与多条线 (P[i], V[i]) 的 2D 交点（向量化）。
    返回:
      pts: (M,2)
      valid: (M,) bool
    """
    diff = P - p0[None, :]  # (M,2)
    denom = cross2d(v0[None, :], V)  # (M,)

    parallel = np.abs(denom) < eps
    safe_denom = np.where(parallel, 1.0, denom)

    t = cross2d(diff, V) / safe_denom
    u = cross2d(diff, v0[None, :]) / safe_denom

    valid = ~parallel
    if require_positive:
        valid &= (t > 0) & (u > 0)

    pts = p0[None, :] + t[:, None] * v0[None, :]
    return pts, valid


def count_valid_intersections_for_frame(
    i,
    cam_2d,
    L_2d,
    R_2d,
    max_frame_gap,
    max_cam_dist,
    min_inter_dist,
    max_inter_dist,
    min_intersections,
    require_positive,
):
    """
    对单个帧 i：
      - 只看 i 之前 max_frame_gap 内的帧
      - 在 2D 中与当前相机距离 <= max_cam_dist 的为候选
      - 当前左右线 vs 候选左右线（4种组合）一次性算交点
      - 交点距离在 [min_inter_dist, max_inter_dist] 内视为有效
      - 返回候选索引、每个候选的有效交点数、哪些候选是“有效帧”
    """
    N = cam_2d.shape[0]
    if i <= 0 or i >= N:
        return (
            np.empty(0, dtype=int),
            np.empty(0, dtype=int),
            np.empty(0, dtype=bool),
        )

    # 时间窗口
    start = max(0, i - max_frame_gap)
    cand_idx = np.arange(start, i)
    if cand_idx.size == 0:
        return cand_idx, np.zeros(0, dtype=int), np.zeros(0, dtype=bool)

    # 距离过滤
    p0 = cam_2d[i]
    cand_pos = cam_2d[cand_idx]
    dist = np.linalg.norm(cand_pos - p0[None, :], axis=1)
    mask_dist = dist <= max_cam_dist
    cand_idx = cand_idx[mask_dist]

    if cand_idx.size == 0:
        return cand_idx, np.zeros(0, dtype=int), np.zeros(0, dtype=bool)

    P = cam_2d[cand_idx]
    vL = L_2d[cand_idx]
    vR = R_2d[cand_idx]

    v0L = L_2d[i]
    v0R = R_2d[i]

    inter_counts = np.zeros(cand_idx.shape[0], dtype=int)

    # 当前 L / R × 候选 L / R
    combos = [
        (v0L, vL),
        (v0L, vR),
        (v0R, vL),
        (v0R, vR),
    ]

    for v0, Vset in combos:
        pts, valid = line_intersections_batch(
            p0, v0, P, Vset, require_positive=require_positive
        )
        if not np.any(valid):
            continue

        d = np.linalg.norm(pts - p0[None, :], axis=1)
        good = valid & (d >= min_inter_dist) & (d <= max_inter_dist)

        inter_counts += good.astype(int)

    valid_mask = inter_counts >= min_intersections
    return cand_idx, inter_counts, valid_mask


def visualize_frame(
    out_path,
    cam_2d,
    L_2d,
    R_2d,
    frame_idx,
    cand_idx,
    valid_mask,
    flatten_axes,
    max_inter_dist,
    rect_margin=2.0,
):
    """
    高效可视化（尽量向量化，减少 Python 循环）:

    - 当前帧：蓝色，点 + 实线两条相机射线
    - hit 帧：橙色，点 + 虚线两条射线
    - 在「当前帧 & 最远 hit 帧」为对角线构成的正方形（向外扩 rect_margin 米）内的
      非 hit 过去帧：灰色点 + 灰色相机射线 (alpha=0.1)
    """
    from matplotlib.collections import LineCollection

    p0 = cam_2d[frame_idx]
    hit_frames = cand_idx[valid_mask]

    # 如果没有 hit 帧，使用默认视图范围（以当前帧为中心）
    if hit_frames.size == 0:
        # 使用 max_inter_dist 的一半作为默认视图范围
        view_range = max_inter_dist * 0.6
        x_min = p0[0] - view_range
        x_max = p0[0] + view_range
        y_min = p0[1] - view_range
        y_max = p0[1] + view_range
    else:
        # 找到距离当前帧最远的 hit 帧
        hit_pos = cam_2d[hit_frames]
        dists = np.linalg.norm(hit_pos - p0[None, :], axis=1)
        furthest_hit = hit_frames[np.argmax(dists)]
        p_far = cam_2d[furthest_hit]

        # 以当前帧和最远 hit 帧为对角线，构建正方形并外扩
        x_min = min(p0[0], p_far[0])
        x_max = max(p0[0], p_far[0])
        y_min = min(p0[1], p_far[1])
        y_max = max(p0[1], p_far[1])

        side = max(x_max - x_min, y_max - y_min)
        x_max = x_min + side
        y_max = y_min + side

        x_min -= rect_margin
        x_max += rect_margin
        y_min -= rect_margin
        y_max += rect_margin

    N = cam_2d.shape[0]
    idx_all = np.arange(N)

    # 在矩形内的过去帧
    in_rect = (
        (cam_2d[:, 0] >= x_min)
        & (cam_2d[:, 0] <= x_max)
        & (cam_2d[:, 1] >= y_min)
        & (cam_2d[:, 1] <= y_max)
    )
    region_mask = (idx_all < frame_idx) & in_rect
    region_indices = idx_all[region_mask]

    # 区分 hit / 非 hit
    hit_set = set(hit_frames.tolist())
    if region_indices.size > 0:
        is_hit_region = np.isin(region_indices, hit_frames)
        nonhit_region_indices = region_indices[~is_hit_region]
    else:
        nonhit_region_indices = np.empty(0, dtype=int)

    # 开始画图
    fig, ax = plt.subplots(figsize=(7, 7))

    # 矩形边框（只作参考）
    ax.plot(
        [x_min, x_max, x_max, x_min, x_min],
        [y_min, y_min, y_max, y_max, y_min],
        linestyle="--",
        linewidth=1,
        alpha=0.3,
    )

    # 非 hit 过去帧（矩形内）：灰色点 + 灰色射线，用 LineCollection 向量化
    if nonhit_region_indices.size > 0:
        P = cam_2d[nonhit_region_indices]  # (M,2)
        dL = L_2d[nonhit_region_indices]  # (M,2)
        dR = R_2d[nonhit_region_indices]  # (M,2)

        ax.scatter(P[:, 0], P[:, 1], s=10, alpha=0.1, color="gray")

        # 两条射线段
        # scale_nonhit = max_inter_dist * 0.5
        # segs_nonhit = np.concatenate(
        #     [
        #         np.stack([P, P + dL * scale_nonhit], axis=1),
        #         np.stack([P, P + dR * scale_nonhit], axis=1),
        #     ],
        #     axis=0,
        # )
        # lc_nonhit = LineCollection(
        #     segs_nonhit,
        #     linewidths=1,
        #     alpha=0.1,
        # )
        # ax.add_collection(lc_nonhit)

    # hit 帧：橙色点 + 橙色虚线射线（向量化）
    if hit_frames.size > 0:
        P_hit = cam_2d[hit_frames]  # (K,2)
        dL_hit = L_2d[hit_frames]
        dR_hit = R_2d[hit_frames]

        ax.scatter(P_hit[:, 0], P_hit[:, 1], s=25)

        scale_hit = max_inter_dist * 0.8
        segs_hit = np.concatenate(
            [
                np.stack([P_hit, P_hit + dL_hit * scale_hit], axis=1),
                np.stack([P_hit, P_hit + dR_hit * scale_hit], axis=1),
            ],
            axis=0,
        )
        lc_hit = LineCollection(
            segs_hit, linewidths=1.5, linestyles="--", colors="orange"
        )
        lc_hit.set_alpha(0.8)
        ax.add_collection(lc_hit)

    # 当前帧：蓝色点 + 蓝色实线射线
    dL0 = L_2d[frame_idx]
    dR0 = R_2d[frame_idx]
    segs_cur = np.stack(
        [
            np.stack([p0, p0 + dL0 * max_inter_dist], axis=0),
            np.stack([p0, p0 + dR0 * max_inter_dist], axis=0),
        ],
        axis=0,
    )

    ax.scatter(p0[0], p0[1], s=60, marker="*")
    lc_cur = LineCollection(
        segs_cur,
        linewidths=1.5,
    )
    ax.add_collection(lc_cur)

    # 视图设置
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Find intersection-based valid camera frame pairs in 2D."
    )

    parser.add_argument(
        "--intr",
        type=str,
        required=True,
        help="Path to intrinsics npz (key: data, shape (N,4)).",
    )
    parser.add_argument(
        "--extr",
        type=str,
        required=True,
        help="Path to extrinsics npz (key: data, shape (N,4,4), c2w).",
    )
    parser.add_argument(
        "--flatten-axes",
        type=int,
        nargs=2,
        default=[0, 2],
        help="World axes to flatten to 2D, e.g. 0 2 for X-Z.",
    )
    parser.add_argument(
        "--world-axis-perm",
        type=int,
        nargs=3,
        default=[0, 1, 2],
        help="Permutation of world axes, e.g. 0 2 1 for x,z,y.",
    )
    parser.add_argument(
        "--world-axis-sign",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        help="Sign for each world axis after perm.",
    )
    parser.add_argument(
        "--img-width",
        type=int,
        default=832,
        help="Image width in pixels; if not set, approximate from cx.",
    )
    parser.add_argument(
        "--max-frame-gap",
        type=int,
        default=1800,
        help="Only consider frames within this many frames in the past.",
    )
    parser.add_argument(
        "--max-cam-dist",
        type=float,
        default=50,
        help="Max 2D distance (m) between cameras to be candidate.",
    )
    parser.add_argument(
        "--min-inter-dist",
        type=float,
        default=0.1,
        help="Min distance (m) from intersection to current camera.",
    )
    parser.add_argument(
        "--max-inter-dist",
        type=float,
        default=50,
        help="Max distance (m) from intersection to current camera.",
    )
    parser.add_argument(
        "--min-intersections",
        type=int,
        default=2,
        help="Min number of valid intersections to accept a frame.",
    )
    parser.add_argument(
        "--no-positive-only",
        action="store_true",
        help="If set, do NOT require t>0,u>0 (both ray directions).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="If set, do NOT require t>0,u>0 (both ray directions).",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        default="test_CAM",
        help="Output npz filename for matches & metadata.",
    )
    parser.add_argument(
        "--vis-prefix",
        type=str,
        default="vis_frame_",
        help="Prefix (also dir) for visualization PNGs.",
    )
    parser.add_argument(
        "-dc",
        "--debug-clip",
        type=int,
        default=None,
        help="Prefix (also dir) for visualization PNGs.",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default=None,
        help="Path to MP4 video to concatenate with visualization frames.",
    )

    args = parser.parse_args()

    intr, extr = load_params(args.intr, args.extr)
    N = intr.shape[0]
    if args.debug_clip is not None:
        N = min(N, args.debug_clip)
        intr = intr[:N]
        extr = extr[:N]
    extr_ = []
    # for ext in extr:
    #     extr_.append(np.linalg.inv(ext))
    # extr = np.stack(extr_, axis=0)

    R = extr[:, :3, :3]
    t = extr[:, :3, 3]
    t *= 100.0
    # 适配世界坐标
    R, t = apply_world_axis_transform(
        R,
        t,
        axis_perm=args.world_axis_perm,
        axis_sign=args.world_axis_sign,
    )

    # 相机坐标左右线
    dirs_L_cam, dirs_R_cam = compute_lr_dirs_in_cam(intr, img_width=args.img_width)

    # 转世界
    dL_world = cam_rays_to_world(R, dirs_L_cam)
    dR_world = cam_rays_to_world(R, dirs_R_cam)

    # 拍平到 2D
    cam_2d, L_2d, R_2d = flatten_to_2d(
        t,
        dL_world,
        dR_world,
        flatten_axes=args.flatten_axes,
    )

    require_positive = not args.no_positive_only

    all_candidate_indices = []
    all_intersection_counts = []
    all_valid_masks = []

    valid_pair_curr = []
    valid_pair_past = []
    valid_pair_counts = []

    from tqdm import tqdm

    basename = os.path.splitext(os.path.basename(args.intr))[0]
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "temp_frames"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "temp_frames", basename), exist_ok=True)
    cam_json = {}
    for i in tqdm(range(N)):
        cand_idx, inter_counts, valid_mask = count_valid_intersections_for_frame(
            i,
            cam_2d,
            L_2d,
            R_2d,
            max_frame_gap=args.max_frame_gap,
            max_cam_dist=args.max_cam_dist,
            min_inter_dist=args.min_inter_dist,
            max_inter_dist=args.max_inter_dist,
            min_intersections=args.min_intersections,
            require_positive=require_positive,
        )

        all_candidate_indices.append(cand_idx)
        all_intersection_counts.append(inter_counts)
        all_valid_masks.append(valid_mask)
        if np.any(valid_mask):
            v_idx = cand_idx[valid_mask]
            v_cnt = inter_counts[valid_mask]
            valid_pair_curr.append(np.full_like(v_idx, i, dtype=int))
            valid_pair_past.append(v_idx.astype(int))
            valid_pair_counts.append(v_cnt.astype(int))
            cam_json[i] = v_idx.tolist()
            if args.verbose:
                out_path = os.path.join(
                    args.out_dir,
                    "temp_frames",
                    basename,
                    f"{i:06d}.png",
                )
                visualize_frame(
                    out_path=out_path,
                    cam_2d=cam_2d,
                    L_2d=L_2d,
                    R_2d=R_2d,
                    frame_idx=i,
                    cand_idx=cand_idx,
                    valid_mask=valid_mask,
                    flatten_axes=args.flatten_axes,
                    max_inter_dist=args.max_inter_dist,
                )
    with open(os.path.join(args.out_dir, f"{basename}.json"), "w") as f:
        json.dump(cam_json, f, indent=4)
    if args.verbose:
        vis_dir = os.path.join(args.out_dir, "temp_frames", basename)
        # 收集所有 png 按文件名排序
        png_files = sorted(f for f in os.listdir(vis_dir) if f.lower().endswith(".png"))
        if not png_files:
            return

        mp4_path = os.path.join(args.out_dir, f"{basename}.mp4")

        # 若已有 mp4 且比所有 png 都新，则不重复生成
        latest_png_mtime = max(
            os.path.getmtime(os.path.join(vis_dir, f)) for f in png_files
        )
        if os.path.exists(mp4_path):
            if os.path.getmtime(mp4_path) >= latest_png_mtime:
                return

        # 生成/更新 mp4
        # 读取视频帧（如果提供了视频路径）
        video_frames = None
        if args.video_path and os.path.exists(args.video_path):
            try:
                video_reader = imageio.get_reader(args.video_path)
                video_frames = []
                for vframe in video_reader:
                    video_frames.append(np.array(vframe))
                video_reader.close()
                print(f"Loaded {len(video_frames)} frames from {args.video_path}")
            except Exception as e:
                print(f"Warning: Failed to load video {args.video_path}: {e}")
                video_frames = None

        with imageio.get_writer(mp4_path, fps=10, codec="libx264") as writer:
            for idx, fname in enumerate(png_files):
                frame = imageio.imread(os.path.join(vis_dir, fname))
                # draw the fname on the frame
                frame = np.array(frame)
                frame = cv2.putText(
                    frame,
                    fname,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

                # 如果有视频帧，将其拼接到右侧
                if video_frames is not None and len(video_frames) > 0:
                    # 使用循环或取模处理帧数不匹配的情况
                    video_frame = video_frames[idx % len(video_frames)]

                    # 调整视频帧高度以匹配 PNG 帧
                    h_png = frame.shape[0]
                    h_vid, w_vid = video_frame.shape[:2]

                    # 按比例缩放视频帧
                    scale = h_png / h_vid
                    new_w_vid = int(w_vid * scale)
                    video_frame_resized = cv2.resize(video_frame, (new_w_vid, h_png))

                    # 确保通道数匹配
                    # if video_frame_resized.shape[2] != frame.shape[2]:
                    #     if video_frame_resized.shape[2] == 4:  # RGBA -> RGB
                    #         video_frame_resized = video_frame_resized[:, :, :3]
                    #     elif video_frame_resized.shape[2] == 1:  # Gray -> RGB
                    video_frame_resized = cv2.cvtColor(
                        video_frame_resized, cv2.COLOR_RGB2RGBA
                    )

                    # 水平拼接
                    frame = np.concatenate([frame, video_frame_resized], axis=1)

                writer.append_data(frame)


if __name__ == "__main__":
    main()
