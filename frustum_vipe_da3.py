"""
GPU-accelerated version of depth-overlap assignment.
- Keeps the original I/O, visualization, and pre-sparse logic.
- Replaces `_winners_one_pair` with CUDA-parallel implementations.
- Batches the window search over source frames on GPU to minimize Python loops.

Requirements:
- PyTorch >= 2.0 (for scatter_reduce_). Falls back to CPU if CUDA not available.
- numpy, scipy, matplotlib, imageio, opencv-python, pillow, tqdm

Author: (ported for GPU by ChatGPT)
"""

from __future__ import annotations
import os
import cv2, random
import math
import copy
import json
import time
import traceback
import subprocess
import numpy as np
from typing import List, Tuple, Optional
from tqdm import tqdm
from skvideo.io import vwrite

# --- plotting & utils (unchanged from original where possible)
import matplotlib

matplotlib.use("Agg")  # no display
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - needed for 3D projection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# import matplotlib.patches as patches
import matplotlib.colors as mcolors
import imageio
from PIL import Image
from scipy.ndimage import convolve

# torch (GPU)
import torch


def lexsort(keys, dim=-1):
    if keys.ndim < 2:
        raise ValueError(f"keys must be at least 2 dimensional, but {keys.ndim=}.")
    if len(keys) == 0:
        raise ValueError(f"Must have at least 1 key, but {len(keys)=}.")

    idx = keys[0].argsort(dim=dim, stable=True)
    for k in keys[1:]:
        idx = idx.gather(dim, k.gather(dim, idx).argsort(dim=dim, stable=True))

    return idx


# ========================= Helper utils (numpy) ========================= #
def fill_inf_with_neighbor_mean(a: np.ndarray, max_iters: int = 10):
    """
    Fill +/-inf using mean of finite neighbors in a 3x3 kernel (iterative diffusion).
    """
    a = a.copy()
    K = np.ones((3, 3), dtype=np.float32)
    for _ in range(max_iters):
        finite = np.isfinite(a)
        if finite.all():
            break
        sum_nb = convolve(np.where(finite, a, 0.0), K, mode="constant", cval=0.0)
        cnt_nb = convolve(finite.astype(np.float32), K, mode="constant", cval=0.0)
        mean_nb = sum_nb / np.maximum(cnt_nb, 1.0)
        to_fill = (~finite) & (cnt_nb > 0)
        a[to_fill] = mean_nb[to_fill]
    return a


def ensure_pixel_intrinsics(K: np.ndarray, W: int, H: int) -> np.ndarray:
    K = K.copy().astype(float)
    if len(K.shape) > 2:
        K = K[0]
    if K[0, 2] <= 2 and K[1, 2] <= 2 and K[0, 0] <= 2 and K[1, 1] <= 2:
        K[0, 0] *= W
        K[1, 1] *= H
        K[0, 2] *= W
        K[1, 2] *= H
    return K


def invert_pose(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Tinv = np.eye(4)
    Tinv[:3, :3] = R.T
    Tinv[:3, 3] = -R.T @ t
    return Tinv


def project_world_to_camera(
    points_world: np.ndarray,
    w2c: np.ndarray,
    K: np.ndarray,
    clamp_z=1e-9,
    return_cam=False,
):
    if points_world is None or len(points_world) == 0:
        if return_cam:
            return np.zeros((0,)), np.zeros((0,)), np.zeros((0,)), np.zeros((0, 3))
        return np.zeros((0,)), np.zeros((0,)), np.zeros((0,))
    n = points_world.shape[0]
    pts_h = np.hstack([points_world, np.ones((n, 1))])
    cam_h = (w2c @ pts_h.T).T
    cam = cam_h[:, :3]
    x, y, z = cam[:, 0], cam[:, 1], cam[:, 2]
    z_safe = np.maximum(z, float(clamp_z))
    uvw = (K @ np.vstack([x, y, z_safe])).T
    u = uvw[:, 0] / uvw[:, 2]
    v = uvw[:, 1] / uvw[:, 2]
    if return_cam:
        return u, v, z, cam
    return u, v, z


def points_inside_image(u, v, z, W, H, zmin=1e-6, zmax=np.inf):
    return (z > zmin) & (z < zmax) & (u >= 0) & (u < W) & (v >= 0) & (v < H)


def _reorder_vec_to_xyz(vec3, order="xyz"):
    order = order.lower()
    assert set(order) == {"x", "y", "z"} and len(order) == 3
    idx = {axis: order.index(axis) for axis in "xyz"}
    return np.array([vec3[idx["x"]], vec3[idx["y"]], vec3[idx["z"]]], dtype=float)


def fix_extrinsic_translation_axis_and_scale(
    T, trans_axis_order="xyz", trans_scale=1.0, flip_up_sign=False
):
    Tout = T.copy().astype(float)
    t_given = Tout[:3, 3].copy()
    t_xyz = _reorder_vec_to_xyz(t_given, order=trans_axis_order)
    if flip_up_sign:
        t_xyz[2] = -t_xyz[2]
    Tout[:3, 3] = t_xyz * float(trans_scale)
    return Tout


# ========================= Visualization (numpy/mpl) ========================= #
# (identical to original for brevity; only essentials kept)


def _normalize01(x, eps=1e-9):
    x = np.asarray(x, dtype=float)
    lo, hi = np.min(x), np.max(x)
    if hi - lo < eps:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo + eps)


def _choose_colors(pts, mode="z", contour_bands=None, cmap_name="turbo", time_idx=None):
    cmap = plt.get_cmap(cmap_name)
    if mode in ("t", "time"):
        v = (
            _normalize01(np.asarray(time_idx, dtype=float))
            if time_idx is not None
            else np.zeros((pts.shape[0],), dtype=float)
        )
        if contour_bands is not None and contour_bands > 1:
            v = np.floor(v * contour_bands) / (contour_bands - 1)
            v = np.clip(v, 0.0, 1.0)
        colors = cmap(v)
        return colors
    if mode in ("x", "y", "z"):
        axis_idx = {"x": 0, "y": 1, "z": 2}[mode]
        v = _normalize01(pts[:, axis_idx])
        if contour_bands is not None and contour_bands > 1:
            v = np.floor(v * contour_bands) / (contour_bands - 1)
            v = np.clip(v, 0.0, 1.0)
        colors = cmap(v)
        return colors
    xy_angle = (np.arctan2(pts[:, 1], pts[:, 0]) + np.pi) / (2 * np.pi)
    z_norm = _normalize01(pts[:, 2])
    if contour_bands is not None and contour_bands > 1:
        z_norm = np.floor(z_norm * contour_bands) / (contour_bands - 1)
        z_norm = np.clip(z_norm, 0.0, 1.0)
    hsv = np.stack([xy_angle, np.ones_like(z_norm), z_norm], axis=1)
    rgb = mcolors.hsv_to_rgb(hsv)
    colors = np.concatenate([rgb, np.ones((rgb.shape[0], 1))], axis=1)
    return colors


def _sample_points_for_viz(Pw_list, stride=1, max_points=200000, rng=None):
    rng = np.random.default_rng(None if rng is None else rng)
    per_frame, per_frame_ts, counts = [], [], []
    for fi, Pw in enumerate(Pw_list):
        if Pw is None or len(Pw) == 0:
            per_frame.append(np.zeros((0, 3), dtype=float))
            per_frame_ts.append(np.zeros((0,), dtype=int))
            counts.append(0)
            continue
        arr = Pw[:: max(1, int(stride))]
        per_frame.append(arr)
        per_frame_ts.append(np.full((arr.shape[0],), int(fi), dtype=int))
        counts.append(arr.shape[0])
    total = int(np.sum(counts))
    if total == 0:
        return np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=int)
    if (max_points is None) or (max_points <= 0) or (total <= max_points):
        return (np.concatenate(per_frame, axis=0), np.concatenate(per_frame_ts, axis=0))
    weights = np.array(counts, dtype=float)
    weights = weights / np.maximum(1, weights.sum())
    alloc = np.ceil(weights * max_points).astype(int)
    picked, picked_ts, remain = [], [], max_points
    for arr, ts_arr, k in zip(per_frame, per_frame_ts, alloc):
        if arr.shape[0] == 0 or remain <= 0:
            continue
        k = int(min(k, arr.shape[0], remain))
        idx = rng.choice(arr.shape[0], size=k, replace=False)
        picked.append(arr[idx])
        picked_ts.append(ts_arr[idx])
        remain -= k
    if len(picked) == 0:
        return np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=int)
    pts = np.concatenate(picked, axis=0)
    ts = np.concatenate(picked_ts, axis=0)
    if pts.shape[0] > max_points:
        idx = rng.choice(pts.shape[0], size=int(max_points), replace=False)
        pts = pts[idx]
        ts = ts[idx]
    return pts, ts


def visualize_sparse_pointcloud(
    Pw_list,
    mode="z",
    contour_bands=20,
    stride=1,
    max_points=200000,
    point_size=1.0,
    alpha=0.8,
    elev=20,
    azim=-60,
    figsize=(9, 7),
    save_path=None,
    title="Sparse Point Cloud (pre-sparse)",
    c2w=None,
    show_cameras=True,
    camera_stride=1,
    camera_size=10.0,
    camera_linewidth=2.0,
    camera_alpha=0.95,
    camera_cmap="viridis",
    camera_show_colorbar=True,
    annotate_start_end=True,
):
    pts, ts = _sample_points_for_viz(Pw_list, stride=stride, max_points=max_points)
    colors = (
        None
        if pts.shape[0] == 0
        else _choose_colors(
            pts, mode="t", contour_bands=None, time_idx=ts, cmap_name="viridis"
        )
    )
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    if pts.shape[0] > 0:
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            c=colors,
            s=float(point_size),
            alpha=float(alpha),
            linewidths=0,
        )
    all_for_extent = [pts] if pts.shape[0] > 0 else []
    if (
        c2w is not None
        and show_cameras
        and isinstance(c2w, np.ndarray)
        and c2w.ndim == 3
        and c2w.shape[1:] == (4, 4)
    ):
        T = c2w.shape[0]
        idxs = np.arange(0, T, max(1, int(camera_stride)))
        C = c2w[idxs, :3, 3]
        all_for_extent.append(C)
        if C.shape[0] > 0:
            cmap = plt.get_cmap(camera_cmap)
            t_norm = _normalize01(idxs.astype(float))
            cam_colors = cmap(t_norm)
            ax.scatter(
                C[:, 0],
                C[:, 1],
                C[:, 2],
                c=cam_colors,
                s=float(camera_size),
                alpha=float(camera_alpha),
                marker="o",
                depthshade=False,
                edgecolors="none",
            )
            if C.shape[0] >= 2:
                p0 = C[:-1]
                p1 = C[1:]
                segs = np.stack([p0, p1], axis=1)
                lc = Line3DCollection(
                    segs,
                    colors=cam_colors[:-1],
                    linewidths=float(camera_linewidth),
                    alpha=float(camera_alpha),
                )
                ax.add_collection3d(lc)
            if annotate_start_end and C.shape[0] >= 1:
                ax.text(C[0, 0], C[0, 1], C[0, 2], "start", color="k", fontsize=8)
                ax.text(C[-1, 0], C[-1, 1], C[-1, 2], "end", color="k", fontsize=8)
            if True and C.shape[0] >= 2:
                sm = plt.cm.ScalarMappable(
                    norm=mcolors.Normalize(vmin=float(idxs[0]), vmax=float(idxs[-1])),
                    cmap=cmap,
                )
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.05)
                cbar.set_label("Frame index (time)")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if len(all_for_extent) > 0:
        cat = np.concatenate(all_for_extent, axis=0)
        ranges = cat.max(axis=0) - cat.min(axis=0)
        ranges = np.where(ranges < 1e-9, 1.0, ranges)
        ax.set_box_aspect(ranges)
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    if True:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
    else:
        plt.close(fig)


# ========================= Pre-sparse (numpy) ========================= #


def _compute_pw_sparse_for_all(
    depths: np.ndarray,
    Kpix: np.ndarray,
    point_stride: int,
    c2w: np.ndarray,
    H_img: int,
    W_img: int,
    Hc: int,
    Wc: int,
    depth_inf_thresh=None,
):
    Hd, Wd = depths.shape[1], depths.shape[2]
    rs = np.arange(0, Hd, point_stride)
    cs = np.arange(0, Wd, point_stride)
    uu_s, vv_s = np.meshgrid(cs + 0.5, rs + 0.5)
    scale_x = W_img / Wd
    scale_y = H_img / Hd
    u_s = uu_s.ravel() * scale_x
    v_s = vv_s.ravel() * scale_y

    Kinv = np.linalg.inv(Kpix)
    rays_cam = (Kinv @ np.stack([u_s, v_s, np.ones_like(u_s)], axis=0)).T

    cell_h = H_img / Hc
    cell_w = W_img / Wc

    T = depths.shape[0]
    Pw_list, hs_list, ws_list = [None] * T, [None] * T, [None] * T
    rays_inf_list, hs_inf_list, ws_inf_list = [None] * T, [None] * T, [None] * T
    timestep_fin_list, timestep_inf_list = [], []

    # 新增：与 Pw_list 对齐的“指向相机中心”的向量与单位方向
    vec_to_cam_list = [None] * T
    dir_to_cam_list = [None] * T
    dir_cam_center_list = [None] * T
    cam_centers = [None] * T

    for n in tqdm(range(T), desc="pre-sparse"):
        z_full = depths[n, rs[:, None], cs[None, :]].astype(float).ravel()
        z_full = np.clip(z_full, a_min=1e-6, a_max=1e6)
        valid_any = np.isfinite(z_full) & (z_full > 0)

        if not np.any(valid_any):
            Pw_list[n] = np.zeros((0, 3))
            hs_list[n] = np.zeros((0,), int)
            ws_list[n] = np.zeros((0,), int)
            rays_inf_list[n] = np.zeros((0, 3))
            hs_inf_list[n] = np.zeros((0,), int)
            ws_inf_list[n] = np.zeros((0,), int)
            # 新增：空对齐
            vec_to_cam_list[n] = np.zeros((0, 3))
            dir_to_cam_list[n] = np.zeros((0, 3))
            continue

        if depth_inf_thresh is None:
            finite_mask = valid_any
            inf_mask = np.zeros_like(finite_mask)
        else:
            finite_mask = valid_any & (z_full <= float(depth_inf_thresh))
            inf_mask = valid_any & (z_full > float(depth_inf_thresh))

        # ---------- 有限深度：相机->世界 ----------
        if np.any(finite_mask):
            Pw_cam = rays_cam[finite_mask] * z_full[finite_mask][:, None]
            Pw_h = np.hstack([Pw_cam, np.ones((Pw_cam.shape[0], 1))])
            Pw = (c2w[n] @ Pw_h.T).T[:, :3]  # 世界系点
            u_src = u_s[finite_mask]
            v_src = v_s[finite_mask]
            hs = np.clip((v_src / cell_h).astype(int), 0, Hc - 1)
            ws = np.clip((u_src / cell_w).astype(int), 0, Wc - 1)

            # ---- 新增：点到相机中心的向量 ----
            Cn = c2w[n][:3, 3]  # 相机中心(世界坐标)
            vec = (Pw - Cn[None, :]).astype(np.float32)  # 从点指向相机中心
            single_vec = (
                Pw[len(Pw) // 2][None, :].repeat(vec.shape[0], axis=0) - Cn[None, :]
            ).astype(np.float32)
            # 单位方向（避免除零）
            norm = np.linalg.norm(vec, axis=1, keepdims=True)
            norm = np.maximum(norm, 1e-9)
            dir_vec = (vec / norm).astype(np.float32)
            norm_single = np.linalg.norm(single_vec, axis=1, keepdims=True)
            norm_single = np.maximum(norm_single, 1e-9)
            dir_single_vec = (single_vec / norm_single).astype(np.float32)

            vec_to_cam_list[n] = vec
            dir_to_cam_list[n] = torch.from_numpy(dir_vec).to("cuda")
            dir_cam_center_list[n] = torch.from_numpy(dir_single_vec).to("cuda")
            cam_centers[n] = torch.from_numpy(Cn.astype(np.float32)).to("cuda")
        else:
            Pw = np.zeros((0, 3))
            hs = np.zeros((0,), int)
            ws = np.zeros((0,), int)
            vec_to_cam_list[n] = np.zeros((0, 3), dtype=np.float32)
            dir_to_cam_list[n] = np.zeros((0, 3), dtype=np.float32)

        # ---------- 无穷远：仅保存世界系射线方向 ----------
        if np.any(inf_mask):
            R = c2w[n][:3, :3]
            dirs_w = (R @ rays_cam[inf_mask].T).T
            dirs_w = dirs_w / (np.linalg.norm(dirs_w, axis=1, keepdims=True) + 1e-9)
            u_src_inf = u_s[inf_mask]
            v_src_inf = v_s[inf_mask]
            hs_inf = np.clip((v_src_inf / cell_h).astype(int), 0, Hc - 1)
            ws_inf = np.clip((u_src_inf / cell_w).astype(int), 0, Wc - 1)
        else:
            dirs_w = np.zeros((0, 3))
            hs_inf = np.zeros((0,), int)
            ws_inf = np.zeros((0,), int)

        # 原有输出
        Pw_list[n], hs_list[n], ws_list[n] = (
            torch.from_numpy(Pw).to("cuda").float(),
            torch.from_numpy(hs).to("cuda"),
            torch.from_numpy(ws).to("cuda"),
        )
        timestep_fin_list.append(np.full((Pw.shape[0],), n, dtype=int))

        rays_inf_list[n], hs_inf_list[n], ws_inf_list[n] = dirs_w, hs_inf, ws_inf
        timestep_inf_list.append(np.full((dirs_w.shape[0],), n, dtype=int))

    # 在原 9 个返回值后追加 2 个新列表
    return (
        Pw_list,
        hs_list,
        ws_list,
        rays_inf_list,
        hs_inf_list,
        ws_inf_list,
        (cell_h, cell_w),
        timestep_fin_list,
        timestep_inf_list,
        vec_to_cam_list,  # 新增
        dir_to_cam_list,  # 新增
        dir_cam_center_list,
        cam_centers,
    )


# ========================= GPU kernels (PyTorch) ========================= #


def _as_device(x: torch.Tensor | np.ndarray, device, dtype=None):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if dtype is not None:
        x = x.to(dtype)
    return x.to(device)


def _compute_occ_depth_range_gpu(
    Pw_ref_np: np.ndarray,
    w2c_t_np: np.ndarray,
    Kpix_np: np.ndarray,
    H_img: int,
    W_img: int,
    Hc: int,
    Wc: int,
    cell_h: float,
    cell_w: float,
    device,
    return_mean=False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute [zmin, zmax] per query cell from reference points (e.g., t-1 dense points)."""
    if Pw_ref_np is None or len(Pw_ref_np) == 0:
        occ_zmin = torch.full((Hc, Wc), float("inf"), device=device)
        occ_zmax = torch.full((Hc, Wc), float("-inf"), device=device)
        return occ_zmin, occ_zmax, None
    Pw = Pw_ref_np.to(torch.float32)
    w2c = _as_device(w2c_t_np, device, torch.float32)
    Kpix = _as_device(Kpix_np, device, torch.float32)

    ones = torch.ones((Pw.shape[0], 1), device=device, dtype=Pw.dtype)
    cam = (w2c @ torch.cat([Pw, ones], dim=1).T).T[:, :3]
    z = cam[:, 2]
    uvw = (Kpix @ cam.T).T
    u = uvw[:, 0] / uvw[:, 2]
    v = uvw[:, 1] / uvw[:, 2]
    inside = (z > 1e-9) & (u >= 0) & (u < W_img) & (v >= 0) & (v < H_img)
    if not torch.any(inside):
        occ_zmin = torch.full((Hc, Wc), float("inf"), device=device)
        occ_zmax = torch.full((Hc, Wc), float("-inf"), device=device)
        HWc_hq = round(Hc * Wc * cell_h * cell_w)
        occ_zmean = torch.full((HWc_hq,), float("-inf"), device=device)
        return (occ_zmin, occ_zmax, occ_zmean)
    u = u[inside]
    v = v[inside]
    z = z[inside]
    hq = torch.clamp((v / cell_h).long(), 0, Hc - 1)
    wq = torch.clamp((u / cell_w).long(), 0, Wc - 1)
    code_q = hq * Wc + wq
    HWc = Hc * Wc
    occ_zmin = torch.full((HWc,), torch.inf, device=device, dtype=z.dtype)
    occ_zmax = torch.full((HWc,), -torch.inf, device=device, dtype=z.dtype)
    occ_zmin.scatter_reduce_(0, code_q, z, reduce="amin")
    occ_zmax.scatter_reduce_(0, code_q, z, reduce="amax")
    if return_mean:
        hhq = torch.clamp((v / 1).long(), 0, Hc * cell_h - 1)
        whq = torch.clamp((u / 1).long(), 0, Wc * cell_w - 1)
        code_hq = hhq * Wc * cell_w + whq
        code_hq = code_hq.long()
        HWc_hq = round(Hc * Wc * cell_h * cell_w)
        occ_zmean = torch.full((HWc_hq,), float("-inf"), device=device)
        occ_zmean.scatter_reduce_(0, code_hq, z, reduce="amax")
        occ_zmean = occ_zmean.view(round(Hc * cell_h), round(Wc * cell_w))
    else:
        occ_zmean = None
    return (occ_zmin.view(Hc, Wc), occ_zmax.view(Hc, Wc), occ_zmean)


def _winners_window_gpu_fastpath(
    Pw: torch.Tensor,
    hs_src: torch.Tensor,
    ws_src: torch.Tensor,
    src_id_fin: torch.Tensor,
    dir_to_cam: torch.Tensor,  # 新增
    current_dir_to_cam: tuple,
    current_t: Optional[int],
    w2c_t: torch.Tensor,
    Kpix: torch.Tensor,
    H_img: int,
    W_img: int,
    Hc: int,
    Wc: int,
    cell_h: float,
    cell_w: float,
    occ_zmin: Optional[torch.Tensor],
    occ_zmax: Optional[torch.Tensor],
    occ_block_range=(0.9, 1.1),
    group_topk: Optional[int] = None,
    group_score_mode: str = "angle_diff",
    device=None,
    dir_to_cam_center_list=None,
    cam_center_list=None,
):
    """
    Finite-depth fast-path: for each query cell q, choose the smallest source-cell s.
    Returns (hh, ww, hs_best, ws_best, sid_best) tensors.
    """
    (current_dir, hs_current, ws_current, cam_center) = current_dir_to_cam
    current_dir_single = current_dir[
        len(current_dir) // 2 : len(current_dir) // 2 + 1
    ].repeat(current_dir.shape[0], 1)
    norm_single = torch.linalg.vector_norm(current_dir_single, dim=1, keepdim=True)
    current_dir_single = current_dir_single / norm_single.clamp_min(1e-12)
    current_code = hs_current * Wc + ws_current
    current_code = current_code.unsqueeze(-1).repeat(1, 3)
    max_code = max(current_code.max().item() + 1, current_code.shape[0])
    base = torch.zeros_like(current_dir[:1]).repeat_interleave(max_code, 0)
    code_base, dir_base = base.clone(), base.clone()
    code_base[: current_code.shape[0]] = current_code
    dir_base[: current_code.shape[0]] = current_dir
    base = base.scatter_reduce(0, code_base.long(), dir_base, reduce="mean")
    norm = torch.linalg.vector_norm(base, dim=1, keepdim=True)
    out = base / norm.clamp_min(1e-12)
    device = device or Pw.device
    ones = torch.ones((Pw.shape[0], 1), dtype=Pw.dtype, device=device)
    cam = (w2c_t @ torch.cat([Pw, ones], dim=1).T).T[:, :3]
    z = cam[:, 2]
    valid = z > 1e-9
    if not torch.any(valid):
        return None, None, None
    cam = cam[valid]
    z = z[valid]
    uvw = (Kpix @ cam.T).T
    u = uvw[:, 0] / uvw[:, 2]
    v = uvw[:, 1] / uvw[:, 2]
    inside = (u >= 0) & (u < W_img) & (v >= 0) & (v < H_img)
    if not torch.any(inside):
        return None, None, None
    u = u[inside]
    v = v[inside]
    z = z[inside]
    hs = hs_src[valid][inside]
    ws = ws_src[valid][inside]
    sid = src_id_fin[valid][inside]
    dtc = dir_to_cam[valid][inside]
    sdtc = dir_to_cam_center_list[valid][inside]
    cct = cam_center_list[valid][inside]
    hq = torch.clamp((v / cell_h).long(), 0, Hc - 1)
    wq = torch.clamp((u / cell_w).long(), 0, Wc - 1)
    hqf = torch.clamp((v / cell_h), 0.0, float(Hc) - 1)
    wqf = torch.clamp((u / cell_w), 0.0, float(Wc) - 1)
    # HWc = Hc * Wc
    # code_qf = hqf * Wc + wqf
    code_q = hq * Wc + wq
    code_s = hs * Wc + ws

    if occ_zmin is not None and occ_zmax is not None:
        zmin_q = occ_zmin.view(-1)[code_q]
        zmax_q = occ_zmax.view(-1)[code_q]
        far_mask = torch.isfinite(zmax_q) & (z > zmax_q * occ_block_range[1])
        near_mask = torch.isfinite(zmin_q) & (z < zmin_q * occ_block_range[0])
        dir_current = out[code_q]
        cross_norm = torch.linalg.vector_norm(
            torch.cross(dir_current, dtc, dim=1), dim=1
        )  # ||a×b||
        dot = (dir_current * dtc).sum(dim=1)
        angles_rad = torch.atan2(cross_norm, dot)  # (N,)
        angles_deg = torch.rad2deg(angles_rad)
        keep = ~((far_mask | near_mask) & (angles_deg.abs() > 5.0))
        if not torch.any(keep):
            return None, None, None
        code_q = code_q[keep]
        hqf = hqf[keep]
        wqf = wqf[keep]
        code_s = code_s[keep]
        z = z[keep]
        sid = sid[keep]
        dtc = dtc[keep]
        sdtc = sdtc[keep]
        cct = cct[keep]

    # min s per q
    # min_s = torch.full((HWc,), fill_value=HWc + 1, device=device, dtype=torch.int64)
    idx_s = lexsort(
        torch.stack((code_s, sid * -1, code_q)), dim=0
    )  # 先按 sid 再按 pair 排序
    # q当前s过去
    sid_sorted = sid[idx_s]
    code_q_sorted = code_q[idx_s]
    hqf = hqf[idx_s]
    wqf = wqf[idx_s]
    code_s_sorted = code_s[idx_s]
    dtc_sorted = dtc[idx_s]
    sdtc_sorted = sdtc[idx_s]
    cct_sorted = cct[idx_s]
    current_dir_single = current_dir_single[code_q_sorted]
    cross_norm_sorted = torch.linalg.vector_norm(
        torch.cross(current_dir_single, sdtc_sorted, dim=1), dim=1
    )  # ||a×b||
    dot_sorted = (current_dir_single * sdtc_sorted).sum(dim=1)
    angles_rad_sorted = torch.atan2(cross_norm_sorted, dot_sorted)  # (N,)
    # maxangle = angles_rad_sorted.abs().max().item()
    # Keep everything on GPU
    if group_topk is not None and group_topk <= 0:
        group_topk = None

    # Sort by (code_q, sid, abs_angles) using torch operations
    angles_rad_sorted = torch.abs(angles_rad_sorted)
    camera_center_distance = torch.linalg.norm((cam_center - cct_sorted), axis=1)
    bias_h = hqf - (hqf.long())
    bias_w = wqf - (wqf.long())
    bias = bias_h**2 + bias_w**2
    # bias_coord = torch.abs(torch.abs(sid_sorted % Wc) - torch.abs(code_q_sorted % Wc))
    order = lexsort(
        torch.stack(
            [
                -camera_center_distance,
                -angles_rad_sorted,
                sid_sorted,
                code_q_sorted,
            ]
        ),
        dim=0,
    )
    code_q_ord = code_q_sorted[order]
    hqf = hqf[order]
    wqf = wqf[order]
    sid_ord = sid_sorted[order]
    code_s_ord = code_s_sorted[order]
    angle_ord = angles_rad_sorted[order]
    camera_center_distance_ord = camera_center_distance[order]
    # bias_w = bias_w[order]
    # bias_coord = bias_coord[order]
    if group_topk is not None:
        # Create pair encoding on GPU
        pair = (code_q_ord.long() << 32) | (sid_ord.long() & 0xFFFFFFFF)
        if pair.numel() == 0:
            keep_mask = torch.zeros((0,), dtype=torch.bool, device=device)
        else:
            # Find start of each unique pair
            start_mask = torch.ones(pair.size(0), dtype=torch.bool, device=device)
            if pair.size(0) > 1:
                start_mask[1:] = pair[1:] != pair[:-1]

            # Compute within-pair rank using cumulative count
            first_idx = torch.cummax(
                torch.where(
                    start_mask,
                    torch.arange(pair.size(0), device=device),
                    torch.zeros_like(pair, dtype=torch.long),
                ),
                dim=0,
            )[0]
            within_rank = torch.arange(pair.size(0), device=device) - first_idx
            keep_mask = within_rank < group_topk
    else:
        keep_mask = torch.ones_like(code_q_ord, dtype=torch.bool)

    code_q_keep = code_q_ord[keep_mask]
    # code_qf_ord = code_qf_ord[keep_mask]
    sid_keep = sid_ord[keep_mask]
    code_s_keep = code_s_ord[keep_mask]
    camera_center_distance_keep = camera_center_distance_ord[keep_mask].to(
        torch.float16
    )
    # bias_coord_keep = bias_coord[keep_mask]
    angle_keep_deg = angle_ord[keep_mask].to(torch.float16)

    # Use torch.unique on GPU
    unique_keys, inverse_indices, unique_counts = torch.unique(
        code_q_keep, return_inverse=True, return_counts=True
    )

    # Find first occurrence of each unique key
    unique_idx = torch.zeros_like(unique_keys, dtype=torch.long)
    for i in range(unique_keys.size(0)):
        unique_idx[i] = (inverse_indices == i).nonzero(as_tuple=True)[0][0]

    maximum = unique_counts.max().item()

    # Create output tensors on GPU first
    assign_n = torch.full((Hc, Wc, maximum), -1, dtype=torch.int16, device=device)
    assign_hs = torch.full((Hc, Wc, maximum), -1, dtype=torch.int8, device=device)
    assign_ws = torch.full((Hc, Wc, maximum), -1, dtype=torch.int8, device=device)
    assign_degree = torch.full((Hc, Wc, maximum), 0, dtype=torch.float16, device=device)
    assign_camera_distance = torch.full(
        (Hc, Wc, maximum), 0.0, dtype=torch.float16, device=device
    )
    # assign_bias_w = torch.full((Hc, Wc, maximum), 0.0, dtype=torch.int8, device=device)
    # assign_bias_coord = torch.full(
    #     (Hc, Wc, maximum), 0, dtype=torch.int16, device=device
    # )
    # sum_maximum_valid = {}
    for n in range(maximum):
        mask = unique_counts > n
        # sum_maximum_valid[n] = int(mask.sum().item())
        if not mask.any():
            break
        keys_n = unique_keys[mask]
        idx_start_n = unique_idx[mask] + n

        h_indices = keys_n // Wc
        w_indices = keys_n % Wc

        assign_n[h_indices, w_indices, n] = sid_keep[idx_start_n].to(torch.int16)
        assign_hs[h_indices, w_indices, n] = (code_s_keep[idx_start_n] // Wc).to(
            torch.int8
        )
        assign_ws[h_indices, w_indices, n] = (code_s_keep[idx_start_n] % Wc).to(
            torch.int8
        )
        assign_degree[h_indices, w_indices, n] = angle_keep_deg[idx_start_n]
        assign_camera_distance[h_indices, w_indices, n] = camera_center_distance_keep[
            idx_start_n
        ]
        # assign_bias_coord[h_indices, w_indices, n] = bias_coord_keep[
        #     torch.clip(idx_start_n, -32768, 32767)
        # ].to(torch.int16)
        # if n > 20:
        #     print(f"maximum: {maximum}, n: {n}")
        #     break
    # Convert to numpy only at the end for compatibility with downstream code
    group = {
        "n": assign_n.cpu().numpy().astype(np.int16),
        "hs": assign_hs.cpu().numpy().astype(np.int8),
        "ws": assign_ws.cpu().numpy().astype(np.int8),
        "angle": assign_degree.cpu().numpy(),
        "camdis": assign_camera_distance.cpu().numpy(),
        # "bias_coord": assign_bias_coord.cpu().numpy().astype(np.int16),
    }
    # os.makedirs("debug_logs", exist_ok=True)
    # plt.figure(figsize=(8, 4))
    # plt.bar(list(sum_maximum_valid.keys()), (sum_maximum_valid.values()))
    # plt.grid()
    # idd = int(assign_n.cpu().numpy().astype(np.int16).max() + 1)
    # plt.savefig(f"debug_logs/winner_distribution_{idd}.png")
    # plt.close()
    # plt.clf()
    # plt.cla()
    return group


def _winners_window_gpu_inf(
    rays_inf: torch.Tensor,
    hs_inf: torch.Tensor,
    ws_inf: torch.Tensor,
    src_id_inf: torch.Tensor,
    w2c_t: torch.Tensor,
    Kpix: torch.Tensor,
    H_img: int,
    W_img: int,
    Hc: int,
    Wc: int,
    cell_h: float,
    cell_w: float,
    t: int,
    device=None,
):
    """Infinity branch: choose one (q,s) by minimal source-cell index per q (fast path)."""
    device = device or rays_inf.device
    if rays_inf.numel() == 0:
        return (torch.zeros((0,), dtype=torch.int16, device=device),) * 5
    Rcw = w2c_t[:3, :3]
    d_cam = (Rcw @ rays_inf.T).T
    zc = d_cam[:, 2]
    valid = zc > 1e-9
    if not torch.any(valid):
        return None
    d_cam = d_cam[valid]
    hs = hs_inf[valid]
    ws = ws_inf[valid]
    sid = src_id_inf[valid]
    uvw = (Kpix @ d_cam.T).T
    u = uvw[:, 0] / uvw[:, 2]
    v = uvw[:, 1] / uvw[:, 2]
    inside = (u >= 0) & (u < W_img) & (v >= 0) & (v < H_img)
    if not torch.any(inside):
        return None
    u = u[inside]
    v = v[inside]
    hs = hs[inside]
    ws = ws[inside]
    sid = sid[inside]
    hq = torch.clamp((v / cell_h).long(), 0, Hc - 1)
    wq = torch.clamp((u / cell_w).long(), 0, Wc - 1)
    HWc = Hc * Wc
    code_q = hq * Wc + wq
    code_s = hs * Wc + ws
    idx_s = lexsort(
        torch.stack((code_s, sid, code_q)), dim=0
    )  # 先按 sid 再按 pair 排序
    # q当前s过去
    sid_sorted = sid[idx_s].cpu().data.numpy()
    code_q_sorted = code_q[idx_s].cpu().data.numpy()
    code_s_sorted = code_s[idx_s].cpu().data.numpy()
    unique_keys, unique_idx, unique_counts = np.unique(
        code_q_sorted, return_index=True, return_counts=True
    )
    unique_keys = unique_keys.astype(np.int32)
    unique_idx = unique_idx.astype(np.int32)
    unique_counts = unique_counts.astype(np.int32)
    maximum = unique_counts.max()
    assign_n = -np.ones((Hc, Wc, maximum), dtype=np.int32)
    assign_hs = -np.ones((Hc, Wc, maximum), dtype=np.int16)
    assign_ws = -np.ones((Hc, Wc, maximum), dtype=np.int16)
    assign_degree = np.full((Hc, Wc, maximum), np.nan, dtype=np.int16)
    for n in range(maximum):
        mask = unique_counts > n
        if not np.any(mask):
            break
        keys_n = unique_keys[mask]
        idx_start_n = unique_idx[mask] + n
        idx_end_n = idx_start_n + 1
        assign_n[keys_n // Wc, keys_n % Wc, n] = sid_sorted[idx_start_n].astype(
            np.int32
        )
        assign_hs[keys_n // Wc, keys_n % Wc, n] = (
            code_s_sorted[idx_start_n] // Wc
        ).astype(np.int16)
        assign_ws[keys_n // Wc, keys_n % Wc, n] = (
            code_s_sorted[idx_start_n] % Wc
        ).astype(np.int16)
        assign_degree[keys_n // Wc, keys_n % Wc, n] = -2
    group = {"n": assign_n, "hs": assign_hs, "ws": assign_ws, "angle": assign_degree}
    return group


# ========================= Main sequence (CPU+GPU hybrid) ========================= #


def check_depth_overlap_sequence(
    depths,
    extrinsics,
    intrinsic,
    pathify_size,
    is_c2w=True,
    trans_axis_order="xyz",
    trans_scale=1.0,
    flip_up_sign=False,
    img_size=None,
    point_stride=4,
    exclude_window=(20, 50),
    topk_per_query=1,
    clip_length=None,
    viz_sparse=False,
    viz_mode="z",
    viz_contour_bands=20,
    viz_stride=1,
    viz_max_points=200000,
    viz_point_size=1.0,
    viz_alpha=0.8,
    viz_elev=20,
    viz_azim=-60,
    viz_figsize=(9, 7),
    viz_save_path=None,
    viz_show_cameras=True,
    viz_camera_stride=1,
    viz_camera_size=10.0,
    viz_camera_linewidth=2.0,
    viz_camera_alpha=0.95,
    viz_camera_cmap="viridis",
    viz_camera_colorbar=True,
    viz_camera_annotate=True,
    viz_per_t=False,
    viz_per_t_stride=1,
    viz_per_t_all_stride=1,
    viz_per_t_all_max_points=120000,
    viz_per_t_all_point_size=2,
    viz_per_t_all_alpha=0.2,
    viz_per_t_point_size=18,
    viz_per_t_alpha=0.98,
    viz_per_t_edge=True,
    viz_per_t_edge_color="k",
    viz_per_t_ray_len=0.35,
    viz_per_t_elev=20,
    viz_per_t_azim=-60,
    viz_per_t_figsize=(10, 8),
    viz_per_t_show_target_camera=True,
    viz_per_t_frustum_scale=0.15,
    # --- strategy ---
    min_support_per_cell=1,
    depth_inf_thresh=None,
    angle_thresh_deg=None,
    occlusion_margin=0.01,
    occ_block_range=[0.9, 1.1],
    # --- FOV prefilter options ---
    use_fov_prefilter: bool = False,
    fov_prefilter_min_distance: float = 0.5,
    fov_prefilter_max_distance: float = 10.0,
    fov_prefilter_always_include_recent: int = 0,
    fov_prefilter_max_final: Optional[int] = None,
    fov_prefilter_seed: Optional[int] = None,
    fov_prefilter_axis_order: Optional[str] = None,
    # --- GPU options ---
    use_gpu: bool = True,
    window_batching: bool = True,
    max_points_per_chunk: int = 2_000_000,
    group_store_topk: Optional[int] = None,
    group_store_score_mode: str = "angle_diff",
):
    """
    GPU-parallel implementation of the original logic. Falls back to CPU if use_gpu=False or no CUDA.
    """
    depths = np.asarray(depths)
    extrinsics = np.asarray(extrinsics)
    intrinsic = np.asarray(intrinsic)
    T, Hd, Wd = depths.shape
    if pathify_size is None:
        pathify_size = (img_size[0] // 16, img_size[1] // 16)
    Hc, Wc = pathify_size
    H_img, W_img = Hd, Wd
    near_back, far_back = exclude_window
    if near_back > far_back:
        near_back, far_back = far_back, near_back
    if group_store_topk is not None:
        group_store_topk = int(group_store_topk)
        if group_store_topk <= 0:
            group_store_topk = None
    group_store_score_mode = group_store_score_mode or "angle_diff"

    # intrinsics to pixel unit
    Kpix = ensure_pixel_intrinsics(intrinsic, W_img, H_img)

    def _fixT(Tm):
        return fix_extrinsic_translation_axis_and_scale(
            Tm, trans_axis_order, trans_scale, flip_up_sign
        )

    if is_c2w:
        c2w = np.stack([_fixT(E) for E in extrinsics], axis=0)
    else:
        c2w = np.stack([invert_pose(_fixT(E)) for E in extrinsics], axis=0)
    w2c = np.stack([invert_pose(C) for C in c2w], axis=0)

    fov_prefilter_helper = None
    rng_prefilter = None
    fov_prefilter_recent = 0
    if use_fov_prefilter:
        try:
            from CAM import CameraFOVRetrieval
        except ImportError as exc:
            raise ImportError(
                "FOV prefilter requires `CameraFOVRetrieval` defined in CAM.py."
            ) from exc
        K_for_fov = Kpix.copy().astype(float)
        if W_img > 0:
            K_for_fov[0, :] = K_for_fov[0, :] / float(W_img)
        if H_img > 0:
            K_for_fov[1, :] = K_for_fov[1, :] / float(H_img)
        axis_order = (fov_prefilter_axis_order or "xyz").lower()
        fov_prefilter_helper = CameraFOVRetrieval(
            extrinsics=c2w.copy(),
            intrinsics=K_for_fov,
            is_c2w=True,
            axis_order=axis_order,
            trans_scale=1.0,
            pos_scale=1.0,
        )
        fov_prefilter_helper.use_depth_lengths = False
        rng_prefilter = (
            np.random.default_rng(int(fov_prefilter_seed))
            if fov_prefilter_seed is not None
            else None
        )
        fov_prefilter_recent = max(0, int(fov_prefilter_always_include_recent))

    def _prefilter_window_indices(indices: List[int], current_idx: int) -> List[int]:
        if not use_fov_prefilter or fov_prefilter_helper is None:
            return list(indices)
        if not indices:
            return []
        sorted_indices = sorted(indices)
        protected = set()
        if fov_prefilter_recent > 0:
            for idx in sorted_indices:
                if idx < current_idx and (current_idx - idx) <= fov_prefilter_recent:
                    protected.add(idx)
        mandatory = set(protected)
        if current_idx in sorted_indices:
            mandatory.add(current_idx)

        min_dist_val = max(0.0, float(fov_prefilter_min_distance))
        max_dist_val = fov_prefilter_max_distance
        if max_dist_val is None:
            max_dist_val = 1e6
        max_dist_val = max(min_dist_val, float(max_dist_val))

        def _pick_group(group_vals: List[int]) -> int:
            if len(group_vals) == 1:
                return int(group_vals[0])
            if rng_prefilter is not None:
                return int(rng_prefilter.choice(group_vals))
            return int(np.random.choice(group_vals))

        filtered = []
        for idx in sorted_indices:
            if idx in mandatory:
                continue
            if idx == current_idx:
                continue
            try:
                overlaps = fov_prefilter_helper.check_fov_overlap(
                    idx,
                    current_idx,
                    max_distance=max_dist_val,
                    min_distance=min_dist_val,
                )
            except Exception:
                overlaps = True
            if overlaps:
                filtered.append(int(idx))

        representatives: List[int] = []
        if filtered:
            current_group = [filtered[0]]
            for candidate_idx in filtered[1:]:
                if candidate_idx == current_group[-1] + 1:
                    current_group.append(candidate_idx)
                else:
                    representatives.append(_pick_group(current_group))
                    current_group = [candidate_idx]
            representatives.append(_pick_group(current_group))

        selected = set(mandatory)
        selected.update(representatives)
        selected = sorted(selected)

        if (
            fov_prefilter_max_final is not None
            and fov_prefilter_max_final > 0
            and len(selected) > fov_prefilter_max_final
        ):
            mandatory_sorted = sorted(mandatory)
            if len(mandatory_sorted) >= fov_prefilter_max_final:
                # 保留全部必选项（即使数量超过限制）
                return mandatory_sorted
            remaining_slots = fov_prefilter_max_final - len(mandatory_sorted)
            optional = [idx for idx in selected if idx not in mandatory]
            optional.sort(key=lambda idx: (abs(current_idx - idx), -idx))
            trimmed = mandatory_sorted + optional[:remaining_slots]
            return sorted(set(trimmed))

        return selected

    if point_stride is None:
        point_stride = max(Hd // Hc, Wd // Wc) - 1

    # pre-sparse (CPU)
    (
        Pw_list,
        hs_list,
        ws_list,
        rays_inf_list,
        hs_inf_list,
        ws_inf_list,
        (cell_h, cell_w),
        timestep_fin_list,
        timestep_inf_list,
        vec_to_cam_list,  # 新增
        dir_to_cam_list,  # 新增
        dir_cam_center_list,
        cam_centers,
    ) = _compute_pw_sparse_for_all(
        depths,
        Kpix,
        point_stride,
        c2w,
        H_img,
        W_img,
        Hc,
        Wc,
        depth_inf_thresh=depth_inf_thresh,
    )
    (Pw_Dense, _, _, _, _, _, _, _, _, _, _, _, _) = _compute_pw_sparse_for_all(
        depths,
        Kpix,
        max(1, point_stride // 2),
        c2w,
        H_img,
        W_img,
        Hc,
        Wc,
        depth_inf_thresh=1e9,
    )

    K = int(topk_per_query)
    assign_n = -np.ones((T, Hc, Wc, K), dtype=np.int32)
    assign_hs = -np.ones((T, Hc, Wc, K), dtype=np.int16)
    assign_ws = -np.ones((T, Hc, Wc, K), dtype=np.int16)
    assign_angles = np.full((T, Hc, Wc, K), np.nan, dtype=np.float32)

    # --- device selection ---
    cuda_ok = torch.cuda.is_available()
    device = torch.device("cuda") if (use_gpu and cuda_ok) else torch.device("cpu")
    grouped_info_dict = {}  # 新增
    CAM_result = {}
    # move constant matrices lazily per t
    with tqdm(range(T), desc="overlap-seq(gpu)") as pbar:
        for t in pbar:
            # reference occ range from t-1 (dense)
            occ_ref = Pw_Dense[t - 1] if t - 1 >= 0 else None
            n_hi = max(-1, t - int(near_back))
            n_lo = max(0, t - int(far_back))
            if n_hi < n_lo:
                continue
            # concat candidates in window
            window_indices = list(range(n_lo, n_hi + 1))
            window_indices = _prefilter_window_indices(window_indices, t)
            CAM_result[t] = window_indices
            Pw_fin_list = []
            hs_fin_list = []
            ws_fin_list = []
            sid_fin_list = []
            dir_to_cam_fin_list = []
            dir_to_cam_center_list = []
            cam_center_list = []
            rays_inf_cat = []
            hs_inf_cat = []
            ws_inf_cat = []
            sid_inf_list = []
            for n in window_indices:
                Pw_n = Pw_list[n]
                if Pw_n is not None and len(Pw_n) > 0:
                    Pw_fin_list.append(Pw_n)
                    hs_fin_list.append(hs_list[n])
                    ws_fin_list.append(ws_list[n])
                    dir_to_cam_fin_list.append(dir_to_cam_list[n])  # 新增
                    sid_fin_list.append(np.full((Pw_n.shape[0],), n, dtype=np.int32))
                    dir_to_cam_center_list.append(dir_cam_center_list[n])
                    cam_center_list.append(
                        cam_centers[n][None, :].repeat(Pw_n.shape[0], 1)
                    )
                Ri = rays_inf_list[n]
                if Ri is not None and len(Ri) > 0:
                    rays_inf_cat.append(Ri)
                    hs_inf_cat.append(hs_inf_list[n])
                    ws_inf_cat.append(ws_inf_list[n])
                    sid_inf_list.append(np.full((Ri.shape[0],), n, dtype=np.int32))
            current_dir = dir_to_cam_list[t]
            hs_current = hs_list[t]
            ws_current = ws_list[t]
            # if no candidates at all
            has_fin = len(Pw_fin_list) > 0
            has_inf = len(rays_inf_cat) > 0
            if not has_fin and not has_inf:
                continue

            # build occ ranges on GPU
            if device.type == "cuda":
                occ_zmin, occ_zmax, _ = _compute_occ_depth_range_gpu(
                    occ_ref, w2c[t], Kpix, H_img, W_img, Hc, Wc, cell_h, cell_w, device
                )
            else:
                # CPU fallback: compute with numpy then send to torch
                if occ_ref is None or len(occ_ref) == 0:
                    occ_zmin_np = np.full((Hc, Wc), np.inf)
                    occ_zmax_np = np.full((Hc, Wc), -np.inf)
                else:
                    u_r, v_r, z_r = project_world_to_camera(occ_ref, w2c[t], Kpix)
                    inside_r = points_inside_image(u_r, v_r, z_r, W_img, H_img)
                    occ_zmin_np = np.full((Hc, Wc), np.inf)
                    occ_zmax_np = np.full((Hc, Wc), -np.inf)
                    if np.any(inside_r):
                        u_r = u_r[inside_r]
                        v_r = v_r[inside_r]
                        z_r = z_r[inside_r]
                        hq_r = np.clip((v_r / cell_h).astype(int), 0, Hc - 1)
                        wq_r = np.clip((u_r / cell_w).astype(int), 0, Wc - 1)
                        for uu, vv, zz in zip(hq_r, wq_r, z_r):
                            occ_zmin_np[uu, vv] = min(occ_zmin_np[uu, vv], zz)
                            occ_zmax_np[uu, vv] = max(occ_zmax_np[uu, vv], zz)
                occ_zmin = torch.from_numpy(occ_zmin_np).to(device)
                occ_zmax = torch.from_numpy(occ_zmax_np).to(device)

            # tensors for camera & Kpix
            w2c_t = torch.from_numpy(w2c[t]).to(device=device, dtype=torch.float32)
            Kpix_t = torch.from_numpy(Kpix).to(device=device, dtype=torch.float32)

            # ---- finite-depth branch ----
            if has_fin:
                Pw_cat = torch.cat(
                    Pw_fin_list, 0
                )  # torch.from_numpy(np.concatenate(Pw_fin_list, 0)).to(device).float()
                hs_cat = torch.cat(hs_fin_list, 0)
                ws_cat = torch.cat(ws_fin_list, 0)
                sid_cat = torch.from_numpy(np.concatenate(sid_fin_list, 0)).to(device)
                dir_to_cam_cat = torch.cat(dir_to_cam_fin_list, 0)
                dir_to_cam_center_list = torch.cat(dir_to_cam_center_list, 0)
                cam_center_list = torch.cat(cam_center_list, 0)
                # (
                #     torch.from_numpy(np.concatenate(dir_to_cam_fin_list, 0))
                #     .to(device)
                #     .float()
                # )
                # current_dir = torch.from_numpy(current_dir).to(
                #     device=device, dtype=torch.float32
                # )
                # hs_current = torch.from_numpy(hs_current).to(device=device)
                # ws_current = torch.from_numpy(ws_current).to(device=device)
                # chunking to limit memory if needed
                grouped_info = _winners_window_gpu_fastpath(
                    Pw_cat,
                    hs_cat,
                    ws_cat,
                    sid_cat,
                    dir_to_cam_cat,  # 新增
                    (current_dir, hs_current, ws_current, cam_centers[t]),
                    t,
                    w2c_t,
                    Kpix_t,
                    H_img,
                    W_img,
                    Hc,
                    Wc,
                    cell_h,
                    cell_w,
                    occ_zmin,
                    occ_zmax,
                    occ_block_range,
                    group_topk=group_store_topk,
                    group_score_mode=group_store_score_mode,
                    device=device,
                    dir_to_cam_center_list=dir_to_cam_center_list,
                    cam_center_list=cam_center_list,
                )
                # if t2t1 is not None and t3t2 is not None:
                #     pbar.set_postfix({"t2": f"{t2t1:.3f}s", "t3": f"{t3t2:.3f}s"})
                # else:
                #     hh, ww, hs_best, ws_best, sid_best = _winners_window_gpu_general(
                #         Pw_cat,
                #         hs_cat,
                #         ws_cat,
                #         sid_cat,
                #         w2c_t,
                #         Kpix_t,
                #         H_img,
                #         W_img,
                #         Hc,
                #         Wc,
                #         cell_h,
                #         cell_w,
                #         int(min_support_per_cell),
                #         int(topk_per_query),
                #         occ_zmin,
                #         occ_zmax,
                #         occ_block_range,
                #         device,
                #     )
                # write back
                # if hh.numel() > 0:
                #     assign_n[t, hh.cpu().numpy(), ww.cpu().numpy(), 0] = (
                #         sid_best.cpu().numpy()
                #     )
                #     assign_hs[t, hh.cpu().numpy(), ww.cpu().numpy(), 0] = (
                #         hs_best.cpu().numpy()
                #     )
                #     assign_ws[t, hh.cpu().numpy(), ww.cpu().numpy(), 0] = (
                #         ws_best.cpu().numpy()
                #     )
                #     assign_angles[t, hh.cpu().numpy(), ww.cpu().numpy(), 0] = (
                #         angles_diff.cpu().numpy()
                #     )
                # 新增
            # ---- infinity branch ---- (kept simple; only fills cells still empty)
            if has_inf and grouped_info is not None:
                mask_inf = occ_zmax > depth_inf_thresh
                mask_empty = assign_n[t, :, :, 0] < 0
                mask_empty = mask_empty & mask_inf.cpu().numpy()
                if np.any(mask_empty):
                    rays_cat = (
                        torch.from_numpy(np.concatenate(rays_inf_cat, 0))
                        .to(device)
                        .float()
                    )
                    hs_ic = torch.from_numpy(np.concatenate(hs_inf_cat, 0)).to(device)
                    ws_ic = torch.from_numpy(np.concatenate(ws_inf_cat, 0)).to(device)
                    sid_ic = torch.from_numpy(np.concatenate(sid_inf_list, 0)).to(
                        device
                    )
                    group_inf = _winners_window_gpu_inf(
                        rays_cat,
                        hs_ic,
                        ws_ic,
                        sid_ic,
                        w2c_t,
                        Kpix_t,
                        H_img,
                        W_img,
                        Hc,
                        Wc,
                        cell_h,
                        cell_w,
                        t,
                        device,
                    )
                    # if hh_i.numel() > 0:
                    #     hh_i_np = hh_i.cpu().numpy()
                    #     ww_i_np = ww_i.cpu().numpy()
                    #     fill_mask = mask_empty[hh_i_np, ww_i_np]
                    #     if np.any(fill_mask):
                    #         hh_i_np = hh_i_np[fill_mask]
                    #         ww_i_np = ww_i_np[fill_mask]
                    #         assign_n[t, hh_i_np, ww_i_np, 0] = sid_bi.cpu().numpy()[
                    #             fill_mask
                    #         ]
                    #         assign_hs[t, hh_i_np, ww_i_np, 0] = hs_bi.cpu().numpy()[
                    #             fill_mask
                    #         ]
                    #         assign_ws[t, hh_i_np, ww_i_np, 0] = ws_bi.cpu().numpy()[
                    #             fill_mask
                    #         ]
                    if group_inf is not None:
                        unfilled_finites = np.all(grouped_info["n"] == -1, axis=2)
                        final_inf_fill = unfilled_finites & mask_empty
                        maximum_inf_fill = grouped_info["n"].shape[2]
                        indice = boundary_pick_indices(
                            group_inf["n"], maximum_inf_fill, t - 80
                        )
                        sid_inf = gather_l_with_invalid(group_inf["n"], indice)
                        hs_inf = gather_l_with_invalid(group_inf["hs"], indice)
                        ws_inf = gather_l_with_invalid(group_inf["ws"], indice)
                        grouped_info["n"][final_inf_fill] = sid_inf[
                            final_inf_fill
                        ].astype(np.int16)
                        grouped_info["hs"][final_inf_fill] = hs_inf[
                            final_inf_fill
                        ].astype(np.int8)
                        grouped_info["ws"][final_inf_fill] = ws_inf[
                            final_inf_fill
                        ].astype(np.int8)
                        grouped_info["angle"][final_inf_fill] = -2
                    # inf_valid = indice >= 0
                    # print(indice)
                    # grouped_info["n"][
                    #     hh_i.cpu().numpy(), ww_i.cpu().numpy(), 0
                    # ] = sid_bi.cpu().numpy()
                    # grouped_info["hs"][
                    #     hh_i.cpu().numpy(), ww_i.cpu().numpy(), 0
                    # ] = hs_bi.cpu().numpy()
                    # grouped_info["ws"][
                    #     hh_i.cpu().numpy(), ww_i.cpu().numpy(), 0
                    # ] = ws_bi.cpu().numpy()
                    # grouped_info["angle"][hh_i.cpu().numpy(), ww_i.cpu().numpy(), 0] = -2
            grouped_info_dict[f"frame_{t}"] = grouped_info
            # (optional) per-frame visualization retained from original (omitted here for brevity)
            # if viz_per_t and (t % max(1, int(viz_per_t_stride)) == 0):
            #     ...

    meta = {
        "Hc": Hc,
        "Wc": Wc,
        "topk": int(K),
        "near_back": int(near_back),
        "far_back": int(far_back),
        "H_img": int(H_img),
        "W_img": int(W_img),
        "point_stride": int(point_stride),
        # "trans_axis_order": str(trans_axis_order),
        "trans_scale": float(trans_scale),
        # "flip_up_sign": bool(flip_up_sign),
        "is_c2w": bool(is_c2w),
        # "min_support_per_cell": int(min_support_per_cell),
        "depth_inf_thresh": (
            None if depth_inf_thresh is None else float(depth_inf_thresh)
        ),
        # "angle_thresh_deg": (
        #     None if angle_thresh_deg is None else float(angle_thresh_deg)
        # ),
        # "viz_sparse": bool(viz_sparse),
        # "viz_mode": str(viz_mode),
        # "viz_contour_bands": (
        #     None if viz_contour_bands is None else int(viz_contour_bands)
        # ),
        # "viz_stride": int(max(1, viz_stride)),
        # "viz_max_points": int(viz_max_points) if viz_max_points is not None else None,
        # "viz_show_cameras": bool(viz_show_cameras),
        # "viz_camera_stride": int(max(1, viz_camera_stride)),
        # "viz_camera_cmap": str(viz_camera_cmap),
        # # per-t subset not recorded for brevity
        # "use_gpu": bool(use_gpu),
        "device": str(device),
        # "grouped_info_dict": ,
        "group_selection_mode": str(group_store_score_mode),
        "group_store_topk": (
            None if group_store_topk is None else int(group_store_topk)
        ),
        # "CAM_result": CAM_result,
    }
    return (
        assign_n,
        assign_hs,
        assign_ws,
        assign_angles,
        grouped_info_dict,
        CAM_result,
        meta,
    )


def top_l_indices(a: np.ndarray, l: int) -> np.ndarray:
    """
    a: shape = (H, W, N)，每条序列先严格递增，随后全为 -1
    l: 需要的个数

    返回:
        idx: shape = (H, W, l) 的整数数组。
             对于每个 (h,w)，给出“最大”的 l 个非 -1 元素在第 3 维的索引；
             若不足 l 个，则在末尾以 -1 补齐。
             若全为 -1，则该行返回全 -1。
    """
    H, W, N = a.shape
    valid_counts = (a != -1).sum(axis=2)  # 每条序列非 -1 的数量 (H, W)
    k = np.minimum(valid_counts, l)  # 实际可返回的个数 (H, W)

    start = valid_counts - k  # 这 l 个里最小索引的起点 (H, W)
    rng = np.arange(l)  # (l,)
    # 先生成起点起的连续 l 个位置，再用 mask 把多出来的尾部置为 -1
    idx = start[..., None] + rng  # (H, W, l)
    mask = rng[None, None, :] < k[..., None]  # (H, W, l)
    idx = np.where(mask, idx, -1).astype(np.int64)  # 不足的在末尾补 -1
    return idx


# 上面下面两个function都是为了infinite branch最后挑选出和finite branch最大量的点存在的


def boundary_pick_indices(a: np.ndarray, l: int, B: float) -> np.ndarray:
    """
    a: 形状 (H, W, N)，每条序列先严格递增，随后全为 -1
    l: 需要的个数
    B: 界限

    返回:
        idx: 形状 (H, W, l) 的整数数组。每个 (h,w) 一行索引，表示
             - 若存在 < B：从最靠右的 < B 开始，取该处及其后的 l 个候选（越界/无效补 -1）
             - 否则若存在 == B：从最靠右的 == B 开始同理
             - 否则（全都 > B）：取最后的 l 个有效元素（不足补 -1）
             - 若全为 -1：返回全 -1
    """
    H, W, N = a.shape
    valid = a != -1  # (H,W,N)
    valid_counts = valid.sum(axis=2)  # (H,W) 每条序列有效个数

    # ---- 计算“从右往左”的第一个 <B 与 ==B 的位置（即最靠右的） ----
    idx_range = np.arange(N)  # (N,)

    # 对于布尔掩码，取“最靠右 True”的位置 = 每点沿轴2的最大 idx(True)，无 True 则为 -1
    def rightmost_index(mask: np.ndarray) -> np.ndarray:
        cand = np.where(mask, idx_range, -1)  # True -> idx，False -> -1
        return cand.max(axis=2)  # (H,W)，若全 False 得到 -1

    rightmost_lt = rightmost_index(valid & (a < B))  # (H,W)，无则 -1
    rightmost_eq = rightmost_index(valid & (a == B))  # (H,W)，无则 -1

    # ---- 回退策略（全都 > B）：取最后的 l 个有效元素 ----
    k_last = np.minimum(valid_counts, l)  # 实际可取个数
    fallback_start = valid_counts - k_last  # 从最后 k_last 的起点
    # 注意：若 valid_counts==0，则 k_last==0，fallback_start==0；后续掩码会把它们全部置为 -1

    # ---- 合成起始下标 start_idx ----
    # 优先 <B，其次 ==B，最后回退
    start_idx = np.where(
        rightmost_lt != -1,
        rightmost_lt,
        np.where(rightmost_eq != -1, rightmost_eq, fallback_start),
    )  # (H,W)

    # ---- 生成长度为 l 的连续索引，并按有效性/越界置 -1 ----
    steps = np.arange(l)  # (l,)
    idx = start_idx[..., None] + steps  # (H,W,l)

    # 合法条件：
    # 1) 起点是有效的情形（对回退与 <B/==B 都是 >=0）
    # 2) idx < valid_counts（不可越过最后一个有效位置）
    # 3) idx < N（防止越界）
    ok = (start_idx[..., None] >= 0) & (idx < valid_counts[..., None]) & (idx < N)
    idx = np.where(ok, idx, -1).astype(np.int64)
    return idx


# ========================= I/O helpers (mostly unchanged) ========================= #


def _load_depth_npz_auto(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    d = data["depths"] if "depths" in data else data["depth"]
    return d


def load_video(file_path):
    reader = imageio.get_reader(file_path)
    try:
        num_frames = int(reader.count_frames())
    except Exception:
        frames = []
        for frame in reader:
            frames.append(np.asarray(Image.fromarray(frame)))
        reader.close()
        return frames
    frames = []
    for frame_id in range(num_frames):
        frame = reader.get_data(frame_id)
        frame = Image.fromarray(frame)
        frames.append(np.asarray(frame))
    reader.close()
    return frames


def save_video(filename: str, video, fps: int = 30) -> None:
    first_frame = video[0]
    if isinstance(first_frame, Image.Image):
        if first_frame.mode != "RGB":
            first_frame = first_frame.convert("RGB")
        first_array = np.array(first_frame)
    else:
        first_array = np.array(first_frame, dtype=np.uint8)
        if len(first_array.shape) == 2:
            first_array = np.stack([first_array] * 3, axis=-1)
    height, width = first_array.shape[:2]
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "rgb24",
        "-r",
        "60",
        "-i",
        "-",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        filename,
    ]
    process = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    for frame in video:
        if isinstance(frame, Image.Image):
            if frame.mode != "RGB":
                frame = frame.convert("RGB")
            array = np.array(frame, dtype=np.uint8)
        else:
            array = np.array(frame, dtype=np.uint8)
            if len(array.shape) == 2:
                array = np.stack([array] * 3, axis=-1)
        if array.shape[0] != height or array.shape[1] != width:
            array = cv2.resize(array, (width, height))
        process.stdin.write(array.tobytes())
    process.stdin.close()
    process.wait()


def gather_l_with_invalid(a: np.ndarray, idx: np.ndarray, fill_value=-1):
    """
    a   : shape = (H, W, N)
    idx : shape = (H, W, l)，要在 axis=2 上按位置取值；-1 代表无效索引
    fill_value: 无效索引处的填充值（默认 -1）

    返回:
        out : shape = (H, W, l)
    """
    if a.ndim != 3 or idx.ndim != 3:
        raise ValueError("a must be (H,W,N) and idx must be (H,W,l)")
    if a.shape[:2] != idx.shape[:2]:
        raise ValueError("Leading (H,W) of a and idx must match")

    H, W, N = a.shape

    # 将无效/越界索引标记出来；-1 或任何越界都视为无效
    invalid = (idx < 0) | (idx >= N)

    # 为防越界，先 clip 到 [0, N-1] 再 take
    idx_clip = np.clip(idx, 0, max(N - 1, 0))
    gathered = np.take_along_axis(a, idx_clip, axis=2)  # 形状 (H, W, l)

    # 无效位置填充 fill_value
    fv = np.asarray(fill_value, dtype=a.dtype)
    out = np.where(invalid, fv, gathered)
    return out


def _write_grouped_diff_video(
    video_path,
    out_path,
    assign_n,
    assign_hs,
    assign_ws,
    assign_angles,
    near_back,
    far_back,
    video_range=None,
    cell_px=18,
    fps=24,
    overlay_text=True,
    clip_video=None,
    img_size=None,
    is_megasam=False,
    write_related=False,
    grouped_info_dict=None,
    group_selection_mode="angle_diff",
):
    if type(assign_n) != type(None):
        T, Hc, Wc, K = assign_n.shape
    else:
        T = len(grouped_info_dict.keys()) + 1
        Hc, Wc, K = grouped_info_dict["frame_1"].item()["n"].shape
    grouped_info_dict = grouped_info_dict or {}

    def _safe_int(key):
        if isinstance(key, int):
            return key
        elif isinstance(key, str):
            return int(key.split("_")[-1])

    def _parse_patch_key(patch_key):
        try:
            h_part, w_part = patch_key.split("w", 1)
            h_idx = int(h_part[1:])
            w_idx = int(w_part)
            return h_idx, w_idx
        except (ValueError, IndexError):
            return None, None

    def _build_out_path(base_path, offset):
        base, ext = os.path.splitext(base_path)
        if not ext:
            ext = ".mp4"
        return f"{base}_nminus{offset}{ext}"

    def last_valid_pos(a: np.ndarray, limit: float, b=None) -> np.ndarray:
        """
        a: shape = (H, W, N)，每条序列先严格递增，随后为 -1 填充
        limit: 上限

        返回 shape=(H, W) 的整数数组 idx，表示每条序列中 ≤ limit 的最后一个索引 n；
        若整条序列都没有 ≤ limit 的合法值，则该位置返回 -1。
        """
        # 合法且不超过上限
        mask = (a >= 0) & (a <= limit)  # (H, W, N)
        if not isinstance(b, type(None)):
            masked_vals = np.where(mask, b, np.inf)
            new_mask = mask.any(axis=-1)  # (H, W), logical OR over time axis
            # 为了在相同最小值时选择“最后一个”位置，使用反转索引再折返
            rev_idx = (
                masked_vals.shape[-1] - 1 - np.argmin(masked_vals[..., ::-1], axis=-1)
            )
            out = rev_idx
            return np.where(new_mask, out, -1)
        counts = mask.sum(axis=2)  # (H, W), 计数即“最后一个位置 + 1”
        idx = counts - 1  # (H, W), 如果 counts==0 则为 -1
        return idx.astype(np.int64)

    def gather_with_invalid(a: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """
        从形状为 (H, W, N) 的数组 a 中，根据 (H, W) 的索引 idx 取出对应值。
        若 idx[h,w] == -1，则返回值为 -1。

        参数：
            a   : np.ndarray，形状 (H, W, N)
            idx : np.ndarray，形状 (H, W)，包含索引（可能为 -1）

        返回：
            result : np.ndarray，形状 (H, W)
        """
        H, W, N = a.shape
        # 先 clip 防止越界
        idx_clip = np.clip(idx, 0, N - 1)
        # 利用 take_along_axis 取对应值
        gathered = np.take_along_axis(a, idx_clip[..., None], axis=2)[..., 0]
        # 对无效索引（-1）赋值为 -1
        result = np.where(idx == -1, -1, gathered)
        return result

    def _compute_scores(angle_rad, time_diff, mode):
        mode = (mode or "angle_diff").lower()
        angle_abs = np.abs(angle_rad)
        if mode in ("angle", "angle_diff"):
            score = angle_abs
        elif mode in ("time", "time_diff"):
            score = time_diff
        elif mode in ("angle_diff*time_diff", "angle*time"):
            score = angle_abs * time_diff
        elif mode in ("angle_diff+time_diff", "angle+time"):
            score = angle_abs + time_diff
        else:
            score = angle_abs
        return np.where(np.isnan(score), np.inf, score)

    def _prepare_assignments_from_group(offset, mode):
        assign_n_sel = np.full((T, Hc, Wc, 1), -1, dtype=np.int32)
        assign_h_sel = np.full((T, Hc, Wc, 1), -1, dtype=np.int16)
        assign_w_sel = np.full((T, Hc, Wc, 1), -1, dtype=np.int16)
        assign_a_sel = np.full((T, Hc, Wc, 1), np.nan, dtype=np.float32)
        assign_b_sel = np.full((T, Hc, Wc, 2), 30000, dtype=np.int16)
        for t_key, patches in grouped_info_dict.items():
            # print(t_key)
            if isinstance(patches, np.ndarray):
                patches = patches.item()
            t_idx = _safe_int(t_key)
            valid = last_valid_pos(patches["n"], t_idx - offset, 1 / patches["n"])
            # gather_with_invalid(patches["n"], valid)
            assign_n_sel[t_idx] = gather_with_invalid(patches["n"], valid)[..., None]
            assign_h_sel[t_idx] = gather_with_invalid(patches["hs"], valid)[..., None]
            assign_w_sel[t_idx] = gather_with_invalid(patches["ws"], valid)[..., None]
            assign_a_sel[t_idx] = gather_with_invalid(patches["angle"], valid)[
                ..., None
            ]
            assign_b_sel[t_idx] = gather_with_invalid(patches["camdis"], valid)[
                ..., None
            ]

        return assign_n_sel, assign_h_sel, assign_w_sel, assign_a_sel, assign_b_sel

    frames_full = None
    if video_path is not None and os.path.exists(video_path):
        try:
            frames_full = load_video(video_path)  # RGB
        except Exception:
            frames_full = None

    if frames_full is not None and video_range is not None:
        start, end = video_range
        frames_full = frames_full[start:end]

    H_from_video = W_from_video = None
    if frames_full:
        H_from_video, W_from_video = frames_full[0].shape[:2]

    if img_size is not None:
        H, W = img_size
    elif H_from_video is not None and W_from_video is not None:
        H, W = H_from_video, W_from_video
    else:
        H = max(cell_px * Hc, 1)
        W = max(cell_px * Wc, 1)
    cell_px = max(1, min(H // max(Hc, 1), W // max(Wc, 1)))

    raw_video = frames_full[:] if frames_full else None
    if raw_video is not None and clip_video is not None:
        raw_video = raw_video[:clip_video]
    if raw_video is None or len(raw_video) == 0:
        raw_video = [np.zeros((H, W, 3), dtype=np.uint8) for _ in range(max(T, 1))]

    def _render_single(
        assign_n_src,
        assign_h_src,
        assign_w_src,
        assign_angles_src,
        assign_bias,
        path,
        desc,
    ):
        if assign_n_src is None:
            return
        if assign_angles_src is None:
            assign_angles_src = np.zeros(assign_n_src.shape, dtype=np.float32)
        local_pic_dir = os.path.join(os.path.dirname(path), "frames")
        os.makedirs(local_pic_dir, exist_ok=True)
        render_base = np.zeros_like(assign_n_src)
        # angles_base[angles_valid] = assign_n_src[angles_valid]
        for i, ans in enumerate(assign_n_src):
            valid = ans >= 0
            render_base[i][valid] = i - ans[valid]
            pass
        norm = render_base.astype(np.uint8)

        video_frames = []
        for t in tqdm(range(T), total=T, desc=desc):
            img_small = norm[t]
            color = cv2.applyColorMap(img_small, cv2.COLORMAP_JET)
            assign_bias_t = assign_bias[t, :, :, :]
            n0_t = assign_n_src[t, :, :, 0]
            ah = assign_h_src[t, :, :, 0]
            aw = assign_w_src[t, :, :, 0]
            invalid_mask = (n0_t < 0) | (n0_t >= len(raw_video))
            valid_indices = n0_t[~invalid_mask]
            if valid_indices.size > 0:
                unique_values, counts = np.unique(valid_indices, return_counts=True)
                idx_sorted = np.argsort(counts)
                selected_indice = unique_values[idx_sorted[::-1]][:9]
            else:
                selected_indice = np.array([], dtype=np.int32)
            # if invalid_masks[t].any():
            color[invalid_mask[..., None].repeat(3, -1)] = 0
            background = np.zeros((H * 3, W * 6, 3), dtype=np.uint8)
            frame = cv2.resize(color, (W, H), interpolation=cv2.INTER_NEAREST)

            si_start_coord = {}
            si_color = {}
            # background_alpha = np.zeros((H, W, 3), dtype=np.uint8)

            # for h_ in range(n0_t.shape[0]):
            #     for w_ in range(n0_t.shape[1]):
            #         if invalid_mask[h_, w_]:
            #             continue
            #         n0i = int(n0_t[h_, w_])
            #         if n0i < 0 or n0i >= len(raw_video):
            #             continue
            #         hh = int(ah[h_, w_])
            #         ww = int(aw[h_, w_])
            #         hb = assign_bias_t[h_, w_, 0] / 200.0
            #         wb = assign_bias_t[h_, w_, 1] / 200.0
            #         ovrh_ = hb + h_ + 0.5
            #         ovrw_ = wb + w_ + 0.5
            #         background_alpha[
            #             np.clip(
            #                 round(ovrh_ * cell_px),
            #                 0,
            #                 background_alpha.shape[0] - cell_px - 1,
            #             ) : np.clip(
            #                 round(ovrh_ * cell_px),
            #                 0,
            #                 background_alpha.shape[0] - cell_px - 1,
            #             )
            #             + cell_px,
            #             np.clip(
            #                 round(ovrw_ * cell_px),
            #                 0,
            #                 background_alpha.shape[1] - cell_px - 1,
            #             ) : np.clip(
            #                 round(ovrw_ * cell_px),
            #                 0,
            #                 background_alpha.shape[1] - cell_px - 1,
            #             )
            #             + cell_px,
            #         ] = raw_video[n0i][
            #             hh * cell_px : (hh + 1) * cell_px,
            #             ww * cell_px : (ww + 1) * cell_px,
            #         ]
            background_alpha_1 = np.zeros((H, W, 3), dtype=np.uint8)
            for h_ in range(n0_t.shape[0]):
                for w_ in range(n0_t.shape[1]):
                    if invalid_mask[h_, w_]:
                        continue
                    n0i = int(n0_t[h_, w_])
                    if n0i < 0 or n0i >= len(raw_video):
                        continue
                    hh = int(ah[h_, w_])
                    ww = int(aw[h_, w_])
                    hb = assign_bias_t[h_, w_, 0] / 200.0
                    wb = assign_bias_t[h_, w_, 1] / 200.0
                    ovrh_ = h_
                    ovrw_ = w_
                    background_alpha_1[
                        np.clip(
                            round(ovrh_ * cell_px),
                            0,
                            background_alpha_1.shape[0] - cell_px - 1,
                        ) : np.clip(
                            round(ovrh_ * cell_px),
                            0,
                            background_alpha_1.shape[0] - cell_px - 1,
                        )
                        + cell_px,
                        np.clip(
                            round(ovrw_ * cell_px),
                            0,
                            background_alpha_1.shape[1] - cell_px - 1,
                        ) : np.clip(
                            round(ovrw_ * cell_px),
                            0,
                            background_alpha_1.shape[1] - cell_px - 1,
                        )
                        + cell_px,
                    ] = raw_video[n0i][
                        hh * cell_px : (hh + 1) * cell_px,
                        ww * cell_px : (ww + 1) * cell_px,
                    ]
            # if overlay_text:
            #     for hh in range(Hc):
            #         y = int((hh + 0.5) * cell_px)
            #         for ww in range(Wc):
            #             if invalid_mask[hh, ww]:
            #                 continue
            #             x = int((ww + 0.5) * cell_px)
            #             val = int(norm[t, hh, ww])
            #             txt = str(val)
            #             cv2.putText(
            #                 frame,
            #                 txt,
            #                 (x - 8, y + 5),
            #                 cv2.FONT_HERSHEY_SIMPLEX,
            #                 0.35,
            #                 (0, 0, 0),
            #                 2,
            #                 cv2.LINE_AA,
            #             )
            #             cv2.putText(
            #                 frame,
            #                 txt,
            #                 (x - 8, y + 5),
            #                 cv2.FONT_HERSHEY_SIMPLEX,
            #                 0.35,
            #                 (255, 255, 255),
            #                 1,
            #                 cv2.LINE_AA,
            #             )

            frame_rgb = frame[:, :, ::-1]

            raw_frame = cv2.resize(raw_video[t % len(raw_video)], (W, H))
            frame_rgb = cv2.addWeighted(frame_rgb, 0.7, raw_frame, 0.3, 0)
            frame_rgb = np.concatenate(
                [raw_frame, frame_rgb, background_alpha_1], axis=0
            )

            # frame_rgb = draw_grid(frame_rgb, cell_px * 4)
            frame_rgb = cv2.putText(
                frame_rgb,
                f"frame {t}",
                (10, frame_rgb.shape[0] // 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            h_frame, w_frame = frame_rgb.shape[:2]
            video_frames.append(
                cv2.resize(
                    frame_rgb,
                    (w_frame // 2, h_frame // 2),
                    interpolation=cv2.INTER_NEAREST,
                )
            )
        if video_frames:
            save_video(path, video_frames, fps=fps)

    if grouped_info_dict:
        for offset in [40]:
            assign_n_sel, assign_h_sel, assign_w_sel, assign_a_sel, assign_bias = (
                _prepare_assignments_from_group(offset, group_selection_mode)
            )
            out_path_sel = _build_out_path(out_path, offset)
            desc = f"write-diff-video@Δ{offset}"
            _render_single(
                assign_n_sel,
                assign_h_sel,
                assign_w_sel,
                assign_a_sel,
                assign_bias,
                out_path_sel,
                desc,
            )
    # else:
    #     _render_single(
    #         assign_n, assign_hs, assign_ws, assign_angles, out_path, "write-diff-video"
    #     )


# import OpenEXR, Imath
from pathlib import Path
import zipfile, tempfile


def _sharpen_depths_with_guided_filter(
    depths: np.ndarray,
    radius: int = 8,
    eps: float = 1e-3,
    amount: float = 0.6,
) -> np.ndarray:
    """Edge-sharpen depth maps via guided filtering (unsharp mask style)."""
    ximgproc = getattr(cv2, "ximgproc", None)
    guided = getattr(ximgproc, "guidedFilter", None) if ximgproc is not None else None
    if guided is None or depths.ndim < 2:
        return depths

    is_single_map = depths.ndim == 2
    depth_batch = depths[None, ...] if is_single_map else depths
    for idx in tqdm(range(depth_batch.shape[0]), desc="sharpen-depths"):
        depth_map = depth_batch[idx]
        if depth_map.size == 0:
            continue
        guide = depth_map.astype(np.float32, copy=False)
        if not np.isfinite(guide).any():
            continue
        smoothed = guided(guide, guide, radius, eps)
        sharpened = guide + amount * (guide - smoothed)
        depth_batch[idx] = np.maximum(sharpened, 0.0)

    return depth_batch[0] if is_single_map else depth_batch


# ========================= Batch processing (uses GPU-capable check_) ========================= #
def load_depth_zip_to_array(zip_path: str | Path) -> np.ndarray:
    """
    读取由：每帧 EXR (HALF) 存 Z 通道 的 zip，返回 float32 的 (T,H,W) 数组（米）。
    会按文件名数字顺序（00005.exr -> 5）排序。
    """
    zip_path = Path(zip_path)
    frames: list[np.ndarray] = []

    with zipfile.ZipFile(zip_path, "r") as z:
        names = sorted([n for n in z.namelist() if n.lower().endswith(".exr")])

        H = W = None
        for name in names:
            # 只接受纯数字文件名（去掉扩展名）
            stem = Path(name).stem
            try:
                int(stem)
            except ValueError:
                continue

            # 读二进制写到临时文件，让 OpenEXR 读取
            with (
                z.open(name, "r") as f,
                tempfile.NamedTemporaryFile(suffix=".exr") as tmp,
            ):
                tmp.write(f.read())
                tmp.flush()
                exr = OpenEXR.InputFile(tmp.name)

                # 读尺寸（EXR 用 dataWindow）
                dw = exr.header()["dataWindow"]
                width = dw.max.x - dw.min.x + 1
                height = dw.max.y - dw.min.y + 1

                if H is None:
                    H, W = height, width
                else:
                    assert (H, W) == (height, width), "所有帧的尺寸必须一致"

                half = Imath.PixelType(Imath.PixelType.HALF)
                z_bytes = exr.channel("Z", half)
                exr.close()

                depth = (
                    np.frombuffer(z_bytes, dtype=np.float16)
                    .astype(np.float32)
                    .reshape(H, W)
                )
                frames.append(depth)
    return np.stack(frames, axis=0)


def decode_rgb_to_depth_bitpack(img_bgr):
    """解码"""
    img_val = img_bgr.astype(np.uint32)
    b, g, r = img_val[..., 0], img_val[..., 1], img_val[..., 2]
    # 组合回整数
    depth_mm = (r << 16) | (g << 8) | b
    # 转回米
    depth_m = depth_mm.astype(np.float32) / 1000.0
    return depth_m


def crop_resize_depth_if_required(depth, pathify_size):
    height, width = depth.shape[1], depth.shape[2]
    new_height = width / pathify_size[1] * pathify_size[0]
    new_height = int((new_height // 2) * 2)
    if abs(new_height - height) < 4:
        return depth
    else:
        top = int((height - new_height) / 2)
        bottom = int(top + new_height)
        depth = depth[:, top:bottom, :]
    return depth


def process_single_video(
    name,
    video_base_path,
    saving_base_path,
    cam_dir,
    depths_path,
    megasam_path,
    depth_scale,
    pathify_size,
    exclude_window,
    topk_per_query,
    is_c2w,
    axis_order,
    trans_scale,
    flip_up_sign,
    img_size,
    point_stride,
    overwrite,
    verbose,
    verbose_prob,
    write_related,
    use_fov_prefilter,
    fov_prefilter_min_distance,
    fov_prefilter_max_distance,
    fov_prefilter_always_include_recent,
    fov_prefilter_max_final,
    fov_prefilter_seed,
    fov_prefilter_axis_order,
    video_range=None,
    ema_alpha=None,
    ema_passes=None,
    ema_bidirectional=None,
    build_spatial_index=None,
    spatial_grid_cell=None,
    index_max_distance=None,
    overlay_text=True,
    cell_px=18,
    video_fps=24,
    # strategy
    min_support_per_cell=1,
    occlusion_margin=0.01,
    occ_block_range=[0.9, 1.1],
    depth_inf_thresh=100,
    angle_thresh_deg=10.0,
    group_store_topk=None,
    group_store_score_mode="angle_diff",
    clip_length=1e9,
    # viz
    viz_sparse=False,
    viz_mode="z",
    viz_contour_bands=20,
    viz_stride=1,
    viz_max_points=200000,
    viz_point_size=1.0,
    viz_alpha=0.8,
    viz_elev=20,
    viz_azim=-60,
    viz_figsize=(9, 7),
    viz_show_cameras=True,
    viz_camera_stride=1,
    viz_camera_size=10.0,
    viz_camera_linewidth=2.0,
    viz_camera_alpha=0.95,
    viz_camera_cmap="viridis",
    viz_camera_colorbar=True,
    viz_camera_annotate=True,
    viz_per_t=False,
    viz_per_t_stride=1,
    viz_per_t_all_stride=1,
    viz_per_t_all_max_points=120000,
    viz_per_t_all_point_size=2,
    viz_per_t_all_alpha=0.2,
    viz_per_t_point_size=18,
    viz_per_t_alpha=0.98,
    viz_per_t_edge=True,
    viz_per_t_edge_color="k",
    viz_per_t_ray_len=0.35,
    viz_per_t_elev=20,
    viz_per_t_azim=-60,
    viz_per_t_figsize=(10, 8),
    viz_per_t_show_target_camera=True,
    viz_per_t_frustum_scale=0.15,
    # GPU
    use_gpu=True,
):
    video_path = os.path.join(video_base_path, name + ".mp4")
    assert os.path.exists(video_path), f"视频文件不存在: {video_path}"
    out_npz = os.path.join(
        saving_base_path,
        f"{name}_frustum.npz",
    )
    diff_mp4 = os.path.join(
        saving_base_path, f"{name}_diff_{pathify_size[0]}x{pathify_size[1]}.mp4"
    )

    def _meta_value(val, default=None):
        if val is None:
            return default
        if isinstance(val, np.ndarray):
            if val.size == 1:
                return val.item()
            return val
        return val

    if megasam_path is None:
        intrinsic_path = os.path.join(cam_dir, name + "_intrinsics_da3nested.npy")
        extrinsic_path = os.path.join(cam_dir, name + "_extrinsics_da3nested.npy")
        if not os.path.exists(intrinsic_path):
            exit()
        # dep_path = os.path.join(depths_path, name + ".zip")
        viz_png = os.path.join(
            saving_base_path, f"{name}_sparse3d_{pathify_size[0]}x{pathify_size[1]}.png"
        )
        intrinsic = np.load(intrinsic_path)[0]
        # intrinsic = np.array(
        #     [
        #         [fxfycxcy[0], 0, fxfycxcy[2]],
        #         [0, fxfycxcy[1], fxfycxcy[3]],
        #         [0, 0, 1],
        #     ],
        #     dtype=np.float32,
        # )
        extrinsic = np.load(extrinsic_path)
        # depths = _load_depth_npz_auto(dep_path).astype(np.float32) * float(depth_scale)
        if os.path.exists(os.path.join(depths_path, name + "_depth_da3nested.npz")):
            raw_depths = np.load(
                os.path.join(depths_path, name + "_depth_da3nested.npz")
            )["arr_0"]
        elif os.path.exists(os.path.join(depths_path, name + "_depth_da3nested.npy")):
            raw_depths = np.load(
                os.path.join(depths_path, name + "_depth_da3nested.npy")
            )
        # 根据存储格式选择是否解码
        if raw_depths.dtype == np.uint8:
            raw_depths = decode_rgb_to_depth_bitpack(raw_depths).astype(np.float32)
        elif raw_depths.dtype != np.float32:
            raw_depths = raw_depths.astype(np.float32)
        # depths = _sharpen_depths_with_guided_filter(raw_depths)
        # raw_depths = crop_resize_depth_if_required(raw_depths, pathify_size)
        depths = raw_depths

    else:
        data = np.load(megasam_path)
        intrinsic = data["intrinsic"]
        extrinsic = data["cam_c2w"]
        depths = (data["depths"] * float(depth_scale)).astype(np.float32)
        depths = _sharpen_depths_with_guided_filter(depths)
    if os.path.exists(out_npz) and not overwrite:
        meta = {
            "Hc": pathify_size[0],
            "Wc": pathify_size[1],
            "topk": 1,
            "near_back": 1,
            "far_back": 600,
            "H_img": 448,
            "W_img": 832,
            "point_stride": int(point_stride),
            # "trans_axis_order": str(trans_axis_order),
            "trans_scale": float(trans_scale),
            # "flip_up_sign": bool(flip_up_sign),
            "is_c2w": bool(is_c2w),
            # "min_support_per_cell": int(min_support_per_cell),
            "depth_inf_thresh": (
                None if depth_inf_thresh is None else float(depth_inf_thresh)
            ),
            # "grouped_info_dict": ,
            "group_selection_mode": str(group_store_score_mode),
            "group_store_topk": (
                None if group_store_topk is None else int(group_store_topk)
            ),
            # "CAM_result": CAM_result,
        }
        grouped_info_dict = np.load(out_npz, allow_pickle=True)
        assign_n = None
        print(f"⚠️  存在 {name}, output exists at {out_npz}")
    else:
        (
            assign_n,
            assign_hs,
            assign_ws,
            assign_angles,
            grouped_info_dict,
            CAM_result,
            meta,
        ) = check_depth_overlap_sequence(
            depths=depths if clip_length is None else depths[:clip_length],
            extrinsics=(extrinsic if clip_length is None else extrinsic[:clip_length]),
            intrinsic=intrinsic if clip_length is None else intrinsic[:clip_length],
            pathify_size=pathify_size,
            is_c2w=is_c2w,
            trans_axis_order=axis_order,
            trans_scale=trans_scale,
            flip_up_sign=flip_up_sign,
            img_size=None if img_size is None else tuple(img_size),
            point_stride=point_stride,
            exclude_window=exclude_window,
            topk_per_query=topk_per_query,
            min_support_per_cell=min_support_per_cell,
            depth_inf_thresh=depth_inf_thresh,
            angle_thresh_deg=angle_thresh_deg,
            group_store_topk=group_store_topk,
            group_store_score_mode=group_store_score_mode,
            occlusion_margin=occlusion_margin,
            occ_block_range=occ_block_range,
            clip_length=clip_length,
            viz_sparse=viz_sparse,
            viz_mode=viz_mode,
            viz_contour_bands=viz_contour_bands,
            viz_stride=viz_stride,
            viz_max_points=viz_max_points,
            viz_point_size=viz_point_size,
            viz_alpha=viz_alpha,
            viz_elev=viz_elev,
            viz_azim=viz_azim,
            viz_figsize=viz_figsize,
            viz_save_path=viz_png if viz_sparse else None,
            viz_show_cameras=viz_show_cameras,
            viz_camera_stride=viz_camera_stride,
            viz_camera_size=viz_camera_size,
            viz_camera_linewidth=viz_camera_linewidth,
            viz_camera_alpha=viz_camera_alpha,
            viz_camera_cmap=viz_camera_cmap,
            viz_camera_colorbar=viz_camera_colorbar,
            viz_camera_annotate=viz_camera_annotate,
            viz_per_t=viz_per_t,
            viz_per_t_stride=viz_per_t_stride,
            viz_per_t_all_stride=viz_per_t_all_stride,
            viz_per_t_all_max_points=viz_per_t_all_max_points,
            viz_per_t_all_point_size=viz_per_t_all_point_size,
            viz_per_t_all_alpha=viz_per_t_all_alpha,
            viz_per_t_point_size=viz_per_t_point_size,
            viz_per_t_alpha=viz_per_t_alpha,
            viz_per_t_edge=viz_per_t_edge,
            viz_per_t_edge_color=viz_per_t_edge_color,
            viz_per_t_ray_len=viz_per_t_ray_len,
            viz_per_t_elev=viz_per_t_elev,
            viz_per_t_azim=viz_per_t_azim,
            viz_per_t_figsize=viz_per_t_figsize,
            viz_per_t_show_target_camera=viz_per_t_show_target_camera,
            viz_per_t_frustum_scale=viz_per_t_frustum_scale,
            use_gpu=use_gpu,
            use_fov_prefilter=use_fov_prefilter,
            fov_prefilter_min_distance=fov_prefilter_min_distance,
            fov_prefilter_max_distance=fov_prefilter_max_distance,
            fov_prefilter_always_include_recent=fov_prefilter_always_include_recent,
            fov_prefilter_max_final=fov_prefilter_max_final,
            fov_prefilter_seed=fov_prefilter_seed,
            fov_prefilter_axis_order=fov_prefilter_axis_order,
        )
        os.makedirs(saving_base_path, exist_ok=True)
        np.savez_compressed(
            out_npz,
            **grouped_info_dict,
        )
        # with open(out_npz.replace(".npz", ".json"), "w") as jf:
        #     json.dump(grouped_info_dict, jf, indent=4)
        # grouped_info_dict = meta.get("grouped_info_dict")

    group_selection_mode_meta = _meta_value(
        meta.get("group_selection_mode"), group_store_score_mode
    )
    if group_selection_mode_meta is None:
        group_selection_mode_meta = "angle_diff"
    else:
        group_selection_mode_meta = str(group_selection_mode_meta)
    grouped_info_dict = grouped_info_dict or {}
    save_depths = depths.copy()[:, ::point_stride, ::point_stride]
    save_depths_dict = {}
    for ide, dep in enumerate(save_depths):
        save_depths_dict[f"depth_{ide:05d}"] = dep.astype(np.float16)
    np.savez_compressed(
        out_npz.replace("frustum.npz", "depth.npz"),
        **save_depths_dict,
    )
    save_depths_verbose = np.clip(save_depths, 0, 255).astype(np.uint8)
    vwrite(out_npz.replace("frustum.npz", "depth.mp4"), save_depths_verbose)
    if verbose and random.uniform(0, 1) <= verbose_prob:
        _write_grouped_diff_video(
            video_path,
            diff_mp4,
            assign_n,
            None,
            None,
            None,
            near_back=meta["near_back"],
            far_back=meta["far_back"],
            video_range=video_range,
            is_megasam=megasam_path is not None,
            cell_px=cell_px,
            fps=video_fps,
            overlay_text=overlay_text,
            clip_video=clip_length,
            img_size=img_size,
            write_related=write_related,
            grouped_info_dict=grouped_info_dict,
            group_selection_mode=group_selection_mode_meta,
        )
    return f"✅ {name} -> saved {os.path.basename(out_npz)}"


def batch_process_overlap_multiprocessing(
    video_path,
    saving_path,
    cam_dir,
    depths_path,
    megasam_path,
    use_fov_prefilter,
    fov_prefilter_min_distance,
    fov_prefilter_max_distance,
    fov_prefilter_always_include_recent,
    fov_prefilter_max_final,
    fov_prefilter_seed,
    fov_prefilter_axis_order,
    assign_name=None,
    num_processes=8,
    pathify_size=(12, 16),
    exclude_window=(20, 50),
    topk_per_query=1,
    is_c2w=False,
    axis_order="xyz",
    trans_scale=1.0,
    flip_up_sign=False,
    img_size=None,
    point_stride=6,
    overwrite=False,
    write_related=False,
    verbose=False,
    verbose_prob=0.05,
    overlay_text=True,
    cell_px=18,
    video_fps=24,
    video_range=None,
    min_distance=None,
    max_distance=None,
    depth_scale=1.0,
    occ_block_range=(5, 95),
    ema_alpha=0.0,
    ema_passes=0,
    ema_bidirectional=False,
    build_spatial_index=False,
    spatial_grid_cell=None,
    index_max_distance=None,
    min_support_per_cell=1,
    occlusion_margin=0.01,
    depth_inf_thresh=100,
    angle_thresh_deg=10.0,
    clip_length=None,
    viz_sparse=False,
    viz_mode="z",
    viz_contour_bands=20,
    viz_stride=1,
    viz_max_points=200000,
    viz_point_size=1.0,
    viz_alpha=0.8,
    viz_elev=20,
    viz_azim=-60,
    viz_figsize=(9, 7),
    viz_show_cameras=True,
    viz_camera_stride=1,
    viz_camera_size=10.0,
    viz_camera_linewidth=2.0,
    viz_camera_alpha=0.95,
    viz_camera_cmap="viridis",
    viz_camera_colorbar=True,
    viz_camera_annotate=True,
    viz_per_t=False,
    viz_per_t_stride=1,
    viz_per_t_all_stride=1,
    viz_per_t_all_max_points=120000,
    viz_per_t_all_point_size=2,
    viz_per_t_all_alpha=0.2,
    viz_per_t_point_size=18,
    viz_per_t_alpha=0.98,
    viz_per_t_edge=True,
    viz_per_t_edge_color="k",
    viz_per_t_ray_len=0.35,
    viz_per_t_elev=20,
    viz_per_t_azim=-60,
    viz_per_t_figsize=(10, 8),
    viz_per_t_show_target_camera=True,
    viz_per_t_frustum_scale=0.15,
    use_gpu=True,
    group_store_score_mode="time_diff",
    group_store_topk=None,
):
    os.makedirs(saving_path, exist_ok=True)
    if assign_name is not None:
        # npz_paths = [assign_name]
        if "." in assign_name:
            assign_name = assign_name.split(".")[0]
        npz_paths = [assign_name]
    # single-process for simplicity (multiprocessing omitted for brevity)
    for name in tqdm(npz_paths, desc="single-process"):
        _ = process_single_video(
            name,
            video_base_path=video_path,
            saving_base_path=saving_path,
            cam_dir=cam_dir,
            depths_path=depths_path,
            megasam_path=megasam_path,
            depth_scale=depth_scale,
            pathify_size=pathify_size,
            exclude_window=exclude_window,
            topk_per_query=topk_per_query,
            is_c2w=is_c2w,
            axis_order=axis_order,
            trans_scale=trans_scale,
            flip_up_sign=flip_up_sign,
            img_size=img_size,
            video_range=video_range,
            point_stride=point_stride,
            overwrite=overwrite,
            write_related=write_related,
            verbose=verbose,
            verbose_prob=verbose_prob,
            ema_alpha=ema_alpha,
            ema_passes=ema_passes,
            ema_bidirectional=ema_bidirectional,
            build_spatial_index=build_spatial_index,
            spatial_grid_cell=spatial_grid_cell,
            index_max_distance=index_max_distance,
            overlay_text=overlay_text,
            cell_px=cell_px,
            video_fps=video_fps,
            occ_block_range=occ_block_range,
            min_support_per_cell=min_support_per_cell,
            occlusion_margin=occlusion_margin,
            depth_inf_thresh=depth_inf_thresh,
            angle_thresh_deg=angle_thresh_deg,
            clip_length=clip_length,
            viz_sparse=viz_sparse,
            viz_mode=viz_mode,
            viz_contour_bands=viz_contour_bands,
            viz_stride=viz_stride,
            viz_max_points=viz_max_points,
            viz_point_size=viz_point_size,
            viz_alpha=viz_alpha,
            viz_elev=viz_elev,
            viz_azim=viz_azim,
            viz_figsize=viz_figsize,
            viz_show_cameras=viz_show_cameras,
            viz_camera_stride=viz_camera_stride,
            viz_camera_size=viz_camera_size,
            viz_camera_linewidth=viz_camera_linewidth,
            viz_camera_alpha=viz_camera_alpha,
            viz_camera_cmap=viz_camera_cmap,
            viz_camera_colorbar=viz_camera_colorbar,
            viz_camera_annotate=viz_camera_annotate,
            viz_per_t=viz_per_t,
            viz_per_t_stride=viz_per_t_stride,
            viz_per_t_all_stride=viz_per_t_all_stride,
            viz_per_t_all_max_points=viz_per_t_all_max_points,
            viz_per_t_all_point_size=viz_per_t_all_point_size,
            viz_per_t_all_alpha=viz_per_t_all_alpha,
            viz_per_t_point_size=viz_per_t_point_size,
            viz_per_t_alpha=viz_per_t_alpha,
            viz_per_t_edge=viz_per_t_edge,
            viz_per_t_edge_color=viz_per_t_edge_color,
            viz_per_t_ray_len=viz_per_t_ray_len,
            viz_per_t_elev=viz_per_t_elev,
            viz_per_t_azim=viz_per_t_azim,
            viz_per_t_figsize=viz_per_t_figsize,
            viz_per_t_show_target_camera=viz_per_t_show_target_camera,
            viz_per_t_frustum_scale=viz_per_t_frustum_scale,
            use_gpu=use_gpu,
            group_store_score_mode=group_store_score_mode,
            group_store_topk=group_store_topk,
            use_fov_prefilter=use_fov_prefilter,
            fov_prefilter_min_distance=fov_prefilter_min_distance,
            fov_prefilter_max_distance=fov_prefilter_max_distance,
            fov_prefilter_always_include_recent=fov_prefilter_always_include_recent,
            fov_prefilter_max_final=fov_prefilter_max_final,
            fov_prefilter_seed=fov_prefilter_seed,
            fov_prefilter_axis_order=fov_prefilter_axis_order,
        )


def draw_grid(
    image: np.ndarray,
    grid_interval: int,
    line_width: int = 1,
    color=(0, 255, 0),
    copy: bool = True,
) -> np.ndarray:
    """
    在RGB图像上画规则网格线（等间距）。

    参数：
        image: H x W x 3 的 uint8 图像（RGB）
        grid_interval: 网格间隔（像素），即多少像素画一条线
        line_width: 线宽（像素）
        color: 线的颜色，RGB 元组，如 (0,255,0)
        copy: True 时在拷贝上画，不改原图；False 时原地修改

    返回：
        画好网格线的图像（np.ndarray）
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image 必须是 H x W x 3 的 RGB 图像")
    if image.dtype != np.uint8:
        raise ValueError("image 的 dtype 必须是 uint8")
    if grid_interval <= 0:
        raise ValueError("grid_interval 必须为正整数")
    if line_width <= 0:
        raise ValueError("line_width 必须为正整数")

    img = image.copy() if copy else image
    H, W, _ = img.shape

    # 垂直线：每隔 grid_interval 像素画一条
    for ix, x in enumerate(range(0, W, grid_interval)):
        x_start = x
        x_end = min(x + line_width, W)  # 防止越界
        img[:, x_start:x_end, :] = [
            round((ix / W) * 255),
            0,
            255 - round((ix / W) * 255),
        ]

    # 水平线：每隔 grid_interval 像素画一条
    for iy, y in enumerate(range(0, H, grid_interval)):
        y_start = y
        y_end = min(y + line_width, H)
        img[y_start:y_end, :, :] = [
            255 - round((iy / H) * 255),
            0,
            round((iy / H) * 255),
        ]

    return img


if __name__ == "__main__":
    # Example: adjust paths as needed
    from argparse import ArgumentParser
    import time

    t = time.strftime("%Y-%m-%d-%H:%M", time.localtime())
    safe_string_t = t.replace(" ", "_").replace(":", "-")
    parser = ArgumentParser()
    parser.add_argument("--assign_name", type=str, default=None)
    parser.add_argument("--cam_dir", type=str, default="out_test_da3")
    parser.add_argument("--depth_dir", type=str, default="out_test_da3")
    parser.add_argument(
        "-o", "--out_dir", type=str, default=f"2077-11-25_da3_frustum_{safe_string_t}"
    )
    parser.add_argument("--video_dir", type=str, default="raw_2077-11-25_576p")
    parser.add_argument("-c", "--clip_num", type=int, default=1000000)
    parser.add_argument("--verbose_prob", type=float, default=1)
    parser.add_argument(
        "-or", "--overwrite", action="store_true", help="是否覆盖已存在的输出文件"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="是否覆盖已存在的输出文件"
    )
    parser.add_argument(
        "-ps", "--point_stride", type=int, default=8, help="是否覆盖已存在的输出文件"
    )
    parser.add_argument(
        "-ph",
        "--patchify_height",
        type=int,
        default=36,
        help="是否覆盖已存在的输出文件",
    )
    parser.add_argument(
        "-pw",
        "--patchify_weight",
        type=int,
        default=64,
        help="是否覆盖已存在的输出文件",
    )
    parser.add_argument(
        "-hrz", "--horizon", type=int, default=1000, help="是否覆盖已存在的输出文件"
    )
    args = parser.parse_args()
    cam_dir = args.cam_dir
    depth_dir = args.depth_dir
    out_dir = args.out_dir
    video_dir = args.video_dir
    if os.path.exists(cam_dir) and os.path.exists(depth_dir):
        batch_process_overlap_multiprocessing(
            video_path=video_dir,
            saving_path=out_dir,
            cam_dir=depth_dir,
            depths_path=depth_dir,
            assign_name=args.assign_name,
            megasam_path=None,
            num_processes=1,
            pathify_size=(args.patchify_height, args.patchify_weight),
            exclude_window=(1, args.horizon),
            topk_per_query=1,
            is_c2w=False,
            axis_order="xyz",
            trans_scale=1,
            depth_scale=1,
            flip_up_sign=False,
            img_size=None,
            video_range=(0, args.clip_num),
            point_stride=3,
            overwrite=args.overwrite,
            write_related=False,
            verbose=args.verbose,
            verbose_prob=args.verbose_prob,
            overlay_text=True,
            cell_px=3,
            video_fps=24,
            clip_length=args.clip_num,
            occ_block_range=[0.9, 1.1],
            min_support_per_cell=1,
            occlusion_margin=0.01,
            depth_inf_thresh=1e9,
            angle_thresh_deg=1e9,
            viz_sparse=True,
            viz_per_t=True,
            group_store_score_mode="time_diff",
            group_store_topk=1,
            use_fov_prefilter=False,
            fov_prefilter_min_distance=0.5,
            fov_prefilter_max_distance=100,
            fov_prefilter_always_include_recent=90,
            fov_prefilter_max_final=120,
            fov_prefilter_seed=42,
            fov_prefilter_axis_order="xyz",
        )
