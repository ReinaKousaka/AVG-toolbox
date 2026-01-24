#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import shutil
import numpy as np

# ✅ headless: 强制使用Agg后端（必须在import pyplot前）
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def invert_w2c_to_c2w(T_wc: np.ndarray) -> np.ndarray:
    """T_wc: (N,4,4) world->camera -> T_cw: (N,4,4) camera->world"""
    R = T_wc[:, :3, :3]
    t = T_wc[:, :3, 3:4]
    Rt = np.transpose(R, (0, 2, 1))
    T_cw = np.zeros_like(T_wc)
    T_cw[:, :3, :3] = Rt
    T_cw[:, :3, 3:4] = -Rt @ t
    T_cw[:, 3, 3] = 1.0
    return T_cw


def rotation_angle_deg(R: np.ndarray) -> float:
    tr = np.trace(R)
    cos_theta = (tr - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def load_extrinsics(path: Path, key: str | None) -> np.ndarray:
    suf = path.suffix.lower()
    if suf == ".npy":
        T = np.load(path)
    elif suf == ".npz":
        z = np.load(path)
        if key is None:
            for c in ["extrinsics", "poses", "w2c", "T_wc", "data"]:
                if c in z:
                    key = c
                    break
        if key is None or key not in z:
            raise KeyError(f"npz里没找到key='{key}'。可用keys={list(z.keys())}")
        T = z[key]
    else:
        raise ValueError("只支持 .npy / .npz")
    T = np.asarray(T)
    if T.ndim != 3 or T.shape[1:] != (4, 4):
        raise ValueError(f"外参应为(N,4,4)，但得到 {T.shape}")
    return T


def parse_region(region: str | None):
    """
    region:
      - None -> auto
      - "auto"
      - "minx,maxx,miny,maxy,minz,maxz" (6 floats)
    """
    if region is None or region.strip().lower() == "auto":
        return None
    parts = [p.strip() for p in region.split(",")]
    if len(parts) != 6:
        raise ValueError("region格式应为 'minx,maxx,miny,maxy,minz,maxz' 或 'auto'")
    vals = list(map(float, parts))
    return (vals[0], vals[1], vals[2], vals[3], vals[4], vals[5])


def ensure_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "没找到ffmpeg，无法输出mp4。\n"
            "Ubuntu可用: sudo apt-get update && sudo apt-get install -y ffmpeg\n"
        )


def main():
    ap = argparse.ArgumentParser("Headless animate camera motion (save mp4)")
    ap.add_argument(
        "--input", "-i", type=str, required=True, help="w2c外参路径 (.npy/.npz)"
    )
    ap.add_argument("--key", "-k", type=str, default=None, help="npz键名")

    ap.add_argument(
        "--out", "-o", type=str, required=True, help="输出mp4路径，例如 out.mp4"
    )
    ap.add_argument("--fps", "-f", type=int, default=30, help="fps")
    ap.add_argument("--dpi", "-d", type=int, default=120, help="dpi")
    ap.add_argument(
        "--bitrate", "-br", type=int, default=6000, help="mp4 bitrate(kbps)"
    )

    ap.add_argument("--stride", "-s", type=int, default=1, help="下采样步长")
    ap.add_argument(
        "--max_frames", "-mf", type=int, default=-1, help="最多可视化多少帧(-1=全部)"
    )
    ap.add_argument(
        "--tail", "-t", type=int, default=60, help="尾巴长度(帧)，0=不画尾巴"
    )

    ap.add_argument(
        "--region",
        "-r",
        type=str,
        default="auto",
        help="限定区域：auto 或 'minx,maxx,miny,maxy,minz,maxz'",
    )
    ap.add_argument(
        "--pad",
        "-p",
        type=float,
        default=0.05,
        help="auto region时padding比例(相对bbox对角线)",
    )

    ap.add_argument(
        "--deg_threshold", "-dt", type=float, default=10.0, help="旋转跳变阈值(度)"
    )
    ap.add_argument(
        "--trans_threshold", "-tt", type=float, default=0.2, help="平移跳变阈值(单位)"
    )

    ap.add_argument("--fig_w", "-fw", type=float, default=8.0, help="画布宽(英寸)")
    ap.add_argument("--fig_h", "-fh", type=float, default=7.0, help="画布高(英寸)")

    args = ap.parse_args()
    ensure_ffmpeg()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    T_wc_full = load_extrinsics(Path(args.input), args.key)
    T_wc = T_wc_full[:: max(1, args.stride)].copy()
    if args.max_frames > 0:
        T_wc = T_wc[: args.max_frames]

    N = T_wc.shape[0]
    if N < 2:
        raise ValueError("至少2帧才能做运动动画。")

    # w2c -> c2w -> camera centers in world
    T_cw = invert_w2c_to_c2w(T_wc)
    C = T_cw[:, :3, 3]  # (N,3)
    Cx, Cy, Cz = C[:, 0], C[:, 1], C[:, 2]

    # jump detection
    dC = C[1:] - C[:-1]
    trans_step = np.linalg.norm(dC, axis=1)

    R_cw = T_cw[:, :3, :3]
    rot_step_deg = np.zeros(N - 1, dtype=np.float64)
    for i in range(N - 1):
        R_rel = R_cw[i + 1] @ R_cw[i].T
        rot_step_deg[i] = rotation_angle_deg(R_rel)

    bad_any = (trans_step > args.trans_threshold) | (rot_step_deg > args.deg_threshold)
    jump_frames = (np.where(bad_any)[0] + 1).astype(int)
    jump_set = set(jump_frames.tolist())

    # region / axis limits
    region = parse_region(args.region)
    if region is None:
        mn = C.min(axis=0)
        mx = C.max(axis=0)
        diag = float(np.linalg.norm(mx - mn) + 1e-12)
        pad = args.pad * diag
        mn = mn - pad
        mx = mx + pad
        region = (mn[0], mx[0], mn[1], mx[1], mn[2], mx[2])
    minx, maxx, miny, maxy, minz, maxz = region

    # --- plot setup ---
    fig = plt.figure(figsize=(args.fig_w, args.fig_h))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Camera center motion (world)")
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_zlim(minz, maxz)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # background full trajectory
    ax.plot(Cx, Cy, Cz, linewidth=1.0, alpha=0.25)

    # animated artists
    head_scatter = ax.scatter([Cx[0]], [Cy[0]], [Cz[0]], s=35)
    (tail_line,) = ax.plot([Cx[0]], [Cy[0]], [Cz[0]], linewidth=2.0, alpha=0.9)
    jump_scatter = ax.scatter([], [], [], marker="x", s=55)
    txt = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    def init():
        head_scatter._offsets3d = ([Cx[0]], [Cy[0]], [Cz[0]])
        tail_line.set_data([Cx[0]], [Cy[0]])
        tail_line.set_3d_properties([Cz[0]])
        jump_scatter._offsets3d = ([], [], [])
        txt.set_text("")
        return head_scatter, tail_line, jump_scatter, txt

    def update(frame_idx: int):
        # head
        head_scatter._offsets3d = ([Cx[frame_idx]], [Cy[frame_idx]], [Cz[frame_idx]])

        # tail
        if args.tail > 0:
            s = max(0, frame_idx - args.tail)
            tail_line.set_data(Cx[s : frame_idx + 1], Cy[s : frame_idx + 1])
            tail_line.set_3d_properties(Cz[s : frame_idx + 1])
        else:
            tail_line.set_data([Cx[frame_idx]], [Cy[frame_idx]])
            tail_line.set_3d_properties([Cz[frame_idx]])

        # jump marker at this frame
        if frame_idx in jump_set:
            jump_scatter._offsets3d = (
                [Cx[frame_idx]],
                [Cy[frame_idx]],
                [Cz[frame_idx]],
            )
        else:
            jump_scatter._offsets3d = ([], [], [])

        # info
        if frame_idx == 0:
            txt.set_text(f"frame {frame_idx}/{N-1}")
        else:
            txt.set_text(
                f"frame {frame_idx}/{N-1}\n"
                f"trans_step={trans_step[frame_idx-1]:.4f}  rot_step={rot_step_deg[frame_idx-1]:.2f}deg"
            )
        return head_scatter, tail_line, jump_scatter, txt

    anim = FuncAnimation(
        fig,
        update,
        frames=N,
        init_func=init,
        interval=1000 / max(1, args.fps),
        blit=False,
    )

    # save mp4 (ffmpeg writer)
    # bitrate: kbps -> str
    writer = matplotlib.animation.FFMpegWriter(
        fps=args.fps,
        bitrate=args.bitrate,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p"],
    )

    print(f"[INFO] frames={N}, fps={args.fps}, stride={args.stride}, tail={args.tail}")
    if len(jump_frames) > 0:
        print(
            f"[WARN] jump frames (in downsampled seq): {jump_frames[:50].tolist()}"
            + (" ..." if len(jump_frames) > 50 else "")
        )
    else:
        print("[INFO] no jump detected under thresholds")

    print(f"[INFO] saving mp4 -> {out_path}")
    anim.save(str(out_path), writer=writer, dpi=args.dpi)
    plt.close(fig)
    print("[OK] done.")


if __name__ == "__main__":
    main()
