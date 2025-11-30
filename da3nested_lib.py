# da3nested_lib.py
import os
import sys
import glob
from typing import List, Tuple

import numpy as np
import torch

# depth-anything-3 仓库路径按你原来的习惯来
sys.path.append(os.path.join(os.getcwd(), "depth-anything-3"))

from depth_anything_3.api import DepthAnything3  # noqa: E402
from decord import VideoReader, cpu  # noqa: E402
from PIL import Image  # noqa: E402
from skvideo.io import vwrite  # noqa: E402


# ===========================
# 深度 <-> RGB bitpack 工具
# ===========================


def encode_depth_to_rgb_bitpack(depth_map: np.ndarray) -> np.ndarray:
    """
    depth_map: float32, 单位: 米
    以毫米精度打包到 24bit BGR 图像中。
    """
    depth_mm = (depth_map * 1000.0).astype(np.uint32)

    r = (depth_mm >> 16) & 0xFF
    g = (depth_mm >> 8) & 0xFF
    b = depth_mm & 0xFF

    img_bgr = np.dstack((b.astype(np.uint8), g.astype(np.uint8), r.astype(np.uint8)))
    return img_bgr


def decode_rgb_to_depth_bitpack(img_bgr: np.ndarray) -> np.ndarray:
    """
    对称的解码函数，当前流程里主要用于检查/调试。
    """
    img_val = img_bgr.astype(np.uint32)
    b, g, r = img_val[..., 0], img_val[..., 1], img_val[..., 2]
    depth_mm = (r << 16) | (g << 8) | b
    depth_m = depth_mm.astype(np.float32) / 1000.0
    return depth_m


# ===========================
# 工具函数：多目录找 mp4
# ===========================


def find_videos(
    input_dirs: List[str], exts: Tuple[str, ...] = (".mp4", ".MP4")
) -> List[str]:
    """
    在若干目录下找所有 mp4（大小写都支持）。
    """
    videos: List[str] = []
    for d in input_dirs:
        d = os.path.abspath(d)
        for ext in exts:
            pattern = os.path.join(d, f"*{ext}")
            videos.extend(glob.glob(pattern))
    videos = sorted(set(videos))
    return videos


# ===========================
# Worker：单卡模型 + 单视频处理
# ===========================


class DA3NestedVideoWorker:
    def __init__(
        self,
        device: str,
        model_name: str = "depth-anything/DA3NESTED-GIANT-LARGE",
    ):
        """
        device: "cuda:0" / "cuda:1" / "cpu" ...
        """
        self.device = torch.device(device)
        print(f"[Init] Loading model on device: {self.device}")
        self.model = DepthAnything3.from_pretrained(model_name)
        self.model = self.model.to(device=self.device)
        self.model.eval()

    @torch.no_grad()
    def _infer_chunk(
        self,
        pil_images: List[Image.Image],
        process_res: int | None = None,
        use_ray_pose: bool = True,
    ):
        """
        对一个 chunk 的图像做一次完整推理:
        返回 depth / intrinsics / extrinsics
        - depth       : [N, H, W]
        - intrinsics  : [N, 3, 3]
        - extrinsics  : [N, 3, 4]  (w2c)
        """
        kwargs = {}
        if process_res is not None:
            kwargs["process_res"] = process_res
        kwargs["use_ray_pose"] = use_ray_pose

        prediction = self.model.inference(pil_images, **kwargs)

        depth = prediction.depth  # [N, H, W], float32, 已经是米制
        intrinsics = prediction.intrinsics  # [N, 3, 3]
        extrinsics = prediction.extrinsics  # [N, 3, 4], w2c

        return depth, intrinsics, extrinsics

    def process_video(
        self,
        video_path: str,
        output_dir: str,
        chunk_size: int = 1000,
        process_res: int | None = 504,
        pose_overlap: int = 1,
        use_ray_pose: bool = True,
    ):
        """
        对单个视频:
        - 按 chunk_size 分段推理 depth / K / w2c
        - depth / K 直接 concat
        - w2c 用 overlap 帧对齐世界坐标系, 然后 concat
        - 保存 depth / K / w2c 的 npy + bitpack 深度视频

        输出文件：
        - <name>_depth_da3nested.npy      : [N, H, W] float32, m
        - <name>_intrinsics_da3nested.npy : [N, 3, 3] float32
        - <name>_extrinsics_da3nested.npy : [N, 4, 4] float32, w2c
        - <name>_depth_da3nested.mp4      : BGR 24bit 打包的深度视频
        """
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.basename(video_path)
        stem, _ = os.path.splitext(base_name)

        depth_npy_path = os.path.join(output_dir, f"{stem}_depth_da3nested.npy")
        intr_npy_path = os.path.join(output_dir, f"{stem}_intrinsics_da3nested.npy")
        extr_npy_path = os.path.join(output_dir, f"{stem}_extrinsics_da3nested.npy")
        depth_video_path = os.path.join(output_dir, f"{stem}_depth_da3nested.mp4")

        print(f"[Video] {video_path}")
        print(f"  -> depth npy      : {depth_npy_path}")
        print(f"  -> intrinsics npy : {intr_npy_path}")
        print(f"  -> extrinsics npy : {extr_npy_path}")
        print(f"  -> depth video    : {depth_video_path}")
        print(f"  -> device         : {self.device}")

        vr = VideoReader(video_path, ctx=cpu(0))
        num_frames = len(vr)
        print(f"  Total frames: {num_frames}")

        if num_frames == 0:
            print("  [Warn] Empty video, skip.")
            return

        # 防止 overlap >= chunk_size 导致 step <= 0
        if pose_overlap >= chunk_size:
            print(
                f"  [Warn] pose_overlap={pose_overlap} >= chunk_size={chunk_size}, "
                f"强制把 pose_overlap 改成 {chunk_size - 1}"
            )
            pose_overlap = chunk_size - 1

        step = chunk_size if pose_overlap <= 0 else (chunk_size - pose_overlap)
        if step <= 0:
            step = chunk_size

        all_depth_chunks: List[np.ndarray] = []
        all_intr_chunks: List[np.ndarray] = []
        all_extr_chunks: List[np.ndarray] = []  # [*, 4, 4]
        all_bitpacked_frames: List[np.ndarray] = []

        # 用来对齐 pose：按“全局帧 index”顺序存 4x4 w2c
        global_extrinsics_4x4: List[np.ndarray] = []

        seg_idx = 0
        start = 0
        prev_end = 0

        while start < num_frames:
            end = min(start + chunk_size, num_frames)
            idxs = list(range(start, end))
            print(f"  Chunk {seg_idx}: frames [{start}, {end}) " f"(size={len(idxs)})")

            frames_np = vr.get_batch(idxs).asnumpy()  # [S, H, W, 3]
            pil_images = [Image.fromarray(f) for f in frames_np]

            depth_chunk, intr_chunk, extr_chunk = self._infer_chunk(
                pil_images,
                process_res=process_res,
                use_ray_pose=use_ray_pose,
            )
            S = depth_chunk.shape[0]
            assert S == len(idxs), "模型返回帧数和输入不一致？"

            # extrinsics: [S, 3, 4] -> [S, 4, 4]
            extr4 = np.zeros((S, 4, 4), dtype=np.float32)
            extr4[:, 3, 3] = 1.0
            extr4[:, :3, :4] = extr_chunk

            if seg_idx == 0 or pose_overlap <= 0:
                # 第一段或者不做 pose 对齐：直接用
                depth_to_use = depth_chunk
                intr_to_use = intr_chunk
                extr_to_use = extr4
            else:
                # 有 overlap：用 overlap 的第一帧把当前 chunk 的世界系
                # 对齐到“全局”世界系上
                anchor_global_idx = start  # 当前 chunk 第 0 帧在全局的 index
                anchor_local_idx = 0  # 当前 chunk 的第 0 帧

                if anchor_global_idx >= len(global_extrinsics_4x4):
                    # 理论上不会发生：没有真正 overlap 了
                    print(
                        "  [Warn] 没找到 overlap 帧的全局外参，"
                        "当前 chunk 外参直接拼接（会有坐标系跳变）"
                    )
                    depth_to_use = depth_chunk
                    intr_to_use = intr_chunk
                    extr_to_use = extr4
                else:
                    E_global_anchor = global_extrinsics_4x4[anchor_global_idx]  # 4x4
                    E_local_anchor = extr4[anchor_local_idx]  # 4x4

                    # A 把“当前 chunk 世界系”映射到“全局世界系”:
                    # E_global = E_local * A  =>  A = inv(E_local) @ E_global
                    A = np.linalg.inv(E_local_anchor) @ E_global_anchor

                    # 把当前 chunk 的所有外参转到全局世界系:
                    extr4_aligned = extr4 @ A

                    # 真正重叠了多少帧
                    num_overlap = max(0, min(prev_end - start, S))

                    # 已经在前一个 chunk 中保存过的重叠帧，这里就丢掉
                    depth_to_use = depth_chunk[num_overlap:]
                    intr_to_use = intr_chunk[num_overlap:]
                    extr_to_use = extr4_aligned[num_overlap:]

            # 累积
            all_depth_chunks.append(depth_to_use)
            all_intr_chunks.append(intr_to_use)
            all_extr_chunks.append(extr_to_use)

            # 更新全局外参列表（按“全局帧 index”的顺序）
            for ex in extr_to_use:
                global_extrinsics_4x4.append(ex)

            # 打包 depth -> BGR 用于写视频（只对“新帧”做）
            for d in depth_to_use:
                all_bitpacked_frames.append(encode_depth_to_rgb_bitpack(d))

            prev_end = end
            seg_idx += 1

            # 清理中间变量，方便显存 / 内存回收
            del frames_np, pil_images, depth_chunk, intr_chunk, extr_chunk, extr4

            # 下一段起点
            if pose_overlap <= 0:
                start += chunk_size
            else:
                start += step

        # 合并
        depth_full = np.concatenate(all_depth_chunks, axis=0)
        intr_full = np.concatenate(all_intr_chunks, axis=0)
        extr_full = np.concatenate(all_extr_chunks, axis=0)  # [N, 4, 4]

        assert (
            depth_full.shape[0] == num_frames
        ), f"Depth 帧数不等于视频帧数: {depth_full.shape[0]} vs {num_frames}"
        assert (
            intr_full.shape[0] == num_frames
        ), f"Intrinsics 帧数不等于视频帧数: {intr_full.shape[0]} vs {num_frames}"
        assert (
            extr_full.shape[0] == num_frames
        ), f"Extrinsics 帧数不等于视频帧数: {extr_full.shape[0]} vs {num_frames}"

        # 保存 npy
        np.save(depth_npy_path, depth_full.astype(np.float32))
        np.save(intr_npy_path, intr_full.astype(np.float32))
        np.save(extr_npy_path, extr_full.astype(np.float32))
        print(f"  Saved depth npy      : {depth_npy_path}, shape={depth_full.shape}")
        print(f"  Saved intrinsics npy : {intr_npy_path}, shape={intr_full.shape}")
        print(f"  Saved extrinsics npy : {extr_npy_path}, shape={extr_full.shape}")

        # 保存深度视频
        depth_frames_array = np.asarray(all_bitpacked_frames, dtype=np.uint8)
        assert depth_frames_array.shape[0] == num_frames
        vwrite(
            depth_video_path,
            depth_frames_array,
            outputdict={
                "-vcodec": "libx264",
                "-pix_fmt": "yuv420p",
                "-crf": "17",
                "-preset": "veryslow",
            },
        )
        print(
            f"  Saved depth video    : {depth_video_path}, shape={depth_frames_array.shape}"
        )
