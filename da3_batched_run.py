import os
import sys
import glob
import argparse
from typing import List, Tuple

import numpy as np
import torch

# 假设 depth-anything-3 目录就在当前工作目录下
sys.path.append(os.path.join(os.getcwd(), "depth-anything-3"))

from depth_anything_3.api import DepthAnything3  # noqa: E402
from decord import VideoReader, cpu  # noqa: E402
from PIL import Image  # noqa: E402
from skvideo.io import vwrite  # noqa: E402


# ===========================
# 工具函数：深度 <-> RGB 打包
# ===========================
def encode_depth_to_rgb_bitpack(depth_map: np.ndarray) -> np.ndarray:
    """
    将 metric 深度图 (float32, 单位: 米) 打包成 24bit 的 BGR 图像 (uint8)。
    - 精度：毫米级 (mm)
    - 最大深度：2^24 mm / 1000 ≈ 16.7 km
    """
    depth_mm = (depth_map * 1000.0).astype(np.uint32)  # -> 毫米整数

    r = (depth_mm >> 16) & 0xFF  # 高 8 位
    g = (depth_mm >> 8) & 0xFF  # 中 8 位
    b = depth_mm & 0xFF  # 低 8 位

    img_bgr = np.dstack((b.astype(np.uint8), g.astype(np.uint8), r.astype(np.uint8)))
    return img_bgr


def decode_rgb_to_depth_bitpack(img_bgr: np.ndarray) -> np.ndarray:
    """
    反解码：BGR 24bit -> metric 深度 (米)。
    这里仅作为对称函数保留，你当前流程用不到。
    """
    img_val = img_bgr.astype(np.uint32)
    b, g, r = img_val[..., 0], img_val[..., 1], img_val[..., 2]
    depth_mm = (r << 16) | (g << 8) | b
    depth_m = depth_mm.astype(np.float32) / 1000.0
    return depth_m


# ===========================
# 工具函数：读内参 -> focal
# ===========================
def compute_focal_from_intrinsics(intrinsics_path: str) -> float:
    """
    从 intrinsics.npy 里计算平均焦距（像素单位）。
    兼容:
    - (N, 3, 3)
    - (3, 3)
    """
    intr = np.load(intrinsics_path)

    if intr.ndim == 3:  # (N, 3, 3)
        fx = intr[:, 0, 0]
        fy = intr[:, 1, 1]
        focal = ((fx + fy) * 0.5).mean()
    elif intr.ndim == 2:  # (3, 3)
        fx = intr[0, 0]
        fy = intr[1, 1]
        focal = (fx + fy) * 0.5
    else:
        raise ValueError(
            f"Unexpected intrinsics shape: {intr.shape}, " "expect (3,3) or (N,3,3)"
        )

    return float(focal)


# ===========================
# Worker 类：负责单卡上的模型 + 单个视频的处理
# ===========================
class DepthAnythingVideoWorker:
    def __init__(self, device: str, model_name: str = "depth-anything/DA3METRIC-LARGE"):
        """
        device: 例如 "cuda:0" / "cuda:1" / "cpu"
        """
        self.device = torch.device(device)
        print(f"[Init] Loading model on device: {self.device}")
        self.model = DepthAnything3.from_pretrained(model_name)
        self.model = self.model.to(device=self.device)
        self.model.eval()

    @torch.no_grad()
    def _infer_depth_chunk(
        self, pil_images: List[Image.Image], process_res: int | None = None
    ):
        """
        对一个批次（chunk）的 PIL 图像进行深度推理。
        返回:
        - depth: np.ndarray [N, H, W] float32
        """
        kwargs = {}
        if process_res is not None:
            kwargs["process_res"] = process_res

        prediction = self.model.inference(pil_images, **kwargs)
        # 根据你原始代码：prediction.depth 是 [N, H, W] float32 ndarray
        depth = prediction.depth
        return depth

    def process_video(
        self,
        video_path: str,
        output_dir: str,
        focal: float,
        chunk_size: int = 1000,
        process_res: int | None = None,
        metric_scale: float = 300.0,
    ) -> Tuple[str, str]:
        """
        对单个视频进行完整处理：
        - 分 chunk 推理深度
        - 合并 metric 深度保存为 npy
        - 将 metric 深度打包成 RGB 再写成 mp4

        返回:
        - depth_npy_path
        - depth_video_path
        """
        os.makedirs(output_dir, exist_ok=True)

        # 取文件名
        base_name = os.path.basename(video_path)
        stem, _ = os.path.splitext(base_name)

        depth_npy_path = os.path.join(output_dir, f"{stem}_metric_depth.npy")
        depth_video_path = os.path.join(output_dir, f"{stem}_metric_depth.mp4")

        print(f"[Video] {video_path}")
        print(f"  -> depth npy: {depth_npy_path}")
        print(f"  -> depth mp4: {depth_video_path}")
        print(f"  -> device: {self.device}")

        # 用 Decord 逐 chunk 读取
        vr = VideoReader(video_path, ctx=cpu(0))
        num_frames = len(vr)
        print(f"  Total frames: {num_frames}")

        all_metric_depth_chunks: List[np.ndarray] = []
        all_bitpacked_frames: List[np.ndarray] = []

        # 遍历 chunk
        for start in range(0, num_frames, chunk_size):
            end = min(start + chunk_size, num_frames)
            idxs = list(range(start, end))
            print(f"  Chunk {start} : {end} (size={len(idxs)})")

            # Decord 批量拿一块
            frames_np = vr.get_batch(idxs).asnumpy()  # [chunk, H, W, 3], uint8
            pil_images = [Image.fromarray(frame) for frame in frames_np]

            # 深度推理 (原始 depth)
            depth_chunk = self._infer_depth_chunk(
                pil_images, process_res=process_res
            )  # [chunk, H, W]

            # metric 深度
            metric_depth_chunk = focal * depth_chunk / float(metric_scale)
            metric_depth_chunk = metric_depth_chunk.astype(np.float32)

            # 存到 list，待会儿 concat
            all_metric_depth_chunks.append(metric_depth_chunk)

            # 打包成 BGR 用于写视频
            for d in metric_depth_chunk:
                bgr = encode_depth_to_rgb_bitpack(d)
                all_bitpacked_frames.append(bgr)

            # 清理一下引用，帮助显存回收
            del frames_np, pil_images, depth_chunk, metric_depth_chunk

        # 合并所有 chunk 的 metric 深度
        metric_depth_full = np.concatenate(all_metric_depth_chunks, axis=0)
        assert (
            metric_depth_full.shape[0] == num_frames
        ), f"Frame count mismatch: {metric_depth_full.shape[0]} vs {num_frames}"

        # 保存 npy，保证一帧一个深度图
        np.save(depth_npy_path, metric_depth_full)
        print(f"  Saved depth npy: {depth_npy_path}, shape={metric_depth_full.shape}")

        # 保存 BGR 视频
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
            f"  Saved depth video: {depth_video_path}, shape={depth_frames_array.shape}"
        )

        return depth_npy_path, depth_video_path


# ===========================
# 批处理：多目录 / 多视频 / 多卡分配
# ===========================
def find_videos(
    input_dirs: List[str], exts: Tuple[str, ...] = (".mp4", ".MP4")
) -> List[str]:
    """
    在若干目录下查找所有视频文件。
    默认只找 mp4，如需扩展可以自己加后缀。
    """
    all_files = []
    for d in input_dirs:
        d = os.path.abspath(d)
        for ext in exts:
            pattern = os.path.join(d, f"*{ext}")
            all_files.extend(glob.glob(pattern))
    all_files = sorted(set(all_files))
    return all_files


def main():
    parser = argparse.ArgumentParser(
        description="Depth Anything v3 metric depth on videos (chunked, multi-GPU)."
    )
    parser.add_argument(
        "--input_dirs",
        nargs="+",
        required=True,
        help="一个或多个目录，这些目录下的 .mp4 都会被处理",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="输出目录，深度 npy 和 mp4 都会保存在这里",
    )
    parser.add_argument(
        "--intrinsics",
        required=True,
        help="intrinsics.npy 的路径，用于计算 focal（metric 深度用）",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="视频切片长度（按帧），默认 1000 帧一块",
    )
    parser.add_argument(
        "--process_res",
        type=int,
        default=None,
        help=(
            "传给 DepthAnything3.inference 的 process_res，"
            "默认 None 使用模型默认分辨率；"
            "如想贴合原代码，可设为视频宽度。"
        ),
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="depth-anything/DA3METRIC-LARGE",
        help="Depth Anything v3 模型权重名",
    )
    parser.add_argument(
        "--metric_scale",
        type=float,
        default=300.0,
        help="metric 深度缩放系数，默认与你原来的一致 focal * depth / 300",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 读取内参，计算 focal
    focal = compute_focal_from_intrinsics(args.intrinsics)
    print(f"[Info] Focal (pixels): {focal}")

    # 查找所有要处理的视频
    videos = find_videos(args.input_dirs)
    if not videos:
        print("[Warning] 没找到任何 mp4 视频，请检查 input_dirs。")
        return

    print(f"[Info] Found {len(videos)} video(s).")

    # 检测设备（多卡）
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        devices = [f"cuda:{i}" for i in range(num_gpus)]
    else:
        print("[Warning] 没有检测到 GPU，将使用 CPU（会非常慢）。")
        devices = ["cpu"]

    print(f"[Info] Devices: {devices}")

    # 为每张卡初始化一个 worker
    workers = [
        DepthAnythingVideoWorker(device=d, model_name=args.model_name) for d in devices
    ]

    # 轮询分配视频给不同的 worker
    num_workers = len(workers)
    for idx, video_path in enumerate(videos):
        worker = workers[idx % num_workers]
        print(
            f"\n========== Processing video {idx + 1}/{len(videos)} "
            f"on {worker.device} =========="
        )
        worker.process_video(
            video_path=video_path,
            output_dir=args.output_dir,
            focal=focal,
            chunk_size=args.chunk_size,
            process_res=args.process_res,
            metric_scale=args.metric_scale,
        )


if __name__ == "__main__":
    main()
