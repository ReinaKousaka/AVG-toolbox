import OpenEXR, Imath
from pathlib import Path
import zipfile, tempfile
import numpy as np


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
            with z.open(name, "r") as f, tempfile.NamedTemporaryFile(
                suffix=".exr"
            ) as tmp:
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


import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm  # 用于显示进度条


def save_depth_to_heatmap_video(
    depth_stack, output_path, fps=60, max_threshold=128.0, colormap="inferno"
):
    """
    将 3D 深度 numpy 数组转换为热力图视频。

    Args:
        depth_stack (np.ndarray): 形状为 (Frames, H, W) 的 float32 深度数据。
        output_path (str): 输出视频路径 (例如 'depth_video.mp4')。
        fps (int): 视频帧率。由于有8000帧，建议设置高一点，比如 60。
        max_threshold (float): 截断阈值。大于此距离的值将被截断。用于解决数据分布不均问题。
        colormap (str): Matplotlib 的色谱名称，例如 'inferno', 'magma', 'plasma', 'jet', 'turbo'。
                        'inferno' 或 'magma' 对深度图通常效果很好（近处亮黄，远处暗黑）。
    """
    frames, height, width = depth_stack.shape
    print(f"Input Shape: {depth_stack.shape}")
    print(
        f"Data Range (Before Processing): Min={depth_stack.min():.2f}, Max={depth_stack.max():.2f}"
    )

    # --- 1. 初始化视频写入器 ---
    # 使用 mp4v 编码器 (H.264)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # --- 2. 获取 Matplotlib 色谱对象 ---
    # 它可以直接接收 numpy 数组并将数值映射为颜色
    cmap = plt.get_cmap(colormap)

    print(f"Starting conversion to video with threshold={max_threshold}...")

    # 使用 tqdm 显示进度条
    for i in tqdm(range(frames), desc="Processing Frames"):
        depth_frame = depth_stack[i]

        depth_frame = encode_depth_to_rgb_bitpack(depth_frame)

        # 2. 转换为 RGB uint8
        # 取前三个通道 (RGB)，乘以 255，转换为 8位无符号整数
        colored_rgb_uint8 = depth_frame.astype(np.uint8)

        # 3. 颜色空间转换 (RGB -> BGR)
        # OpenCV 内部使用 BGR 顺序，而 Matplotlib 生成的是 RGB
        colored_bgr = cv2.cvtColor(colored_rgb_uint8, cv2.COLOR_RGB2BGR)

        # 写入帧
        video_writer.write(colored_bgr)

    # 释放资源
    video_writer.release()
    print(f"\nVideo saved successfully to: {output_path}")

def encode_depth_to_rgb_bitpack(depth_map):
    """
    将深度图按位拆分到 RGB。
    前提：假设深度单位是米，我们需要毫米级精度。
    限制：最大深度 2^24 毫米 / 1000 = 16.7 公里 (足够了)
    """
    # 1. 转换为毫米级的整数
    depth_mm = (depth_map * 1000).astype(np.uint32)
    
    # 2. 位运算拆分
    # R: 高8位
    r = (depth_mm >> 16) & 0xFF
    # G: 中8位
    g = (depth_mm >> 8) & 0xFF
    # B: 低8位
    b = depth_mm & 0xFF
    
    # 3. 堆叠 (注意 OpenCV 是 BGR)
    img_bgr = np.dstack((b.astype(np.uint8), g.astype(np.uint8), r.astype(np.uint8)))
    
    return img_bgr
    
def decode_rgb_to_depth_bitpack(img_bgr):
    """
    解码
    """
    img_val = img_bgr.astype(np.uint32)
    b, g, r = img_val[..., 0], img_val[..., 1], img_val[..., 2]
    
    # 组合回整数
    depth_mm = (r << 16) | (g << 8) | b
    
    # 转回米
    depth_m = depth_mm.astype(np.float32) / 1000.0
    return depth_m

if __name__ == "__main__":
    depth = load_depth_zip_to_array(
        "/workspace/AVG-toolbox/vipe_results/depth/Cyberpunk207720251117-05250004_proc_temp_part_000.zip"
    )[:200]
    save_depth_to_heatmap_video(
        depth,
        output_path="Cyberpunk207720251117.mp4",
        fps=60,
        max_threshold=255,
        colormap="rainbow",
    )
    print(depth.shape, depth.dtype, depth.min(), depth.max())
