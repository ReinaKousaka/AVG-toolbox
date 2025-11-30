import os, sys

sys.path.append(os.path.join(os.getcwd(), "depth-anything-3"))
import glob, os, torch
from depth_anything_3.api import DepthAnything3
from decord import VideoReader, cpu
from PIL import Image
from skvideo.io import vwrite

VIDEO_PATH = "/workspace/AVG-toolbox/raw_2077-11-25/part_1/Cyberpunk207720251117-05250004_proc_temp_part_000.mp4"
VIDEO_WIDTH = 504
CLIP_FRAMES = 1000


def video_to_pil_decord(video_path):
    """使用 Decord 将视频解析为 PIL Image 列表 (通常比 OpenCV 更快)"""
    # ctx=cpu(0) 指定使用 CPU 解码，如果显卡支持且显存足够，可以用 gpu(0)
    vr = VideoReader(video_path, ctx=cpu(0))
    # 一次性获取所有帧（Decord 的优化核心）
    # 这比循环读取要快得多，但会占用更多瞬时内存
    frames_array = vr.get_batch(range(len(vr))).asnumpy()
    # 批量转换为 PIL
    pil_images = [Image.fromarray(frame) for frame in frames_array]
    return pil_images


device = torch.device("cuda")
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
model = model.to(device=device)

images = video_to_pil_decord(VIDEO_PATH)[:CLIP_FRAMES]
prediction = model.inference(images, process_res=VIDEO_WIDTH, use_ray_pose=True)

import numpy as np


def encode_depth_to_rgb_bitpack(depth_map):
    """将深度图按位拆分到 RGB。
    前提：假设深度单位是米，我们需要毫米级精度。
    限制：最大深度 2^24 毫米 / 1000 = 16.7 公里 (足够了)"""
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
    """解码"""
    img_val = img_bgr.astype(np.uint32)
    b, g, r = img_val[..., 0], img_val[..., 1], img_val[..., 2]
    # 组合回整数
    depth_mm = (r << 16) | (g << 8) | b
    # 转回米
    depth_m = depth_mm.astype(np.float32) / 1000.0
    return depth_m


# prediction.processed_images : [N, H, W, 3] uint8 array
print(prediction.processed_images.shape)
# prediction.depth : [N, H, W] float32 array
print(prediction.depth.shape)

# intrinsics = np.load("example_intrinsics.npy")
# focal = (intrinsics[:, 0, 0] + intrinsics[:, 1, 1]) / 2.0
# focal = focal.mean()

depth = prediction.depth
# metric_depth = focal * depth / 300
bit_depth = []
np.save("test_da3_DA3NESTED-GIANT-LARGE_depth.npy", depth)
# np.save("test_da3_DA3NESTED-GIANT-LARGE_metric_depth.npy", metric_depth)
np.save("test_da3_DA3NESTED-GIANT-LARGE_intrinsics.npy", prediction.intrinsics)
np.save("test_da3_DA3NESTED-GIANT-LARGE_extrinsics.npy", prediction.extrinsics)
for d in depth:
    bit_depth.append(encode_depth_to_rgb_bitpack(d))

vwrite(
    "test_da3_depth_Large.mp4",
    np.array(bit_depth),
    outputdict={
        "-vcodec": "libx264",
        "-pix_fmt": "yuv420p",
        "-crf": "17",
        "-preset": "veryslow",
    },
)
