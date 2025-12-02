# run_da3nested_multi_gpu.py
import os
import argparse
import multiprocessing as mp

import torch

from da3nested_lib import DA3NestedVideoWorker, find_videos


def worker_process(device: str, video_list, args):
    """
    单个进程：绑定到一张 GPU 上，顺序处理分配给它的所有视频。
    """
    if not video_list:
        print(f"[{device}] No videos assigned, exit.")
        return

    print(f"[{device}] Will process {len(video_list)} video(s).")
    worker = DA3NestedVideoWorker(
        device=device, model_name=args.model_name, intr_path=args.intr_path
    )

    for idx, video_path in enumerate(video_list):
        print(f"\n[{device}] Processing {idx + 1}/{len(video_list)}: {video_path}")
        worker.process_video(
            video_path=video_path,
            output_dir=args.output_dir,
            chunk_size=args.chunk_size,
            process_res=args.process_res,
            pose_overlap=args.pose_overlap,
            use_ray_pose=True,
        )


def main():
    parser = argparse.ArgumentParser(
        description="DA3NESTED-GIANT-LARGE depth+pose, " "chunked + multi-GPU parallel."
    )
    parser.add_argument(
        "--input_dirs",
        nargs="+",
        required=True,
        help="一个或多个目录，这些目录下的 .mp4 会被处理",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="输出目录，npy 和 mp4 都存这里",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="每次送进模型的最大帧数，默认 1000",
    )
    parser.add_argument(
        "--pose_overlap",
        type=int,
        default=1,
        help="相邻 chunk 之间用于对齐 pose 的重叠帧数，默认 1。",
    )
    parser.add_argument(
        "--process_res",
        type=int,
        default=512,
        help="process_res (通常设成视频宽度)，默认 504",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="depth-anything/DA3NESTED-GIANT-LARGE",
        help="DA3 模型名称，默认 DA3NESTED-GIANT-LARGE",
    )
    parser.add_argument(
        "--intr_path",
        type=str,
        default="",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 找到所有要处理的视频
    rawvideos = find_videos(args.input_dirs)
    videos = []
    if not rawvideos:
        print("[Warn] 没找到 mp4 视频，请检查 input_dirs。")
        return
    for v in rawvideos:
        if os.path.exists(
            os.path.join(
                args.output_dir,
                os.path.basename(v).replace(".mp4", "_depth_da3nested.npy"),
            )
        ):
            print(f"[Info] 视频 {v} 已处理，跳过。")
        else:
            videos.append(v)
    print(f"[Info] Found {len(videos)} video(s).")

    # 检测可用设备
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        devices = [f"cuda:{i}" for i in range(num_gpus)]
    else:
        print("[Warn] 没检测到 GPU，将使用 CPU（会很慢）")
        devices = ["cpu"]

    print(f"[Info] Devices: {devices}")

    # 把视频列表平均分配给各个设备（简单 round-robin）
    per_device_videos = [[] for _ in devices]
    for idx, v in enumerate(videos):
        per_device_videos[idx % len(devices)].append(v)

    # 起多个进程，每张卡一个进程
    processes = []
    for dev, vlist in zip(devices, per_device_videos):
        if not vlist:
            continue
        p = mp.Process(target=worker_process, args=(dev, vlist, args))
        p.start()
        processes.append(p)

    # 等待所有子进程结束
    for p in processes:
        p.join()

    print("[Info] All processes finished.")


if __name__ == "__main__":
    # 对于 torch + 多进程，spawn 更安全
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        # 已经设置过的话就忽略
        pass
    main()
