import argparse
import json
import tempfile
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import cv2, time
import numpy as np
import torch
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    Qwen3VLMoeForConditionalGeneration,
)

# === 新增：多进程 & 多 GPU 相关 ===
import os
import multiprocessing as mp


def load_qwen3_vl(
    model_id: str = "Qwen/Qwen3-VL-30B-Instruct",
    device: str | None = None,
):
    """
    按 Qwen 官方推荐方式加载本地 Qwen3-VL 模型和 Processor。

    注意：
    - 需要 transformers >= 4.57.0，否则 AutoProcessor 会各种奇怪报错。
    """
    # 如果没有显式指定 device，就退回原来的 auto 行为（单 GPU 时也没问题）
    if device is None:
        device_map = "auto"
    else:
        # 显式把整个模型放到某一块卡上，例如 "cuda:0" / "cuda:1"
        device_map = {"": device}

    if model_id == "Qwen/Qwen3-VL-30B-A3B-Instruct":
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-30B-A3B-Instruct",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device_map,
        )
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            device_map=device_map,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def summarize_segment_with_qwen3_vl(
    frames_bgr: List[np.ndarray],
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    max_images: int = 8,
    prompt: str = """
        test
        """,
    num_tokens=128,
) -> str:
    """
    对一段视频帧（已下采样）调用 Qwen3-VL 做总结。

    实现完全参考 Qwen 官方 transformers 示例：
    - 用 AutoProcessor.apply_chat_template 构造输入
    - model.generate 生成
    - processor.batch_decode 解码
    """

    if not frames_bgr:
        return ""

    # 为了控制开销，从这一段中均匀采样若干帧
    if len(frames_bgr) > max_images:
        indices = np.linspace(0, len(frames_bgr) - 1, max_images, dtype=int)
        sampled_frames = [frames_bgr[i] for i in indices]
    else:
        sampled_frames = frames_bgr

    # 临时将帧写成本地 jpg，用本地路径形式提供给 Qwen3-VL
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        contents = []

        for i, frame in enumerate(sampled_frames):
            img_path = tmpdir / f"frame_{i:04d}.jpg"
            # OpenCV 写 BGR -> jpg
            cv2.imwrite(str(img_path), frame)
            uri = str(img_path)
            contents.append(
                {
                    "type": "image",
                    "image": uri,  # 官方 README 用的字段名是 image（本地/URL 都可以）
                }
            )

        contents.append({"type": "text", "text": prompt})

        messages = [
            {
                "role": "user",
                "content": contents,
            }
        ]

        # 准备输入，严格按官方示例写法来
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        # 如果你的 transformers 版本生成了 token_type_ids，官方文档建议丢掉它
        inputs.pop("token_type_ids", None)

        # BatchEncoding 有 .to() 方法，可以直接整体搬到模型设备上
        inputs = inputs.to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=num_tokens)

        input_ids = inputs["input_ids"]
        # 去掉 prompt 部分，只保留新生成的 token
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)
        ]

        output_texts = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return str(output_texts[0])


def summarize_video_by_frames(
    video_path: str,
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    segment_size: int = 32,
    downscale_ratio: float = 0.5,
) -> Dict[str, str]:
    """
    对视频按连续 segment_size 帧切段：
    - 每帧按 downscale_ratio 做分辨率缩小
    - 对每一段调用 Qwen3-VL 总结
    - 返回 {"startFrame-endFrame": "描述"}
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")

    summaries: Dict[str, str] = {}

    current_segment_frames: List[np.ndarray] = []
    current_segment_simple_frames = []
    segment_start_frame_idx = 0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 按比例下采样当前帧
        if downscale_ratio != 1.0:
            h, w = frame.shape[:2]
            new_w = max(1, int(w * downscale_ratio))
            new_h = max(1, int(h * downscale_ratio))
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            simple_frame = cv2.resize(
                frame, (new_w // 2, new_h // 2), interpolation=cv2.INTER_AREA
            )
        current_segment_frames.append(frame)
        current_segment_simple_frames.append(simple_frame)
        # 凑够一段，就让 Qwen3-VL 总结这一段
        if len(current_segment_frames) == segment_size:
            segment_end_frame_idx = frame_idx
            start = time.time()
            first = summarize_segment_with_qwen3_vl(
                [current_segment_frames[0]],
                model,
                processor,
                prompt="In English, describe the frame with all visible objects in detail and their spatial positions relative to the viewer. Do not include escape characters in the response. the description should be around 64 words.",
                num_tokens=128,
            )
            remaining = summarize_segment_with_qwen3_vl(
                current_segment_frames[1:],
                model,
                processor,
                prompt="In English, describe frames detailly, do not describe the first frame, focusing on objects and camera movements, especially dynamic object, describe how they move. Ensure descriptions are chronologically ordered, accurate, information-rich. Do not include escape characters in the response. the description should be around 128 words.",
                num_tokens=256,
                max_images=12,
            )
            caption_simple = summarize_segment_with_qwen3_vl(
                current_segment_simple_frames,
                model,
                processor,
                prompt=f"Summarize the content {first}, {remaining} to around 50 English words. Do not include escape characters in the response.",
                num_tokens=128,
                max_images=6,
            )
            caption = {"first": first, "remaining": remaining, "simple": caption_simple}
            assert isinstance(caption, dict)
            key = f"{str(segment_start_frame_idx).zfill(6)}-{str(segment_end_frame_idx).zfill(6)}"
            # print(f"Segment {key} summary: {caption_simple} \n Full: {caption}")
            summaries[key] = caption
            end = time.time()
            print(
                f"Qwen3-VL took {end - start:.2f} seconds for segment {key}. caption: {caption}"
            )

            # 开启下一段
            current_segment_frames = []
            current_segment_simple_frames = []
            segment_start_frame_idx = frame_idx + 1

        frame_idx += 1

    # 处理视频尾巴那一段（不足 segment_size 帧）
    if current_segment_frames:
        segment_end_frame_idx = frame_idx - 1
        first = summarize_segment_with_qwen3_vl(
            [current_segment_frames[0]],
            model,
            processor,
            prompt="In English, describe the frame with all visible objects in detail and their spatial positions relative to the viewer",
        )
        remaining = summarize_segment_with_qwen3_vl(
            current_segment_frames[1:],
            model,
            processor,
            prompt="In English, describe dynamic changes and newly revealed objects or scenes related to the first frame of the video, focusing on describing the objects in the frames and describe how they move when object is a dynamic one. Ensure descriptions are chronologically ordered, accurate, information-rich.",
        )
        caption_simple = summarize_segment_with_qwen3_vl(
            current_segment_simple_frames,
            model,
            processor,
            prompt="summarize the content in the images around 50 English words.",
        )
        caption = {"first": first, "remaining": remaining, "simple": caption_simple}
        key = f"{str(segment_start_frame_idx).zfill(6)}-{str(segment_end_frame_idx).zfill(6)}"
        summaries[key] = caption

    cap.release()
    return summaries


# === 新增：在一个进程里，用某一块 GPU 处理若干个视频 ===
def worker_process(
    gpu_index: int,
    video_paths: List[str],
    model_id: str,
    segment_size: int,
    downscale_ratio: float,
    out_dir: str,
    log_file_path: str,
):
    """
    每个进程绑定到一个物理 GPU，加载一次模型，然后依次处理分配到的视频。
    """
    log_path = Path(log_file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        log_path.open("w", encoding="utf-8", buffering=1) as log_file,
        redirect_stdout(log_file),
        redirect_stderr(log_file),
    ):
        # 显式指定当前进程的默认 GPU（用于一些内部调用）
        torch.cuda.set_device(gpu_index)
        device = f"cuda:{gpu_index}"
        print(f"[Worker GPU {gpu_index}] using device: {device}")
        print(f"[Worker GPU {gpu_index}] processing {len(video_paths)} videos.")
        # 在当前进程 / 当前 GPU 上加载模型：显式 device_map
        model, processor = load_qwen3_vl(model_id, device=device)
        out_dir_path = Path(out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)
        for v in tqdm(video_paths):
            v_path = Path(v)
            print(f"[Worker GPU {gpu_index}] Start video: {v_path}")
            try:
                summaries = summarize_video_by_frames(
                    video_path=str(v_path),
                    model=model,
                    processor=processor,
                    segment_size=segment_size,
                    downscale_ratio=downscale_ratio,
                )
                out_json = out_dir_path / f"{v_path.stem}.json"
                with out_json.open("w", encoding="utf-8") as f:
                    json.dump(summaries, f, ensure_ascii=False, indent=2)
                print(
                    f"[Worker GPU {gpu_index}] Finished {v_path}, saved to {out_json}"
                )
            except Exception as e:
                print(f"[Worker GPU {gpu_index}] Error processing {v_path}: {e}")


# === 新增：多 GPU / 多进程调度逻辑 ===
def run_multi_gpu(
    input_dir: str,
    out_dir: str,
    model_id: str,
    segment_size: int,
    downscale_ratio: float,
    num_gpus: int | None = None,
):
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise ValueError(f"input_dir 不是目录: {input_dir}")

    # 收集该目录下的所有 mp4（不递归，若要递归可改成 rglob）
    video_paths = sorted(str(p) for p in input_path.glob("*.mp4"))
    if not video_paths:
        raise ValueError(f"在目录 {input_dir} 下没有找到任何 .mp4 文件")

    total_videos = len(video_paths)

    # 自动检测 GPU 数量
    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        raise RuntimeError("没有检测到可用的 GPU。")

    if num_gpus is None:
        num_gpus = available_gpus
    else:
        num_gpus = min(num_gpus, available_gpus)

    # 不要开比视频还多的进程
    num_gpus = min(num_gpus, total_videos)

    print(
        f"Found {total_videos} videos in {input_dir}, "
        f"using {num_gpus} GPU(s) out of {available_gpus} available."
    )

    # 按视频数 V 和 GPU 数 G，尽量均匀切成 G 份
    # 简单做法：连续均匀切片
    chunks: List[List[str]] = [[] for _ in range(num_gpus)]
    for idx, v in enumerate(video_paths):
        chunks[idx % num_gpus].append(v)

    log_dir = Path(f"qwen_log_{str(input_path).split('/')[-1]}")
    log_dir.mkdir(parents=True, exist_ok=True)

    processes: List[mp.Process] = []

    for gpu_idx in range(num_gpus):
        vids = chunks[gpu_idx]
        if not vids:
            continue
        log_file = log_dir / f"gpu_{gpu_idx}.log"
        p = mp.Process(
            target=worker_process,
            args=(
                gpu_idx,
                vids,
                model_id,
                segment_size,
                downscale_ratio,
                out_dir,
                str(log_file),
            ),
        )

        p.start()
        processes.append(p)
        print(f"Spawned worker for GPU {gpu_idx} with {len(vids)} videos.")

    # 等待所有进程结束
    for p in processes:
        p.join()
        print(f"Worker PID {p.pid} finished.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "本地 Qwen3-VL 对视频每 N 帧一段做概括：\n"
            "- 单视频模式：输入 video_path，输出一个 JSON\n"
            "- 多 GPU 模式：输入 input_dir 和 out_dir，目录下每个 mp4 输出一个 JSON"
        )
    )
    # 原来的单视频模式入口，保持兼容；现在可选
    parser.add_argument(
        "video_path",
        type=str,
        nargs="?",
        help="单视频模式：输入 mp4 视频路径（如果使用 input_dir 则可以留空）",
    )

    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen3-VL-30B-A3B-Instruct",
        help="Qwen3-VL 模型 ID，默认 Qwen/Qwen3-VL-8B-Instruct",
    )
    parser.add_argument(
        "--segment_size",
        type=int,
        default=32,
        help="每个片段的帧数 N（默认 32）",
    )
    parser.add_argument(
        "--downscale_ratio",
        type=float,
        default=0.5,
        help="下采样比例，例如 0.5 表示分辨率缩小为原来的 1/2（默认 0.5）",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="video_summaries_qwen3vl.json",
        help="单视频模式下的输出 JSON 文件路径（默认 video_summaries_qwen3vl.json）",
    )

    # 新增：多 GPU / 多视频模式参数
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="多 GPU 模式：包含若干 mp4 视频的文件夹路径",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="多 GPU 模式：输出 JSON 的目录，每个视频一个同名 .json",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="多 GPU 模式：使用的 GPU 数量（默认使用所有可见 GPU）",
    )

    args = parser.parse_args()

    # 如果指定了 input_dir，则走多 GPU / 多进程模式
    if args.input_dir is not None:
        if args.out_dir is None:
            raise ValueError("多 GPU 模式下必须指定 --out_dir")
        run_multi_gpu(
            input_dir=args.input_dir,
            out_dir=args.out_dir,
            model_id=args.model_id,
            segment_size=args.segment_size,
            downscale_ratio=args.downscale_ratio,
            num_gpus=args.num_gpus,
        )
        return

    # 否则，保持原有单视频逻辑不变
    if not args.video_path:
        raise ValueError(
            "单视频模式下必须提供 video_path，或者改用 --input_dir 多视频模式。"
        )

    print(f"加载模型 {args.model_id} ...")
    model, processor = load_qwen3_vl(args.model_id)

    print("开始处理视频并调用 Qwen3-VL ...")
    summaries = summarize_video_by_frames(
        video_path=args.video_path,
        model=model,
        processor=processor,
        segment_size=args.segment_size,
        downscale_ratio=args.downscale_ratio,
    )

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    print(f"已保存结果到 {args.output_json}")


if __name__ == "__main__":
    # 在大多数环境下推荐 spawn，避免 CUDA 在父进程里初始化后被子进程复用出问题
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # 已经设置过 start method 的情况
        pass
    main()
