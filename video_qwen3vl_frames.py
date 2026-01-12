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
from skvideo.io import vwrite


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
    max_images: int = 10,
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


def chunk_with_stride(seq, chunk_size=3, stride=2):
    chunk = [
        seq[i : i + chunk_size] for i in range(0, len(seq) - chunk_size + 1, stride)
    ]
    chunk.append([chunk[-1][-1], seq[-1]])
    return chunk


def summarize_video_by_frames(
    video_path: str,
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    frame_interval: int = 10,
    simple_block_size: int = 4,
    downscale_ratio: float = 0.5,
    detail_chunk: int = 3,
) -> Dict[str, Dict[str, str]]:
    """
    对视频按帧间隔抽帧并标注：
    - frame_interval: 抽帧间隔，例如 10 表示每隔 10 帧抽一帧（0, 10, 20, 30...）
    - simple_block_size: 简单标注的分组大小，例如 4 表示每 4 帧共享一个简单标注
    - downscale_ratio: 分辨率缩小比例
    - 返回 {frameIdx: {"detailed": "...", "simple": "..."}}
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 收集需要抽取的帧
    frame_indices = list(range(0, total_frames, frame_interval))
    extracted_frames: Dict[int, np.ndarray] = {}
    simple_frames = {}
    print(f"Total frames: {total_frames}, extracting {len(frame_indices)} frames")

    # 抽取所有需要的帧
    for target_idx in tqdm(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ok, frame = cap.read()
        if not ok:
            break

        # 按比例下采样
        if True:
            h, w = frame.shape[:2]
            new_w = max(1, int(w * downscale_ratio) // 2 * 2)
            new_h = max(1, int(h * downscale_ratio) // 2 * 2)
            frame = cv2.resize(
                frame,
                (new_w, new_h),
                interpolation=cv2.INTER_AREA,
            )

            simple_frame = cv2.resize(
                frame,
                (int(w * 0.5) // 2 * 2, int(h * 0.5) // 2 * 2),
                interpolation=cv2.INTER_AREA,
            )
        extracted_frames[target_idx] = frame
        simple_frames[target_idx] = simple_frame

    chunks = chunk_with_stride(
        chunk_size=detail_chunk, seq=frame_indices, stride=detail_chunk - 1
    )
    print(f"chunks: {chunks}")
    cap.release()

    # 对每一帧进行详细标注
    annotations: Dict[str, Dict[str, str]] = {}
    annotations["detailed"] = {}
    frame_list = sorted(extracted_frames.keys())
    is_debug = False
    print(f"Total detail chunks: {len(chunks)}")
    # chunks = chunks[:5]
    for chunk in tqdm(chunks):
        frames = [extracted_frames[idx] for idx in chunk]
        simple_temp_frames = [simple_frames[idx] for idx in chunk]
        detailed_first = summarize_segment_with_qwen3_vl(
            [frames[0]],
            model,
            processor,
            prompt=f"Please summarize the content of the image. Provide a concise within 130 English words. Most words should describe the objects in the video in detail, including their shape, color, texture & material and words if visible, as well as their approximate location in the frame. Then use only very few words to briefly describe the atmosphere, lighting, and other overall information of the image. Avoid starting with lengthy phrases such as [in the picture] or [this is a picture] Get straight to describing the objects.",
            num_tokens=200,
            max_images=1,
        )
        detailed_first = detailed_first.replace("\n", " ").replace("\r", " ")
        detailed_dynamic = summarize_segment_with_qwen3_vl(
            simple_temp_frames,
            model,
            processor,
            prompt=f"Referring to the description of the first frame description: [start first frame description]{detailed_first}[end first frame description], describe the appearance and movement of objects in the video within 130 English words. DO NOT include descriptions of the overall video information such as moods, lights and atmosphere. DO NOT start with lengthy phrases such as [in the subsequent frame] or [in the video], get straight to describing the objects and movements. Also, when no moving object is observed in the scene, DO NOT write [no movement is observed] or [the scene is static] or [xxx remain static / still] or any similar sentences, instead, describe the scene in more detailed appearances. Assume common world knowledge: Buildings, roads, and large structures of the scene are static by default. Do not explicitly state their lack of motion, describe MORE about their appearances. DO NOT include any words related to camera.",
            num_tokens=200,
            max_images=4,
        )
        detailed_dynamic = detailed_dynamic.replace("\n", " ").replace("\r", " ")
        print(f"detailed_first: {detailed_first}")
        print(f"detailed_dynamic: {detailed_dynamic}")
        for anno_idx in list(range(chunk[0], chunk[-1])):
            annotations["detailed"][f"{anno_idx}"] = {
                "start": detailed_first,
                "dynamic": detailed_dynamic,
            }
        if is_debug:
            break
    annotations["simple"] = {}
    # 对每个 block 生成简单标注
    # for block_start_idx in range(0, len(frame_list), simple_block_size):
    #     block_end_idx = min(block_start_idx + simple_block_size, len(frame_list))
    #     block_frames_indices = frame_list[block_start_idx:block_end_idx]
    #     block_frames = [simple_frames[idx] for idx in block_frames_indices]
    #     start = time.time()
    #     simple_dynamic = summarize_segment_with_qwen3_vl(
    #         block_frames,
    #         model,
    #         processor,
    #         prompt=f"describe the appearance and movement of objects in the video in around 100 English words, paying particular attention to moving objects. Describe the objects and their movements in the video briefly. Do not include any camera-related content or descriptions of camera movement in your reply. Avoid describe the overall mood and atmosphere of the video. Avoid starting with lengthy phrases such as [in the frames] or [in the video] Get straight to describing the objects and movements in your response. Also, when no moving object is observed in the scene, DO NOT write [no movement is observed] or similar sentences, instead, describe the static objects in more detail.",
    #         num_tokens=128,
    #         max_images=8,
    #     )
    #     simple = simple_dynamic.replace("\n", " ").replace("\r", " ")
    #     end = time.time()
    #     print(
    #         f"Block {block_frames_indices} simple annotation took {end - start:.2f}s: [start] {simple_dynamic}, [dynamic] {simple_dynamic}"
    #     )

    #     # 将简单标注赋给这个 block 的所有帧
    #     for anno_idx in list(range(block_frames_indices[0], block_frames_indices[-1])):
    #         annotations["simple"][str(anno_idx)] = {
    #             "simple": simple,
    #         }
    #     if is_debug:
    #         break
    return annotations, extracted_frames


def create_annotated_video(
    extracted_frames,
    annotations: Dict[str, Dict[str, str]],
    output_path: str,
):
    """
    创建带标注的视频：
    - 上半部分显示下采样后的视频帧
    - 下半部分显示黑色背景上的标注文字
    - 每个标注对应一定的帧范围，例如 0 帧标注对应 0-9 帧，10 帧标注对应 10-19 帧

    参数:
        video_path: 输入视频路径
        annotations: 标注字典，格式为 {frameIdx: {"detailed": "...", "simple": "..."}}
        output_path: 输出视频路径
        frame_interval: 抽帧间隔（用于确定每个标注对应的帧范围）
        downscale_ratio: 下采样比例
        fps: 输出视频帧率
        font_scale: 字体大小
        font_thickness: 字体粗细
    """
    font_scale = 0.4
    font_thickness = 1
    out = []
    for frame_key, frame_resized in extracted_frames.items():

        # 创建黑色文字区域
        text_area = np.zeros_like(frame_resized)
        text_area = text_area.repeat(2, axis=1)
        if (
            f"{frame_key}"
            not in annotations["detailed"]
            # or f"{frame_key}" not in annotations["simple"]
        ):
            continue
        detailed_annotation = list(annotations["detailed"].get(f"{frame_key}").values())
        detailed_annotation = " [***** DYNAMIC ******] ".join(detailed_annotation)
        simple_annotation = ""  # annotations["simple"][f"{frame_key}"]["simple"]
        annotation = detailed_annotation  # + " [simple sep] " + simple_annotation

        # 使用 detailed 标注作为显示文本
        text = annotation
        new_w = text_area.shape[1]
        new_h = text_area.shape[0]
        # 将文本分行显示
        max_width = new_w - 10  # 留出边距
        words = text.split()
        words = [f"{frame_key}: "] + words  # 在开头加上帧索引
        lines = []
        current_line = ""

        for word in words:
            test_line = current_line + " " + word if current_line else word
            (text_width, text_height), _ = cv2.getTextSize(
                test_line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )

            if text_width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        # 在文字区域绘制文本
        y_offset = 10
        line_height = int(15 * font_scale / 0.4)  # 根据字体缩放调整行高

        for line in lines:
            if y_offset + line_height > new_h:
                break  # 超出文字区域

            cv2.putText(
                text_area,
                line,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA,
            )
            y_offset += line_height

        # 将帧和文字区域垂直拼接
        combined_frame = np.concatenate([frame_resized[..., ::-1], text_area], 1)

        # 写入输出视频
        out.append(combined_frame)
    vwrite(output_path, np.array(out))
    print(f"Annotated video saved to: {output_path}")


# === 新增：在一个进程里，用某一块 GPU 处理若干个视频 ===
def worker_process(
    gpu_index: int,
    video_paths: List[str],
    model_id: str,
    frame_interval: int,
    simple_block_size: int,
    downscale_ratio: float,
    out_dir: str,
    log_file_path: str,
    create_video: bool = False,
    detail_chunk: int = 3,
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
                out_json = out_dir_path / f"{v_path.stem}.json"
                if os.path.exists(out_json):
                    print(f"exist {out_json}, skipping")
                    continue
                summaries, extracted_frames = summarize_video_by_frames(
                    video_path=str(v_path),
                    model=model,
                    processor=processor,
                    frame_interval=frame_interval,
                    simple_block_size=simple_block_size,
                    downscale_ratio=downscale_ratio,
                    detail_chunk=detail_chunk,
                )

                with out_json.open("w", encoding="utf-8") as f:
                    json.dump(summaries, f, ensure_ascii=False, indent=2)
                print(
                    f"[Worker GPU {gpu_index}] Finished {v_path}, saved to {out_json}"
                )

                # 如果需要创建标注视频
                if True:
                    out_video = out_dir_path / f"{v_path.stem}_annotated.mp4"
                    print(
                        f"[Worker GPU {gpu_index}] Creating annotated video: {out_video}"
                    )
                    create_annotated_video(
                        extracted_frames=extracted_frames,
                        annotations=summaries,
                        output_path=out_video,
                    )
                    print(
                        f"[Worker GPU {gpu_index}] Annotated video saved to {out_video}"
                    )
            except Exception as e:
                print(f"[Worker GPU {gpu_index}] Error processing {v_path}: {e}")


# === 新增：多 GPU / 多进程调度逻辑 ===
def run_multi_gpu(
    input_dir: str,
    out_dir: str,
    model_id: str,
    frame_interval: int,
    simple_block_size: int,
    downscale_ratio: float,
    num_gpus: int | None = None,
    create_video: bool = False,
    detail_chunk: int = 3,
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
                frame_interval,
                simple_block_size,
                downscale_ratio,
                out_dir,
                str(log_file),
                create_video,
                detail_chunk,
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
            "本地 Qwen3-VL 对视频按帧间隔抽帧并标注：\n"
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
        "--frame_interval",
        type=int,
        default=8,
        help="抽帧间隔，例如 10 表示每隔 10 帧抽一帧（默认 10）",
    )
    parser.add_argument(
        "--detail_chunk",
        type=int,
        default=3,
        help="抽帧间隔，例如 10 表示每隔 10 帧抽一帧（默认 10）",
    )
    parser.add_argument(
        "--simple_block_size",
        type=int,
        default=30,
        help="简单标注的分组大小，例如 4 表示每 4 帧共享一个简单标注（默认 4）",
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
    parser.add_argument(
        "--create_video",
        action="store_true",
        help="是否创建带标注的可视化视频（默认不创建）",
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
            frame_interval=args.frame_interval,
            simple_block_size=args.simple_block_size,
            downscale_ratio=args.downscale_ratio,
            num_gpus=args.num_gpus,
            create_video=args.create_video,
            detail_chunk=args.detail_chunk,
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
    summaries, extracted_frames = summarize_video_by_frames(
        video_path=args.video_path,
        model=model,
        processor=processor,
        frame_interval=args.frame_interval,
        simple_block_size=args.simple_block_size,
        downscale_ratio=args.downscale_ratio,
        detail_chunk=args.detail_chunk,
    )

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    print(f"已保存结果到 {args.output_json}")

    # 如果需要创建标注视频
    if True:
        video_output_path = args.output_json.replace(".json", "_annotated.mp4")
        print(f"创建带标注的视频: {video_output_path}")
        create_annotated_video(
            extracted_frames=extracted_frames,
            annotations=summaries,
            output_path=video_output_path,
        )
        print(f"标注视频已保存到 {video_output_path}")


if __name__ == "__main__":
    # 在大多数环境下推荐 spawn，避免 CUDA 在父进程里初始化后被子进程复用出问题
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # 已经设置过 start method 的情况
        pass
    main()
