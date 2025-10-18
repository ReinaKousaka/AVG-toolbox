#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量视频处理（基于 ffmpeg）：
0) 丢弃前后 drop_seconds
1) Resize 到 --resize W x H
2) 抽帧：每 sample_ratio 取 1 帧
3) 中心裁剪到 --crop W x H
4) 按抽帧后的帧序列，每 interval_frames 帧切片（尾段不足丢弃）
输出：H.264 (libx264) + yuv420p + faststart，固定 30 fps，默认无音频（-an）
"""

import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
import argparse
from typing import Optional, Tuple

FFMPEG = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
FFPROBE = "ffprobe.exe" if os.name == "nt" else "ffprobe"


def check_ffmpeg():
    for exe in (FFMPEG, FFPROBE):
        try:
            subprocess.run(
                [exe, "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
        except Exception:
            raise RuntimeError(f"未检测到 {exe}，请先安装 ffmpeg/ffprobe 并加入 PATH。")


def run_cmd(cmd: list, cwd: Optional[str] = None):
    # 更友好的错误抛出
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(f"命令执行失败：\n{' '.join(cmd)}\n\nstderr:\n{proc.stderr}")
    return proc.stdout.strip()


def ffprobe_duration_seconds(video_path: Path) -> float:
    # 用 ffprobe 获取时长（秒）
    cmd = [
        FFPROBE,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    out = run_cmd(cmd)
    data = json.loads(out)
    dur = float(data["format"]["duration"])
    return dur


def ffprobe_read_frames(video_path: Path) -> int:
    # 读取精确帧数（对我们输出的 CFR 30fps 文件通常可靠）
    cmd = [
        FFPROBE,
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "json",
        str(video_path),
    ]
    out = run_cmd(cmd)
    data = json.loads(out)
    nb = data["streams"][0].get("nb_read_frames")
    if nb is None or nb == "N/A":
        # 回退：以时长*30 估算（仅在极端情况下）
        dur = ffprobe_duration_seconds(video_path)
        return int(math.floor(dur * 30.0 + 1e-6))
    return int(nb)


@dataclass
class Settings:
    input_dir: Path
    output_dir: Path
    drop_seconds: float
    resize_wh: Tuple[int, int]
    sample_ratio: int
    crop_wh: Tuple[int, int]
    interval_frames: int
    crf: int
    preset: str
    keep_temp: bool


def parse_size(s: str) -> Tuple[int, int]:
    try:
        w, h = s.lower().split("x")
        w = int(w)
        h = int(h)
        if w <= 0 or h <= 0:
            raise ValueError
        return w, h
    except Exception:
        raise argparse.ArgumentTypeError("尺寸格式应为 WxH，例如 1280x720")


def build_filter_chain(settings: Settings) -> str:
    # 顺序：scale -> select -> crop
    rw, rh = settings.resize_wh
    cw, ch = settings.crop_wh
    r = settings.sample_ratio

    # scale：无论原分辨率是什么，统一到 rw x rh
    scale = f"scale={rw}:{rh}"

    # select 抽帧：每 r 帧取 1 帧；r=1 时不抽帧（但我们仍用表达式，统一处理）
    # 注意转义逗号：mod(n\,r)
    select = f"select='not(mod(n\\,{r}))'"

    # 中心裁剪（基于 resize 后的帧）
    # 这里用 in_w/in_h（即当前链中上一环节输出大小），居中裁。
    crop = f"crop={cw}:{ch}:(in_w-{cw})/2:(in_h-{ch})/2"

    # 拼接
    vf = ",".join([scale, select, crop])
    return vf


def make_temp_processed(
    input_path: Path, temp_path: Path, settings: Settings, effective_duration: float
):
    # 先做 0/1/2/3 步并固化为中间文件（30fps + H.264）
    # -ss 放前面，再 -t 指定有效时长，确保先丢掉前后 x 秒（后 x 秒通过 t=dur-2x 实现）
    vf = build_filter_chain(settings)

    cmd = [
        FFMPEG,
        "-y",
        "-ss",
        f"{settings.drop_seconds:.3f}",
        "-i",
        str(input_path),
        "-t",
        f"{effective_duration:.3f}",
        "-vf",
        vf,
        "-r",
        "30",  # 固定输出 30 fps
        "-an",  # 丢弃音频，避免切片时音视频对齐问题
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "high",
        "-movflags",
        "+faststart",
        "-preset",
        settings.preset,
        "-crf",
        str(settings.crf),
        str(temp_path),
    ]
    run_cmd(cmd)


def slice_by_frames(
    temp_path: Path, out_dir: Path, interval_frames: int, crf: int, preset: str
):
    total = ffprobe_read_frames(temp_path)
    full_segments = total // interval_frames  # 尾段不足直接丢弃

    made = 0
    for idx in range(full_segments):
        start = idx * interval_frames
        end = (
            idx + 1
        ) * interval_frames  # trim end_frame 为开区间还是闭区间？设为开区间更稳妥
        # 用 trim 的 start_frame/end_frame 然后重置时间戳
        vf = f"trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS"

        out_name = temp_path.stem + f"_part_{idx:03d}.mp4"
        out_path = out_dir / out_name

        cmd = [
            FFMPEG,
            "-y",
            "-i",
            str(temp_path),
            "-vf",
            vf,
            "-r",
            "30",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-profile:v",
            "high",
            "-movflags",
            "+faststart",
            "-preset",
            preset,
            "-crf",
            str(crf),
            str(out_path),
        ]
        run_cmd(cmd)
        made += 1

    return made, total, full_segments


def process_one_video(path: Path, settings: Settings):
    # 仅处理 .mp4
    if path.suffix.lower() != ".mp4":
        return

    print(f"\n处理：{path.name}")
    try:
        dur = ffprobe_duration_seconds(path)
    except Exception as e:
        print(f"  跳过（无法读取时长）：{e}")
        return

    # 计算有效时长（先丢前后 drop_seconds）
    effective_duration = max(0.0, dur - settings.drop_seconds * 2.0)
    if effective_duration <= 0.0:
        print("  跳过：丢弃前后时长后没有有效内容。")
        return

    # 安全检查：裁剪尺寸不得大于 resize 后尺寸
    rw, rh = settings.resize_wh
    cw, ch = settings.crop_wh
    if cw > rw or ch > rh:
        print(f"  跳过：裁剪尺寸 {cw}x{ch} 大于 resize 后尺寸 {rw}x{rh}")
        return

    # 生成 temp
    temp_dir = settings.output_dir / "_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / (path.stem + "_proc_temp.mp4")

    try:
        make_temp_processed(path, temp_path, settings, effective_duration)
        made, total_frames, full_segments = slice_by_frames(
            temp_path,
            settings.output_dir,
            settings.interval_frames,
            settings.crf,
            settings.preset,
        )
        print(
            f"  抽帧后总帧数：{total_frames}，切出 {full_segments} 段（每段 {settings.interval_frames} 帧），实际生成：{made} 个切片。"
        )
    except Exception as e:
        print(f"  失败：{e}")
    finally:
        if not settings.keep_temp and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


def collect_videos(input_dir: Path):
    return sorted([p for p in input_dir.glob("**/*.mp4")])


def main():
    parser = argparse.ArgumentParser(description="批量 mp4 帧处理 + 切片（ffmpeg）")
    parser.add_argument(
        "--input_dir", type=Path, required=True, help="输入 mp4 文件夹（会递归扫描）"
    )
    parser.add_argument("--output_dir", type=Path, required=True, help="输出文件夹")
    parser.add_argument(
        "--drop_seconds",
        type=float,
        default=0.0,
        help="在一切开始前丢弃的前/后秒数（两头各丢弃）",
    )
    parser.add_argument(
        "--resize",
        type=parse_size,
        required=True,
        help="resize 到的分辨率，格式：WxH，例如 1280x720",
    )
    parser.add_argument(
        "--sample_ratio",
        type=int,
        default=1,
        help="抽帧比：每 N 帧取 1 帧（2 表示一隔一）",
    )
    parser.add_argument(
        "--crop",
        type=parse_size,
        required=True,
        help="中心裁剪尺寸，格式：WxH，例如 1024x576",
    )
    parser.add_argument(
        "--interval_frames",
        type=int,
        required=True,
        help="切片间隔（抽帧后，每段的帧数）",
    )
    parser.add_argument(
        "--crf", type=int, default=18, help="x264 画质（0-51，越小越清晰，常用 17-23）"
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="medium",
        help="x264 编码速度预设（ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow）",
    )
    parser.add_argument(
        "--keep_temp", type=str, default="false", help="是否保留临时文件（true/false）"
    )
    args = parser.parse_args()

    keep_temp = str(args.keep_temp).lower() in ("1", "true", "yes", "y")

    settings = Settings(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        drop_seconds=max(0.0, args.drop_seconds),
        resize_wh=args.resize,
        sample_ratio=max(1, args.sample_ratio),
        crop_wh=args.crop,
        interval_frames=max(1, args.interval_frames),
        crf=max(0, min(51, args.crf)),
        preset=args.preset,
        keep_temp=keep_temp,
    )

    check_ffmpeg()
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    videos = collect_videos(settings.input_dir)
    if not videos:
        print("未在输入目录下发现 mp4 文件。")
        return

    print(f"共发现 {len(videos)} 个 mp4，开始处理……")
    for v in videos:
        process_one_video(v, settings)

    # 清理空的 _temp
    temp_dir = settings.output_dir / "_temp"
    if temp_dir.exists() and not any(temp_dir.iterdir()):
        try:
            temp_dir.rmdir()
        except Exception:
            pass

    print("\n全部完成。")


if __name__ == "__main__":
    main()
