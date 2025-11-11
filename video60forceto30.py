#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

# 可识别的视频扩展名
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".wmv", ".m4v", ".webm", ".flv", ".mts", ".m2ts", ".3gp"}

def sanitize_name(name: str) -> str:
    """
    删除文件名中的空格和“非安全字符”（只保留字母数字、下划线、连字符）。
    不保留原来的扩展点；扩展名统一由调用方决定（.mp4）。
    """
    # 去除空格
    name = name.replace(" ", "")
    # 只保留 [A-Za-z0-9_-]
    name = re.sub(r"[^A-Za-z0-9_-]", "", name)
    # 防止空名
    return name or "video"

def safe_output_path(out_dir: Path, stem: str, ext: str = ".mp4") -> Path:
    """
    避免重名：已存在则在末尾加 _1, _2, ...
    """
    candidate = out_dir / f"{stem}{ext}"
    i = 1
    while candidate.exists():
        candidate = out_dir / f"{stem}_{i}{ext}"
        i += 1
    return candidate

def run(cmd):
    """运行子进程并在失败时抛异常。"""
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDERR:\n{proc.stderr}")
    return proc.stdout

def get_avg_fps(path: Path) -> float | None:
    """
    用 ffprobe 读取平均帧率（avg_frame_rate）。
    返回 float，失败时返回 None。
    """
    try:
        out = run([
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate",
            "-of", "json",
            str(path)
        ])
        data = json.loads(out)
        r = data["streams"][0].get("avg_frame_rate", "0/0")
        num, den = r.split("/")
        num, den = int(num), int(den)
        if den == 0:
            return None
        return num / den
    except Exception:
        return None

def needs_reencode_to_30fps(fps: float | None) -> bool:
    """
    判断是否需要转为 30fps。
    允许一定误差（例如 29.97）。
    """
    if fps is None:
        return True
    return not (29.9 <= fps <= 30.1)

def transcode_to_30fps(src: Path, dst: Path, crf: int = 20, preset: str = "medium"):
    """
    使用 fps 滤镜把视频统一到 30fps；视频用 libx264 重编码，音频拷贝。
    通过 -movflags +faststart 优化网页播放起播。
    """
    # -y 覆盖输出（但我们上一步已经避免重名了）
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-vf", "fps=30",             # 统一帧率，保持时长
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", preset,
        "-crf", str(crf),
        "-c:a", "copy",              # 音频直接拷贝（避免质量损失与转码时间）
        "-movflags", "+faststart",
        str(dst)
    ]
    print(f"[FFmpeg] {src.name}  ->  {dst.name}")
    subprocess.check_call(cmd)

def maybe_copy(src: Path, dst: Path):
    """
    直接拷贝（已是 ~30fps 的情况）。也可以改成“重封装为 mp4”以统一容器。
    这里保持统一输出容器为 mp4，因此即使已是 30fps，我们也做一次“零变化”重封装：
      -c:v copy -c:a copy
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-c:v", "copy",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(dst)
    ]
    print(f"[Remux ] {src.name}  ->  {dst.name}")
    subprocess.check_call(cmd)

def main():
    parser = argparse.ArgumentParser(description="把文件夹下视频统一为 30fps（时长不变），保存到新文件夹并清理文件名。")
    parser.add_argument("input_dir", type=str, help="输入视频所在文件夹")
    parser.add_argument("output_dir", type=str, help="输出文件夹（将创建）")
    parser.add_argument("--crf", type=int, default=20, help="x264 CRF（质量参数，数值越大压缩越狠，默认20）")
    parser.add_argument("--preset", type=str, default="medium", help="x264 preset（编码速度/压缩比平衡，默认medium）")
    parser.add_argument("--force-reencode", action="store_true",
                        help="即使已是30fps也强制转码（默认遇到30fps仅重封装拷贝）")
    args = parser.parse_args()

    in_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()

    if not in_dir.exists() or not in_dir.is_dir():
        raise SystemExit(f"输入目录不存在：{in_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    videos = [p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    if not videos:
        print("未发现可处理的视频文件。")
        return

    for src in sorted(videos):
        # 生成清理后的输出文件名
        stem = sanitize_name(src.stem)
        dst = safe_output_path(out_dir, stem, ".mp4")

        # 判断是否需要转码
        fps = get_avg_fps(src)
        try:
            if args.force_reencode or needs_reencode_to_30fps(fps):
                transcode_to_30fps(src, dst, crf=args.crf, preset=args.preset)
            else:
                # 已是 ~30fps：仅重封装为 mp4，避免重编码
                maybe_copy(src, dst)
        except subprocess.CalledProcessError as e:
            print(f"处理失败：{src.name}\n{e}")
        except Exception as e:
            print(f"处理异常：{src.name}\n{e}")

    print("全部处理完成。输出目录：", out_dir)

if __name__ == "__main__":
    main()
