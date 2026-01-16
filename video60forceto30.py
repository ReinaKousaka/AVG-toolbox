#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any


# 可识别的视频扩展名
VIDEO_EXTS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".wmv",
    ".m4v",
    ".webm",
    ".flv",
    ".mts",
    ".m2ts",
    ".3gp",
    ".MOV",
}


def sanitize_name(name: str) -> str:
    """
    删除文件名中的空格和“非安全字符”（只保留字母数字、下划线、连字符）。
    不保留原来的扩展点；扩展名统一由调用方决定（.mp4）。
    """
    name = name.replace(" ", "")
    name = re.sub(r"[^A-Za-z0-9_-]", "", name)
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
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDERR:\n{proc.stderr}")
    return proc.stdout


def get_avg_fps(path: Path) -> Optional[float]:
    """
    用 ffprobe 读取平均帧率（avg_frame_rate）。
    返回 float，失败时返回 None。
    """
    try:
        out = run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=avg_frame_rate",
                "-of",
                "json",
                str(path),
            ]
        )
        data = json.loads(out)
        r = data["streams"][0].get("avg_frame_rate", "0/0")
        num, den = r.split("/")
        num, den = int(num), int(den)
        if den == 0:
            return None
        return num / den
    except Exception:
        return None


def needs_reencode_to_30fps(fps: Optional[float]) -> bool:
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
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vf",
        "fps=30",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    subprocess.check_call(cmd)


def maybe_copy(src: Path, dst: Path):
    """
    已是 ~30fps：仅重封装为 mp4，避免重编码。
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    subprocess.check_call(cmd)


def _setup_worker_logger(log_file: Path) -> logging.Logger:
    """
    每个 worker 独立写一个 log 文件，避免多进程抢同一文件句柄。
    """
    logger = logging.getLogger(f"worker.{os.getpid()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 防止重复添加 handler（某些平台/运行方式可能会重复初始化）
    if not logger.handlers:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_file), encoding="utf-8")
        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(process)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def process_one(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    单个视频处理任务（在 worker 进程里跑）。
    task 字段：
      - src: str
      - dst: str
      - crf: int
      - preset: str
      - force_reencode: bool
      - worker_log: str
    """
    src = Path(task["src"])
    dst = Path(task["dst"])
    crf = int(task["crf"])
    preset = str(task["preset"])
    force_reencode = bool(task["force_reencode"])
    worker_log = Path(task["worker_log"])

    logger = _setup_worker_logger(worker_log)

    t0 = datetime.now()
    result = {
        "src": str(src),
        "dst": str(dst),
        "ok": False,
        "mode": None,  # "transcode" or "remux"
        "fps": None,
        "error": None,
        "elapsed_sec": None,
    }

    try:
        fps = get_avg_fps(src)
        result["fps"] = fps

        # 保留原逻辑：force 或需要转码 -> transcode，否则 remux(copy)
        if force_reencode or needs_reencode_to_30fps(fps):
            logger.info(
                f"[FFmpeg] {src.name} -> {dst.name} | fps={fps} | crf={crf} preset={preset}"
            )
            transcode_to_30fps(src, dst, crf=crf, preset=preset)
            result["mode"] = "transcode"
        else:
            logger.info(f"[Remux ] {src.name} -> {dst.name} | fps={fps}")
            maybe_copy(src, dst)
            result["mode"] = "remux"

        result["ok"] = True
        return result

    except subprocess.CalledProcessError as e:
        msg = f"处理失败：{src.name}\n{e}"
        logger.error(msg)
        result["error"] = msg
        return result

    except Exception as e:
        tb = traceback.format_exc()
        msg = f"处理异常：{src.name}\n{e}\n{tb}"
        logger.error(msg)
        result["error"] = msg
        return result

    finally:
        dt = (datetime.now() - t0).total_seconds()
        result["elapsed_sec"] = dt


def _setup_main_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_file), encoding="utf-8")
        sh = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(fmt)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)

    return logger


def parse_args():
    p = argparse.ArgumentParser(
        description="把文件夹下视频统一为 30fps（时长不变），保存到新文件夹并清理文件名。支持多进程并行。"
    )
    # 保留你原来的两个 positional 参数
    p.add_argument("input_dir", type=str, help="输入视频所在文件夹")
    p.add_argument("output_dir", type=str, help="输出文件夹（将创建）")

    p.add_argument(
        "--crf",
        type=int,
        default=18,
        help="x264 CRF（质量参数，数值越大压缩越狠，默认18）",
    )
    p.add_argument(
        "--preset",
        type=str,
        default="medium",
        help="x264 preset（编码速度/压缩比平衡，默认medium）",
    )
    p.add_argument(
        "--force-reencode",
        action="store_true",
        help="即使已是30fps也强制转码（默认遇到30fps仅重封装拷贝）",
    )

    # 新增：多进程相关
    p.add_argument(
        "--workers",
        "-w",
        type=int,
        default=max(1, (os.cpu_count() or 8) // 2),
        help="并行进程数（默认 CPU 核数的一半）",
    )
    p.add_argument(
        "--log-dir",
        "-l",
        type=str,
        default=None,
        help="日志目录（默认在 output_dir 下创建 logs_时间戳）",
    )
    return p.parse_args()


def main():
    args = parse_args()

    in_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()

    if not in_dir.exists() or not in_dir.is_dir():
        raise SystemExit(f"输入目录不存在：{in_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # 日志目录
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = (
        Path(args.log_dir).expanduser().resolve()
        if args.log_dir
        else (out_dir / f"logs_{ts}")
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    main_log = log_dir / "main.log"
    logger = _setup_main_logger(main_log)

    videos = [
        p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ]
    if not videos:
        logger.info("未发现可处理的视频文件。")
        return

    # 为并行安全：主进程提前分配每个 src 的唯一 dst（严格复用你的 sanitize + safe_output_path 逻辑）
    tasks = []
    for idx, src in enumerate(sorted(videos)):
        stem = sanitize_name(src.stem)
        dst = safe_output_path(out_dir, stem, ".mp4")

        worker_log = log_dir / f"worker_{idx:05d}_{stem}.log"
        tasks.append(
            {
                "src": str(src),
                "dst": str(dst),
                "crf": args.crf,
                "preset": args.preset,
                "force_reencode": args.force_reencode,
                "worker_log": str(worker_log),
            }
        )

    total = len(tasks)
    logger.info(
        f"发现 {total} 个视频，开始处理。workers={args.workers} | in={in_dir} | out={out_dir}"
    )
    logger.info(f"日志目录：{log_dir}")

    ok_cnt = 0
    fail_cnt = 0

    # 并行处理
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_one, t) for t in tasks]

        done = 0
        for fut in as_completed(futures):
            done += 1
            res = fut.result()

            src = Path(res["src"]).name
            dst = Path(res["dst"]).name
            mode = res["mode"]
            fps = res["fps"]
            elapsed = res["elapsed_sec"]

            if res["ok"]:
                ok_cnt += 1
                logger.info(
                    f"[{done:>4}/{total}] OK   | {mode:9s} | fps={fps} | {src} -> {dst} | {elapsed:.1f}s"
                )
            else:
                fail_cnt += 1
                logger.error(
                    f"[{done:>4}/{total}] FAIL | {src} -> {dst} | {elapsed:.1f}s"
                )
                if res.get("error"):
                    # 错误详情写到总 log（worker log 里也有）
                    logger.error(res["error"])

    logger.info(f"全部处理完成。成功={ok_cnt} 失败={fail_cnt} 输出目录：{out_dir}")
    logger.info(f"总日志：{main_log}")


if __name__ == "__main__":
    main()
