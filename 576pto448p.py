#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

VIDEO_EXTS = {".mp4"}  # 按你的例子只处理 mp4；需要更多格式就自行加进去


def build_jobs(folders_csv: str):
    folders = [s.strip() for s in folders_csv.split(",") if s.strip()]
    jobs = []
    out_folders = set()

    for f in folders:
        in_folder = Path(f)
        if not in_folder.exists() or not in_folder.is_dir():
            print(f"[WARN] Not a folder: {in_folder}", file=sys.stderr)
            continue

        name = in_folder.name
        if not name.endswith("576p"):
            print(
                f"[WARN] Folder does not end with 576p (skipping): {in_folder}",
                file=sys.stderr,
            )
            continue

        out_folder = in_folder.with_name(name[:-4] + "448p")  # 末尾 576p -> 448p
        out_folders.add(out_folder)

        for p in sorted(in_folder.iterdir()):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                out_path = out_folder / p.name
                jobs.append((p, out_path))

    return jobs, out_folders


def run_ffmpeg_job(
    in_path_str: str, out_path_str: str, overwrite: bool, crf: int, preset: str
):
    """
    A(1024x576) -> crop B(1024x552) centered (y=12) -> scale C(832x448)
    """
    in_path = Path(in_path_str)
    out_path = Path(out_path_str)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vf = "crop=1024:552:0:12,scale=832:448"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-stats",
        "-y" if overwrite else "-n",
        "-i",
        str(in_path),
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        str(out_path),
    ]

    try:
        subprocess.run(cmd, check=True)
        return (True, str(in_path), str(out_path), "")
    except subprocess.CalledProcessError as e:
        return (False, str(in_path), str(out_path), f"{e}")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Multi-process: crop 1024x576 -> 1024x552 (center) then resize -> 832x448 for mp4s in 576p folders."
    )
    ap.add_argument(
        "--folders",
        required=True,
        help='一个或多个文件夹（逗号分隔），这些文件夹名保证以576p结尾。例：--folders "A_576p,B_576p"',
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) // 2),
        help="并发进程数（默认：CPU核数的一半）。",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已有输出（ffmpeg -y）。默认跳过已有输出（ffmpeg -n）。",
    )
    ap.add_argument(
        "--crf",
        type=int,
        default=18,
        help="x264 CRF（默认18，越小越清晰/越大文件越大）。",
    )
    ap.add_argument(
        "--preset",
        default="medium",
        help="x264 preset（默认 medium；更快用 veryfast，更小文件/更慢用 slow）。",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    jobs, out_folders = build_jobs(args.folders)
    if not jobs:
        print("[INFO] No input videos found.", file=sys.stderr)
        return 0

    # 先确保输出目录存在（也会在 worker 内再 mkdir，一层保险）
    for of in out_folders:
        of.mkdir(parents=True, exist_ok=True)

    total = len(jobs)
    ok = 0
    fail = 0

    print(f"[INFO] Found {total} video(s). Using {args.workers} worker(s).")

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(
                run_ffmpeg_job,
                str(in_path),
                str(out_path),
                args.overwrite,
                args.crf,
                args.preset,
            )
            for (in_path, out_path) in jobs
        ]

        done_count = 0
        for fut in as_completed(futures):
            success, in_p, out_p, err = fut.result()
            done_count += 1
            if success:
                ok += 1
                print(f"[{done_count}/{total}] OK   {in_p} -> {out_p}")
            else:
                fail += 1
                print(
                    f"[{done_count}/{total}] FAIL {in_p} -> {out_p}\n       {err}",
                    file=sys.stderr,
                )

    print(f"[INFO] Done. OK={ok}, FAIL={fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
