import argparse
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


def run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed ({p.returncode}): {' '.join(cmd)}\n"
            f"--- stderr ---\n{p.stderr.strip()}\n"
            f"--- stdout ---\n{p.stdout.strip()}"
        )
    return p.stdout


def ffprobe_duration_sec(video_path: str) -> float:
    out = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            video_path,
        ]
    )
    j = json.loads(out)
    return float(j["format"]["duration"])


def compute_trim_window(
    duration: float, target_middle: float = 60.0
) -> tuple[float, float, float]:
    """
    返回 (start_sec, trim_sec, pad_sec)

    1) 若 duration >= 80：在 [10, duration-10] 内居中取 60s（符合“去头10s去尾10s”意图）
    2) 若 60 <= duration < 80：无法两头各去10s，则全视频内居中取 60s
    3) 若 duration < 60：取全视频并 pad 到 60s（补最后一帧）
    """
    if duration >= target_middle + 20.0:
        avail = duration - 20.0
        start = 10.0 + max(0.0, (avail - target_middle) / 2.0)
        trim = target_middle
        pad = 0.0
        return start, trim, pad

    if duration >= target_middle:
        start = max(0.0, (duration - target_middle) / 2.0)
        trim = target_middle
        pad = 0.0
        return start, trim, pad

    start = 0.0
    trim = duration
    pad = target_middle - duration
    return start, trim, pad


def build_ffmpeg_cmd(
    inp: str,
    outp: str,
    start: float,
    trim: float,
    pad: float,
    crf: int,
    preset: str,
    width: int | None,
    height: int | None,
) -> list[str]:
    pad_sec = max(0.0, pad)
    tpad_stop = max(2.0, pad_sec + 2.0)

    vf_parts = [
        f"trim=start={start:.6f}:duration={trim:.6f}",
        "setpts=PTS-STARTPTS",
        f"tpad=stop_mode=clone:stop_duration={tpad_stop:.6f}",
        "fps=30:round=near",
        "trim=duration=60",
        "setpts=PTS-STARTPTS",
    ]

    if width is not None and height is not None:
        vf_parts.append(f"scale={width}:{height}")
    else:
        vf_parts.append("scale=trunc(iw/2)*2:trunc(ih/2)*2")

    vf = ",".join(vf_parts)

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        inp,
        "-an",
        "-vf",
        vf,
        "-vsync",
        "1",
        "-r",
        "30",
        "-frames:v",
        "1800",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-movflags",
        "+faststart",
        outp,
    ]
    return cmd


def process_one(
    video_path: str,
    out_dir: str,
    crf: int,
    preset: str,
    overwrite: bool,
    width: int | None,
    height: int | None,
) -> tuple[str, str]:
    in_p = Path(video_path)
    out_p = Path(out_dir) / in_p.name

    if out_p.exists() and not overwrite:
        return str(in_p), "skip"

    out_p.parent.mkdir(parents=True, exist_ok=True)

    dur = ffprobe_duration_sec(str(in_p))
    start, trim, pad = compute_trim_window(dur, target_middle=60.0)
    cmd = build_ffmpeg_cmd(
        str(in_p),
        str(out_p),
        start,
        trim,
        pad,
        crf=crf,
        preset=preset,
        width=width,
        height=height,
    )
    run(cmd)
    return str(in_p), "ok"


def list_mp4s(input_dir: str) -> list[str]:
    p = Path(input_dir)
    if not p.is_dir():
        raise ValueError(f"Not a directory: {input_dir}")
    return [str(x) for x in sorted(p.glob("*.mp4"))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", "-i", type=str, required=True, help="包含 mp4 的输入文件夹"
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default=None,
        help="输出文件夹（默认：<input_dir>_60s30fps）",
    )
    parser.add_argument(
        "--jobs", "-j", type=int, default=max(1, os.cpu_count() or 1), help="并行进程数"
    )
    parser.add_argument(
        "--crf", "-c", type=int, default=18, help="x264 CRF（越小质量越高，文件越大）"
    )
    parser.add_argument(
        "--preset",
        "-p",
        type=str,
        default="medium",
        help="x264 preset：ultrafast~veryslow",
    )
    parser.add_argument(
        "--overwrite", "-w", action="store_true", help="覆盖已存在输出文件"
    )

    parser.add_argument(
        "--width", "-W", type=int, default=1024, help="输出宽（默认 1024）"
    )
    parser.add_argument(
        "--height", "-H", type=int, default=576, help="输出高（默认 576）"
    )
    parser.add_argument(
        "--no_resize",
        "-n",
        action="store_true",
        help="不做固定缩放（保持原分辨率，只保证偶数）",
    )

    # 进度条显示更清晰
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="不逐个打印每个视频状态（仅显示进度条）",
    )

    args = parser.parse_args()

    in_dir = args.input_dir
    out_dir = args.output_dir or (str(Path(in_dir).resolve()) + "_60s30fps")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    width = None if args.no_resize else args.width
    height = None if args.no_resize else args.height

    videos = list_mp4s(in_dir)
    if not videos:
        print(f"No mp4 found in: {in_dir}")
        return

    ok = fail = skip = 0
    errors: list[str] = []

    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = [
            ex.submit(
                process_one,
                vp,
                out_dir,
                args.crf,
                args.preset,
                args.overwrite,
                width,
                height,
            )
            for vp in videos
        ]

        with tqdm(total=len(futs), desc="Processing videos", unit="video") as pbar:
            for fut in as_completed(futs):
                try:
                    src, status = fut.result()
                    if status == "ok":
                        ok += 1
                        if not args.quiet:
                            tqdm.write(f"[ok]   {src}")
                    elif status == "skip":
                        skip += 1
                        if not args.quiet:
                            tqdm.write(f"[skip] {src}")
                except Exception as e:
                    fail += 1
                    errors.append(str(e))
                    if not args.quiet:
                        tqdm.write(f"[fail] {e}")
                finally:
                    pbar.update(1)

    print(f"\nDone. ok={ok}, skip={skip}, fail={fail}")
    if errors:
        print("\nErrors (first 5):")
        for i, err in enumerate(errors[:5], 1):
            print(f"{i}. {err}")


if __name__ == "__main__":
    main()
