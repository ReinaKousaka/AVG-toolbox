import os, glob
import time
import subprocess
from collections import defaultdict
from multiprocessing import Pool
from tqdm import tqdm
from func_timeout import func_timeout, FunctionTimedOut, func_set_timeout
import json

INPUT_FOLDER = "sekai-real-walking-cam"
OUTPUT_DIR = "sekai-real-walking-empty-sunset"
TMP_DIR = os.path.join(OUTPUT_DIR, "_tmp_videos")
from pathlib import Path

# 并行参数：进程数 & 启动间隔（秒）
NUM_WORKERS = 8
START_STAGGER_SECONDS = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)


def load_entries(dir_path: str) -> list[str]:
    dir_path = Path(dir_path)
    jsons = glob.glob(str(dir_path / "*.json"))
    files = {}
    if os.path.exists("file_counts.json"):
        with open("file_counts.json", "r", encoding="utf-8") as f:
            files = json.load(f)
    else:
        for js in tqdm(jsons):
            with open(js, "r", encoding="utf-8") as f:
                data = json.load(f)
                files[js] = len(list(data.keys()))
    with open("file_counts.json", "w", encoding="utf-8") as f:
        json.dump(files, f, indent=2, ensure_ascii=False)
    # 按文件内条目数降序排序
    files_sorted = sorted(files.keys(), key=lambda x: files[x], reverse=True)
    files_sorted = [os.path.basename(f).split(".")[0] for f in files_sorted]
    return files_sorted


def group_by_ytid(entries: list[str]) -> dict[str, list[tuple[int, int]]]:
    """
    把 xxx_0000123_0000456 这种名字按 ytid 分组，并把帧号乘 2（保持你原本的逻辑）。
    """
    groups = defaultdict(list)
    for name in entries:
        try:
            ytid, s, e = name.rsplit("_", 2)
        except ValueError:
            raise ValueError(f"Filename cannot be parsed using rsplit('_',2): '{name}'")

        if not s.isdigit() or not e.isdigit():
            raise ValueError(f"Frame indices are not numeric in filename: '{name}'")

        start_f, end_f = int(s) * 2, int(e) * 2
        if end_f <= start_f or start_f < 0:
            raise ValueError(f"Invalid frame range in filename: {name}")

        groups[ytid].append((start_f, end_f))

    # 每个视频内部按起始帧排序，便于调试/复现
    for ytid in groups:
        groups[ytid].sort(key=lambda x: (x[0], x[1]))
    return dict(groups)


def segment_out_path(output_dir: str, ytid: str, start_f: int, end_f: int) -> str:
    # 输出文件名仍然用 (start_f//2, end_f//2) 的 7 位补零格式（和你原切片阶段一致）
    return os.path.join(
        output_dir,
        f"{ytid}_{str(start_f // 2).zfill(7)}_{str(end_f // 2).zfill(7)}.mp4",
    )


def get_missing_segments(
    output_dir: str, ytid: str, segments: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    missing = []
    for start_f, end_f in segments:
        out_file = segment_out_path(output_dir, ytid, start_f, end_f)
        if not os.path.exists(out_file):
            missing.append((start_f, end_f))
    return missing


def download_video(url: str, tmp_video_path: str) -> bool:
    """
    下载单个原视频到 tmp_video_path。成功返回 True，否则 False。
    """
    if os.path.exists(tmp_video_path):
        os.remove(tmp_video_path)

    try:
        subprocess.run(
            [
                "yt-dlp",
                "-vU",
                "-f",
                "bestvideo[ext=mp4]/best[ext=mp4]",
                "--fixup",
                "warn",
                "--no-warnings",
                "--quiet",
                "--cookies",
                "2.txt",
                "-t",
                "sleep",
                "-o",
                tmp_video_path,  # 这里直接输出到唯一 tmp 文件名
                url,
            ],
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        print(f"[ERROR] failed to download {url}")
        return False


# import subprocess
from fractions import Fraction


def get_fps_ffprobe(video_path: str) -> float:
    """
    用 ffprobe 读取视频 v:0 的 avg_frame_rate（通常 CFR 时可靠）。
    返回 fps(float)。失败则回退到 30.0
    """
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        out = subprocess.check_output(cmd).decode().strip()
        fps = float(Fraction(out))
        if fps > 0:
            return fps
    except Exception:
        pass
    print(f"[WARN] get_fps_ffprobe failed for {video_path}, use default 60.0")
    return 60.0


import csv


def norm(s: str) -> str:
    """Normalize strings: trim, lowercase, convert fullwidth punctuation, etc."""
    if s is None:
        return ""
    s = str(s).strip().lower()
    # common fullwidth punctuation / stray chars
    s = s.replace("，", ",").replace("。", ".")
    return s


def load_labels(csv_path: Path) -> dict:
    """
    Returns mapping: videoFile -> (crowdDensity, timeOfDay)
    """
    mapping = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"videoFile", "crowdDensity", "timeOfDay"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV 缺少列: {missing}. 现有列: {reader.fieldnames}")

        for row in reader:
            vf = (row.get("videoFile") or "").strip()
            if not vf:
                continue
            crowd = norm(row.get("crowdDensity"))
            tod = norm(row.get("timeOfDay"))
            caption = norm(row.get("caption"))
            # Some datasets may have trailing commas / extra words: take first token
            crowd = crowd.split(",")[0].strip()
            tod = tod.split(",")[0].strip()

            mapping[vf] = (crowd, tod, caption)
    return mapping


@func_set_timeout(1000)
def extract_segment(
    tmp_video_path: str, start_f: int, end_f: int, out_file: str
) -> None:
    """
    方法3（加强版）：先 -ss 到接近 start 的位置，再在小窗口内用 select=between(n,...) 按帧精确裁剪。
    - CFR（稳定帧率）时效果最好
    - end_f 是闭区间（包含 end_f）
    """
    fps = get_fps_ffprobe(tmp_video_path)

    # 帧号 -> 秒
    print(fps)
    start_sec = start_f / 60.0
    end_sec = end_f / 60.0

    # 给一个“余量窗口”，先快进到 start_sec - margin（避免从头解码到很后面）
    # 余量建议 1~5 秒；你可以按视频 GOP 情况调大一点
    margin_sec = 10.0
    start_sec -= margin_sec
    end_sec += margin_sec
    # coarse_sec = max(0.0, start_sec - margin_sec)
    # fine_sec = start_sec - coarse_sec  # 进入小窗口后，再精确跳到 start

    # # 在精确跳到 start 后，局部帧号从 0 开始
    # # between(n, 0, L) 是闭区间，所以 L = end_f-start_f
    # local_end = max(0, end_f - start_f)

    # vf = f"select='between(n\\,0\\,{local_end})',setpts=PTS-STARTPTS"
    subprocess.run(
        [
            "ffmpeg",
            "-ss",
            f"{start_sec}",
            "-to",
            f"{end_sec}",
            "-i",
            f"{tmp_video_path}",
            "-c",
            "copy",
            out_file,
        ],
        check=True,
    )


def process_one_video_task(args) -> tuple[str, str]:
    """
    一个“任务”= 一个原视频 + 与它相关的所有切片。
    返回 (ytid, status) 供主进程汇总。
    """
    ytid, segments, output_dir, tmp_dir = args

    # 先判断缺哪些切片；若都存在则跳过（避免无意义下载）
    missing = get_missing_segments(output_dir, ytid, segments)
    if not missing:
        print(f"[INFO] All segments exist for {ytid}, skip download.")
        return (ytid, "skip_all_exist")

    url = f"https://www.youtube.com/watch?v={ytid}"
    tmp_video_path = os.path.join(
        tmp_dir, f"tmp_{ytid}_{os.getpid()}.mp4"
    )  # 唯一 tmp 名

    print(f"\n=== Processing video: {url} | missing segments: {len(missing)} ===")
    if len(glob.glob(os.path.join(output_dir, ytid + "*.mp4"))) > 0:
        print(f"[WARN] Detected existing segments for {ytid} in output dir.")
        return (ytid, "skip_all_exist")
    ok = download_video(url, tmp_video_path)
    if not ok:
        # 下载失败直接返回；不影响其它视频任务
        return (ytid, "download_failed")
    else:
        print(f"[OK] Downloaded video to {tmp_video_path}")

    # 逐段切片（同一个任务/进程内串行；但不同视频间并行）
    for start_f, end_f in missing:
        out_file = segment_out_path(output_dir, ytid, start_f, end_f)
        if os.path.exists(out_file):
            continue
        try:
            start_time = time.time()
            extract_segment(tmp_video_path, start_f, end_f, out_file)
            use_time = time.time() - start_time
            print(f"[OK] Saved snippet: {out_file}, use_time={use_time:.1f}s")
        except Exception as e:
            with open("ffmpeg_error.log", "a") as f:
                f.write(f"{ytid} {start_f}->{end_f} error: {str(e)}\n")
            print(
                f"[ERROR] ffmpeg failed for {ytid} {start_f}->{end_f}, continue...{e}"
            )

    # 清理 tmp
    if os.path.exists(tmp_video_path):
        os.remove(tmp_video_path)

    print(f"Finished {url}, removed temp video.\n")
    return (ytid, "done")


def load_labels(csv_path: Path) -> dict:
    """
    Returns mapping: videoFile -> (crowdDensity, timeOfDay)
    """
    mapping = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"videoFile", "crowdDensity", "timeOfDay"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV 缺少列: {missing}. 现有列: {reader.fieldnames}")

        for row in reader:
            vf = (row.get("videoFile") or "").strip()
            if not vf:
                continue
            crowd = norm(row.get("crowdDensity"))
            tod = norm(row.get("timeOfDay"))

            # Some datasets may have trailing commas / extra words: take first token
            crowd = crowd.split(",")[0].strip()
            tod = tod.split(",")[0].strip()

            mapping[vf] = (crowd, tod)
    return mapping


NUM_TBD = 800


def main():
    entries = load_entries(INPUT_FOLDER)
    labels = load_labels(Path("sekai-real-walking.csv"))
    FILTERED_ENTRIES = []
    for entry in entries:
        if os.path.exists(os.path.join("V_sekai_ed1", f"{entry}.mp4")):
            continue
        if f"{entry}.mp4" in labels:
            crowd, tod = labels[f"{entry}.mp4"]
            if crowd == "empty" and tod == "sunset":
                FILTERED_ENTRIES.append(entry)
        if len(FILTERED_ENTRIES) >= NUM_TBD:
            break
    groups = group_by_ytid(FILTERED_ENTRIES)

    # 只把“确实缺切片”的视频放进任务队列
    tasks = []
    for ytid, segments in groups.items():
        if get_missing_segments(OUTPUT_DIR, ytid, segments):
            tasks.append((ytid, segments, OUTPUT_DIR, TMP_DIR))

    if not tasks:
        print("Nothing to do: all snippets exist.")
        return

    print(
        f"Total videos to process: {len(tasks)} | workers={NUM_WORKERS} | stagger={START_STAGGER_SECONDS}s"
    )

    # 进程池并行：每提交一个任务就 sleep(10)，实现“下载进程启动间隔 10s”
    # （用 Pool 并行属于进程级并行）:contentReference[oaicite:3]{index=3}
    results = []
    with Pool(processes=NUM_WORKERS) as pool:
        for task in tasks:
            results.append(pool.apply_async(process_one_video_task, (task,)))
            # if not (status == "skip_all_exist"):
            time.sleep(START_STAGGER_SECONDS)

        # 等待完成并显示进度
        for r in tqdm(results, desc="videos"):
            _ytid, status = r.get()
            # 你需要的话可以在这里统计各 status 数量

    print("All tasks finished.")


if __name__ == "__main__":
    main()
