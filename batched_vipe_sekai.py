import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from multiprocessing import Process, Queue


def list_mp4s(input_dir: Path, recursive: bool) -> list[Path]:
    if recursive:
        vids = sorted(input_dir.rglob("*.mp4"))
    else:
        vids = sorted(input_dir.glob("*.mp4"))
    return [p for p in vids if p.is_file()]


def run_one_video(
    run_py: Path,
    workdir: Path | None,
    gpu_id: str,
    video_path: Path,
    log_path: Path,
    pipeline_name: str,
    streams_name: str,
) -> int:
    """
    在指定 gpu_id 下跑一次命令，并把 stdout/stderr 全部写入 log_path
    """
    cmd = [
        sys.executable,  # 用当前 python 解释器跑 run.py
        str(run_py),
        f"pipeline={pipeline_name}",
        f"streams={streams_name}",
        f"streams.base_path={str(video_path)}",
        "pipeline.post.depth_align_model=null",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=" * 88 + "\n")
        f.write(f"[START] {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"[GPU]   CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}\n")
        f.write(f"[VIDEO] {video_path}\n")
        f.write(f"[CMD]   {' '.join(cmd)}\n")
        if workdir is not None:
            f.write(f"[CWD]   {workdir}\n")
        f.write("-" * 88 + "\n")
        f.flush()

        import subprocess

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(workdir) if workdir is not None else None,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=False,
            )
            rc = int(proc.returncode)
        except Exception:
            rc = 999
            f.write("\n[EXCEPTION]\n")
            f.write(traceback.format_exc())
            f.write("\n")

        t1 = time.time()
        f.write("-" * 88 + "\n")
        f.write(f"[END]   {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"[RC]    {rc}\n")
        f.write(f"[ELAPSED_SEC] {t1 - t0:.3f}\n")
        f.write("=" * 88 + "\n")

    return rc


def worker_loop(
    gpu_id: str,
    task_q: Queue,
    run_py: Path,
    workdir: Path | None,
    log_dir: Path,
    pipeline_name: str,
    streams_name: str,
    summary_path: Path,
):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as sf:
        sf.write("video_path,return_code,log_path\n")
        sf.flush()

        while True:
            item = task_q.get()
            if item is None:
                break
            video_path: Path = item

            # log 名称：保留相对路径信息，避免重名（转成 __ 分隔）
            # 例如 a/b/c.mp4 -> a__b__c_gpu0.log
            rel = video_path.name
            try:
                # 如果 video_path 在 workdir 或 input_dir 下，会更有区分度
                rel = str(video_path).replace("/", "__").replace("\\", "__")
            except Exception:
                pass

            log_path = log_dir / f"gpu{gpu_id}__{rel}.log"

            rc = run_one_video(
                run_py=run_py,
                workdir=workdir,
                gpu_id=gpu_id,
                video_path=video_path,
                log_path=log_path,
                pipeline_name=pipeline_name,
                streams_name=streams_name,
            )

            sf.write(f"{video_path},{rc},{log_path}\n")
            sf.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", "-i", type=str, required=True, help="包含 mp4 的输入文件夹"
    )
    parser.add_argument(
        "--gpus",
        "-g",
        type=str,
        required=True,
        help='使用哪些 GPU（物理编号），例如 "0,1,2,3"',
    )
    parser.add_argument(
        "--log_dir", "-l", type=str, required=True, help="log 输出文件夹"
    )
    parser.add_argument(
        "--run_py",
        "-r",
        type=str,
        default="run.py",
        help="run.py 的路径（默认 run.py）",
    )
    parser.add_argument(
        "--workdir",
        "-w",
        type=str,
        default=None,
        help="执行命令时的工作目录（比如你的工程根目录）；默认不指定",
    )
    parser.add_argument(
        "--recursive", "-R", action="store_true", help="递归搜索子目录下的 mp4"
    )
    parser.add_argument(
        "--pipeline",
        "-p",
        type=str,
        default="default",
        help="pipeline 名称（默认 default）",
    )
    parser.add_argument(
        "--streams",
        "-s",
        type=str,
        default="raw_mp4_stream",
        help="streams 名称（默认 raw_mp4_stream）",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    log_dir = Path(args.log_dir).expanduser().resolve()
    run_py = Path(args.run_py).expanduser().resolve()
    workdir = Path(args.workdir).expanduser().resolve() if args.workdir else None

    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")
    if not run_py.exists():
        raise FileNotFoundError(f"run.py not found: {run_py}")

    gpu_ids = [x.strip() for x in args.gpus.split(",") if x.strip() != ""]
    if len(gpu_ids) == 0:
        raise ValueError("No valid gpu ids from --gpus")

    mp4s = list_mp4s(input_dir, recursive=args.recursive)
    if len(mp4s) == 0:
        print(f"[WARN] No mp4 found in {input_dir} (recursive={args.recursive})")
        return

    log_dir.mkdir(parents=True, exist_ok=True)

    # 为每个 GPU 建一个队列 + 一个 worker
    queues: dict[str, Queue] = {gid: Queue(maxsize=128) for gid in gpu_ids}
    procs: list[Process] = []

    for gid in gpu_ids:
        summary_path = log_dir / f"summary_gpu{gid}.csv"
        p = Process(
            target=worker_loop,
            args=(
                gid,
                queues[gid],
                run_py,
                workdir,
                log_dir,
                args.pipeline,
                args.streams,
                summary_path,
            ),
            daemon=False,
        )
        p.start()
        procs.append(p)

    # Round-robin 分配任务到各 GPU
    for idx, vp in enumerate(mp4s):
        gid = gpu_ids[idx % len(gpu_ids)]
        queues[gid].put(vp)

    # 发送结束信号
    for gid in gpu_ids:
        queues[gid].put(None)

    # 等待全部结束
    for p in procs:
        p.join()

    print(f"[DONE] processed {len(mp4s)} videos. logs in: {log_dir}")


if __name__ == "__main__":
    main()
