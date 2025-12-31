#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import shlex
import sys
import time
from pathlib import Path
from multiprocessing import Process, JoinableQueue, Queue
from typing import Any, Dict, List, Optional, Tuple
import subprocess
import sys

sys.path.append(os.path.join(os.getcwd(), "video-depth-anything"))


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def load_sorted_items(sorted_json: Path) -> List[Dict[str, Any]]:
    with sorted_json.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if (
        not isinstance(obj, dict)
        or "sorted" not in obj
        or not isinstance(obj["sorted"], list)
    ):
        raise ValueError("sorted_json must contain top-level key 'sorted' as a list")
    return obj["sorted"]


def parse_gpus(gpus_str: str) -> List[str]:
    s = gpus_str.strip().lower()
    if s == "all":
        # 尽量自动推断
        # 1) 优先用 CUDA_VISIBLE_DEVICES
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if cvd:
            # 可能是 "0,1,2" 或者包含空格
            parts = [p.strip() for p in cvd.split(",") if p.strip() != ""]
            # 注意：CUDA_VISIBLE_DEVICES 可能是 UUID；这里直接原样用
            return parts
        # 2) 再尝试 torch
        try:
            import torch  # type: ignore

            n = torch.cuda.device_count()
            return [str(i) for i in range(n)]
        except Exception:
            raise RuntimeError(
                "gpus=all 但无法推断 GPU 列表；请显式传入 --gpus 例如 0,1,2,3"
            )
    # 普通情况：逗号分隔
    parts = [p.strip() for p in gpus_str.split(",") if p.strip() != ""]
    if not parts:
        raise ValueError("Empty --gpus")
    return parts


def build_tasks(
    items: List[Dict[str, Any]],
    top_n: int,
    start_idx: int,
    output_root: Path,
    skip_existing: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    tasks: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    # items 已经是按分数从高到低排好
    filtered: List[Dict[str, Any]] = []
    for it in items:
        score = it.get("score", None)
        file_path = it.get("file", None)
        if not isinstance(file_path, str):
            continue
        if not _is_num(score):
            continue
        filtered.append(it)

    sliced = filtered[start_idx:]
    if top_n > 0:
        sliced = sliced[:top_n]

    for rank, it in enumerate(sliced):
        json_path = Path(it["file"])
        mp4_path = json_path.with_suffix(".mp4")

        out_dir = output_root
        log_path = output_root / "_logs" / f"{rank:05d}_{mp4_path.stem}.log"

        # if skip_existing and out_dir.exists():
        #     # 只要目录存在且非空就认为处理过
        #     try:
        #         has_any = any(out_dir.iterdir())
        #     except Exception:
        #         has_any = True
        #     if has_any:
        #         skipped.append(
        #             {
        #                 "rank": rank,
        #                 "input_json": str(json_path),
        #                 "input_video": str(mp4_path),
        #                 "output_dir": str(out_dir),
        #                 "score": it.get("score"),
        #                 "reason": "skip_existing",
        #             }
        #         )
        #         continue

        tasks.append(
            {
                "rank": rank,
                "input_json": str(json_path),
                "input_video": str(mp4_path),
                "output_dir": str(out_dir),
                "log_path": str(log_path),
                "score": it.get("score"),
            }
        )

    return tasks, skipped


def worker_loop(
    worker_id: int,
    gpu_id: str,
    task_q: JoinableQueue,
    result_q: Queue,
    python_bin: str,
    run_py: str,
    encoder: str,
    metric: bool,
    extra_args: List[str],
    dry_run: bool,
) -> None:
    while True:
        task = task_q.get()
        if task is None:
            task_q.task_done()
            break

        t0 = time.time()
        in_video = task["input_video"]
        out_dir = task["output_dir"]
        log_path = task["log_path"]

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            python_bin,
            run_py,
            "--input_video",
            in_video,
            "--output_dir",
            out_dir,
            "--encoder",
            encoder,
        ]
        if metric:
            cmd.append("--metric")
        if extra_args:
            cmd.extend(extra_args)

        env = os.environ.copy()
        # 关键：把任务“绑”到某个 GPU（脚本里看到的是 device 0）
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        rc = 0
        err: Optional[str] = None

        try:
            with open(log_path, "w", encoding="utf-8") as lf:
                lf.write(f"[worker={worker_id}] gpu={gpu_id}\n")
                lf.write("[cmd] " + " ".join(shlex.quote(x) for x in cmd) + "\n\n")
                lf.flush()

                if dry_run:
                    rc = 0
                else:
                    p = subprocess.run(
                        cmd,
                        stdout=lf,
                        stderr=subprocess.STDOUT,
                        env=env,
                        check=False,
                    )
                    rc = int(p.returncode)
        except Exception as ex:
            rc = 999
            err = f"{type(ex).__name__}: {ex}"

        t1 = time.time()
        result_q.put(
            {
                "rank": task["rank"],
                "score": task.get("score"),
                "gpu": gpu_id,
                "input_json": task["input_json"],
                "input_video": in_video,
                "output_dir": out_dir,
                "log_path": log_path,
                "returncode": rc,
                "error": err,
                "elapsed_sec": round(t1 - t0, 3),
            }
        )
        task_q.task_done()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Process top-N videos (sorted by score) with video-depth-anything on multiple GPUs (no DDP)."
    )
    parser.add_argument(
        "--sorted_json",
        "-s",
        type=str,
        required=True,
        help="Input sorted JSON (contains key 'sorted')",
    )
    parser.add_argument(
        "--output_root", "-o", type=str, required=True, help="Root output directory"
    )
    parser.add_argument(
        "--top_n",
        "-n",
        type=int,
        default=100,
        help="Process top N videos (<=0 means all)",
    )
    parser.add_argument(
        "--start_idx", "-t", type=int, default=0, help="Start index within sorted list"
    )
    parser.add_argument(
        "--gpus", "-g", type=str, default="0", help='GPU ids, e.g. "0,1,2,3" or "all"'
    )
    parser.add_argument(
        "--jobs_per_gpu",
        "-j",
        type=int,
        default=1,
        help="How many parallel workers per GPU (default 1)",
    )
    parser.add_argument(
        "--python",
        "-p",
        type=str,
        default="python3",
        help="Python executable to run depth script",
    )
    parser.add_argument(
        "--run_py",
        "-r",
        type=str,
        default="video-depth-anything/run.py",
        help="Path to video-depth-anything/run.py",
    )
    parser.add_argument(
        "--encoder", "-e", type=str, default="vitl", help='Encoder, e.g. "vitl"'
    )
    parser.add_argument(
        "--extra",
        "-x",
        type=str,
        default="",
        help="Extra args appended to run.py command",
    )
    parser.add_argument(
        "--skip_existing",
        "-k",
        action="store_true",
        help="Skip if output subdir already exists & non-empty",
    )
    parser.add_argument(
        "--dry_run",
        "-d",
        action="store_true",
        help="Print/log commands but do not execute",
    )
    # 默认开启 metric；如需关闭用 --no_metric
    parser.add_argument(
        "--metric",
        dest="metric",
        action="store_true",
        default=True,
        help="Enable --metric (default on)",
    )
    parser.add_argument(
        "--no_metric", dest="metric", action="store_false", help="Disable --metric"
    )
    args = parser.parse_args()

    sorted_json = Path(args.sorted_json).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "_logs").mkdir(parents=True, exist_ok=True)

    items = load_sorted_items(sorted_json)
    tasks, skipped = build_tasks(
        items=items,
        top_n=args.top_n,
        start_idx=args.start_idx,
        output_root=output_root,
        skip_existing=args.skip_existing,
    )

    gpus = parse_gpus(args.gpus)
    if args.jobs_per_gpu < 1:
        print("[ERROR] --jobs_per_gpu must be >= 1", file=sys.stderr)
        return 2

    # 总 worker 数 = len(gpus) * jobs_per_gpu；GPU 分配按重复列表
    gpu_assignments: List[str] = []
    for gid in gpus:
        for _ in range(args.jobs_per_gpu):
            gpu_assignments.append(gid)
    num_workers = len(gpu_assignments)

    extra_args = shlex.split(args.extra) if args.extra.strip() else []

    # 队列与 worker
    task_q: JoinableQueue = JoinableQueue(maxsize=max(8, num_workers * 2))
    result_q: Queue = Queue()

    workers: List[Process] = []
    for wid, gid in enumerate(gpu_assignments):
        p = Process(
            target=worker_loop,
            args=(
                wid,
                gid,
                task_q,
                result_q,
                args.python,
                args.run_py,
                args.encoder,
                args.metric,
                extra_args,
                args.dry_run,
            ),
            daemon=True,
        )
        p.start()
        workers.append(p)

    # 投递任务
    for t in tasks:
        task_q.put(t)

    # 结束信号
    for _ in range(num_workers):
        task_q.put(None)

    # 等待完成
    task_q.join()

    # 收集结果
    results: List[Dict[str, Any]] = []
    expected = len(tasks)
    while len(results) < expected:
        try:
            results.append(result_q.get(timeout=2.0))
        except Exception:
            break

    # 按 rank 排序输出
    results.sort(key=lambda x: int(x.get("rank", 10**9)))

    ok = [r for r in results if r.get("returncode", 1) == 0 and not r.get("error")]
    failed = [
        r for r in results if not (r.get("returncode", 1) == 0 and not r.get("error"))
    ]

    out_obj = {
        "meta": {
            "sorted_json": str(sorted_json),
            "output_root": str(output_root),
            "top_n": args.top_n,
            "start_idx": args.start_idx,
            "gpus": gpus,
            "jobs_per_gpu": args.jobs_per_gpu,
            "python": args.python,
            "run_py": args.run_py,
            "encoder": args.encoder,
            "metric": bool(args.metric),
            "extra": args.extra,
            "skip_existing": bool(args.skip_existing),
            "dry_run": bool(args.dry_run),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "summary": {
            "num_tasks_planned": len(tasks),
            "num_skipped": len(skipped),
            "num_success": len(ok),
            "num_failed": len(failed),
        },
        "skipped": skipped,
        "results": results,
    }

    out_path = output_root / "depth_processing_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote results: {out_path}")
    if failed:
        print(f"[WARN] Failed: {len(failed)} (see logs under {output_root / '_logs'})")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
