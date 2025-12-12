#!/usr/bin/env python
import os
import sys
import argparse
import subprocess
import multiprocessing as mp
import shlex


def build_tasks(intr_dir, extr_dir):
    """
    根据 intr_dir 和 extr_dir 里同名 npz 构建任务列表
    (intr_path, extr_path)
    """
    tasks = []
    intr_files = sorted(f for f in os.listdir(intr_dir) if f.lower().endswith(".npz"))

    for fname in intr_files:
        intr_path = os.path.join(intr_dir, fname)
        extr_path = os.path.join(extr_dir, fname)
        if os.path.isfile(extr_path):
            tasks.append((intr_path, extr_path))
        else:
            print(f"[WARN] extrinsics not found for {fname}, skip.", flush=True)

    return tasks


def run_one(task):
    """
    在子进程中调用原脚本 main（通过命令行调用，避免 argparse 冲突）.
    """
    script_path, intr_path, extr_path, child_args = task

    cmd = [
        sys.executable,
        script_path,
        "--intr",
        intr_path,
        "--extr",
        extr_path,
    ] + child_args

    print(f"[INFO] Start: {os.path.basename(intr_path)}", flush=True)
    try:
        subprocess.run(cmd, check=True)
        print(f"[INFO] Done : {os.path.basename(intr_path)}", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed: {os.path.basename(intr_path)} -> {e}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Multiprocess runner for vipe camera intersection script. "
            "It scans intrinsics / extrinsics dirs for matching names and "
            "runs the existing script on each pair in parallel."
        )
    )

    parser.add_argument(
        "--script",
        type=str,
        required=True,
        help="Path to the existing single-pair script (the one with main()).",
    )
    parser.add_argument(
        "--intr-dir",
        type=str,
        required=True,
        help="Directory containing intrinsics .npz files.",
    )
    parser.add_argument(
        "--extr-dir",
        type=str,
        required=True,
        help="Directory containing extrinsics .npz files.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=0,
        help="Number of parallel processes (0 or less = use os.cpu_count()).",
    )
    parser.add_argument(
        "--child-args",
        type=str,
        default="-v",
        help=(
            "Extra args passed to each child script invocation, "
            "e.g. '-v --max-frame-gap 1800'. Parsed with shlex.split()."
        ),
    )

    args = parser.parse_args()

    script_path = os.path.abspath(args.script)
    intr_dir = os.path.abspath(args.intr_dir)
    extr_dir = os.path.abspath(args.extr_dir)

    if not os.path.isfile(script_path):
        print(f"[FATAL] script not found: {script_path}")
        sys.exit(1)
    if not os.path.isdir(intr_dir):
        print(f"[FATAL] intrinsics dir not found: {intr_dir}")
        sys.exit(1)
    if not os.path.isdir(extr_dir):
        print(f"[FATAL] extrinsics dir not found: {extr_dir}")
        sys.exit(1)

    tasks_pairs = build_tasks(intr_dir, extr_dir)
    if not tasks_pairs:
        print("[FATAL] no matching intr/extr npz pairs found.")
        sys.exit(1)

    child_args = shlex.split(args.child_args) if args.child_args else []

    # 构造最终任务：(script_path, intr_path, extr_path, child_args)
    tasks = [
        (script_path, intr_path, extr_path, child_args)
        for (intr_path, extr_path) in tasks_pairs
    ]

    num_jobs = args.jobs if args.jobs and args.jobs > 0 else (os.cpu_count() or 1)
    print(
        f"[INFO] Found {len(tasks)} pairs. Using {num_jobs} parallel workers.",
        flush=True,
    )

    with mp.Pool(processes=num_jobs) as pool:
        pool.map(run_one, tasks)


if __name__ == "__main__":
    main()
