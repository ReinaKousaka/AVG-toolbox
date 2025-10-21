#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import tempfile
import time
import threading
import shlex
import subprocess
import signal
from pathlib import Path
from typing import List, Sequence, Optional

# ============== 工具函数 ==============

CTRL_BYTES = bytes([*(range(0x00, 0x20)), 0x7F])

def strip_control_chars(s: str) -> str:
    """删除所有控制字符（0x00-0x1F, 0x7F），保留换行之外的可见字符。"""
    if not s:
        return s
    # 逐字符过滤（包含 \r \b 等）
    return "".join(ch for ch in s if ord(ch) >= 0x20 and ord(ch) != 0x7F)

def list_mp4_filenames(input_dir: Path) -> List[str]:
    """列出目录下所有 .mp4 的**文件名**（不含路径），按字典序稳定排序。"""
    names = []
    for p in input_dir.iterdir():
        if p.is_file() and p.name.lower().endswith(".mp4"):
            # 读到的名字也清洗控制字符
            names.append(strip_control_chars(p.name))
    # 使用二进制/Unicode默认排序即可确保可复现（不依赖 locale）
    names.sort()
    return names

def split_round_robin(items: Sequence[str], n: int) -> List[List[str]]:
    """按 index % n 轮询分配，稳定可复现。"""
    buckets = [[] for _ in range(n)]
    for i, it in enumerate(items):
        buckets[i % n].append(it)
    return buckets

def write_text(path: Path, text: str) -> None:
    # 统一用 LF、UTF-8
    path.write_text(text, encoding="utf-8", newline="\n")

def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

# ============== 子进程执行 ==============

class ProcKiller:
    """帮助安全终止子进程（及其子进程）。"""
    def __init__(self):
        self._lock = threading.Lock()
        self._p: Optional[subprocess.Popen] = None

    def set_proc(self, p: Optional[subprocess.Popen]):
        with self._lock:
            self._p = p

    def kill(self):
        with self._lock:
            p = self._p
        if not p:
            return
        try:
            if hasattr(os, "getpgid"):
                # 给整个进程组发信号
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            else:
                # Windows 或不支持 setpgid 的平台（不常见于 CUDA 环境）
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
        except Exception:
            pass

def run_one_command(
    cmd: List[str],
    env: dict,
    timeout_secs: int,
    log_fp,
    killer: ProcKiller,
) -> int:
    """
    运行一条命令，超时则强杀进程组。stdout/stderr 一并写入 log_fp。
    返回进程返回码；超时用 124。
    """
    start_ts = time.time()
    rc = -1
    try:
        preexec = None
        creationflags = 0
        if os.name == "posix":
            # 独立进程组，便于 killpg
            preexec = os.setsid
        elif os.name == "nt":
            # Windows: 创建新进程组
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            preexec_fn=preexec,
            creationflags=creationflags,
        )
        killer.set_proc(p)

        # 流式读取 + 超时
        deadline = start_ts + timeout_secs
        buf = []
        while True:
            # 每次读一小块，避免阻塞
            line = p.stdout.readline()
            if line:
                # 清理控制字符，避免日志显示错乱
                line = strip_control_chars(line)
                buf.append(line)
                if len(buf) >= 32:
                    log_fp.write("".join(buf))
                    log_fp.flush()
                    buf.clear()
            else:
                if p.poll() is not None:
                    break
                # 未结束但暂时无输出；做超时检查
                if time.time() > deadline:
                    # 超时
                    killer.kill()
                    rc = 124
                    break
                time.sleep(0.05)

        # flush 残余
        if buf:
            log_fp.write("".join(buf))
            log_fp.flush()

        # 如果非超时，取真实 rc
        if rc != 124:
            rc = p.wait(timeout=1)
    except subprocess.TimeoutExpired:
        killer.kill()
        rc = 124
    except Exception as e:
        log_fp.write(f"[异常] 运行出错: {e}\n")
        rc = -1
    finally:
        killer.set_proc(None)

    elapsed = int(time.time() - start_ts)
    log_fp.write(f"用时: {elapsed} 秒\n\n")
    log_fp.flush()
    return rc

# ============== Worker ==============

def gpu_worker(
    gpu_id: str,
    file_list: List[str],
    input_dir: Path,
    py_script: Path,
    extra_args: List[str],
    timeout_secs: int,
    temp_dir: Path,
    stop_event: threading.Event,
):
    log_path = temp_dir / f"gpu{gpu_id}.log"
    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        log.write(f"========== GPU {gpu_id} 开始 ==========\n")
        log.write(f"开始时间: {now_str()}\n")
        log.write(f"任务数: {len(file_list)}\n")
        log.write(f"脚本路径: {py_script}\n")
        log.write(f"超时设置: {timeout_secs} 秒\n\n")
        log.flush()

        killer = ProcKiller()

        for fname_raw in file_list:
            if stop_event.is_set():
                log.write("[中断] 收到停止信号，提前退出。\n")
                break

            # 二次清洗文件名（双保险）
            fname = strip_control_chars(fname_raw).strip()

            # 存在性校验（文件名与目录一致性）
            if not (input_dir / fname).exists():
                log.write(f"[警告] 找不到文件: {fname}\n")
                log.flush()
                continue

            log.write(f"---- {now_str()} | 处理文件: {fname} ----\n")
            log.flush()

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            # 构建命令：使用当前 Python 解释器；如需固定用 "python" 可改为 "python"
            cmd = [sys.executable, str(py_script), f"--assign_name={fname}"]
            if extra_args:
                cmd.extend(extra_args)

            rc = run_one_command(
                cmd=cmd,
                env=env,
                timeout_secs=timeout_secs,
                log_fp=log,
                killer=killer,
            )

            if rc == 124:
                log.write(f"[超时] {fname} 超过 {timeout_secs}s，已强制终止。\n")
            elif rc != 0:
                log.write(f"[失败] {fname} 返回码: {rc}\n")
            else:
                log.write(f"[成功] {fname}\n")
            log.flush()

        log.write(f"结束时间: {now_str()}\n")
        log.write(f"========== GPU {gpu_id} 完成 ==========\n\n")
        log.flush()

# ============== 主流程 ==============

def main():
    parser = argparse.ArgumentParser(
        description="按 N 个 GPU 均分 mp4 任务，逐条执行 python 脚本，超时强杀并记录日志（纯 Python 实现）。"
    )
    parser.add_argument("input_dir", type=Path, help="含若干 .mp4 的输入目录")
    parser.add_argument("N", type=int, help="使用的 GPU 个数")
    parser.add_argument("python_script", type=Path, help="要执行的 Python 脚本路径（将以 --assign_name 调用）")
    parser.add_argument("--timeout-secs", type=int, default=3600, help="单条任务超时秒数，默认 3600")
    parser.add_argument("--gpu-list", type=str, default="", help="使用的 GPU ID 列表，逗号分隔；默认 0..N-1")
    parser.add_argument("--extra", type=str, default="", help='额外透传给子脚本的参数字符串，如："--flag_x=1 --mode=train"')
    parser.add_argument("--temp-root", type=Path, default=Path.cwd(), help="临时目录创建位置，默认当前工作目录")
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    n: int = args.N
    py_script: Path = args.python_script
    timeout_secs: int = args.timeout_secs
    gpu_list_str: str = args.gpu_list.strip()
    extra_args_str: str = args.extra.strip()

    if not input_dir.is_dir():
        parser.error(f"输入目录不存在: {input_dir}")
    if not py_script.is_file():
        parser.error(f"Python 脚本不存在: {py_script}")
    if n <= 0:
        parser.error("N 必须为正整数")

    # 解析 GPU 列表
    if gpu_list_str:
        gpu_ids = [s.strip() for s in gpu_list_str.split(",") if s.strip() != ""]
    else:
        gpu_ids = [str(i) for i in range(n)]
    if len(gpu_ids) != n:
        parser.error(f"GPU 个数与 N 不一致: N={n}, GPU 列表={gpu_ids}")

    # 解析 extra 透传参数（安全拆分）
    extra_args = shlex.split(extra_args_str) if extra_args_str else []

    # 列出并排序 mp4 文件
    all_files = list_mp4_filenames(input_dir)
    if not all_files:
        print(f"未在 {input_dir} 中找到任何 .mp4 文件", file=sys.stderr)
        return 0

    # 临时目录
    temp_dir = Path(tempfile.mkdtemp(prefix="mp4_jobs_", dir=str(args.temp_root)))
    input_base = input_dir.name

    print(f"工作目录 : {temp_dir}")
    print(f"文件总数 : {len(all_files)}")
    print(f"GPU 列表 : {' '.join(gpu_ids)}")
    print(f"超时(秒) : {timeout_secs}")
    if extra_args:
        print(f"额外参数 : {extra_args}")

    # 均分并写入 N 份 txt
    buckets = split_round_robin(all_files, n)
    part_paths = []
    for k in range(n):
        part_path = temp_dir / f"{input_base}_part{k+1}.txt"
        # 写入前全面清洗控制字符 + 一行一个文件名
        content = "\n".join(strip_control_chars(x) for x in buckets[k]) + ("\n" if buckets[k] else "")
        write_text(part_path, content)
        part_paths.append(part_path)

    # 启动 N 个 worker（线程），每个 GPU 串行处理其列表
    stop_event = threading.Event()
    threads = []
    try:
        for k in range(n):
            t = threading.Thread(
                target=gpu_worker,
                name=f"gpu-worker-{gpu_ids[k]}",
                kwargs=dict(
                    gpu_id=gpu_ids[k],
                    file_list=buckets[k],
                    input_dir=input_dir,
                    py_script=py_script,
                    extra_args=extra_args,
                    timeout_secs=timeout_secs,
                    temp_dir=temp_dir,
                    stop_event=stop_event,
                ),
                daemon=True,
            )
            t.start()
            threads.append(t)

        # 主线程等待
        for t in threads:
            while t.is_alive():
                t.join(timeout=0.2)
    except KeyboardInterrupt:
        print("\n[中断] 捕获到 Ctrl+C，通知所有 worker 停止...", file=sys.stderr)
        stop_event.set()
        # 给子线程一些清理时间
        for _ in range(25):
            alive = any(t.is_alive() for t in threads)
            if not alive:
                break
            time.sleep(0.1)
    finally:
        print(f"所有子任务已结束。结果与日志位于: {temp_dir}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
