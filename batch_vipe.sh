#!/usr/bin/env bash
set -euo pipefail

# 用法:
#   ./run_multi.sh <PARENT_DIR> <PY_SCRIPT> [-- 任意额外参数...]
#
# 示例:
#   ./run_multi.sh /data/runs xxx.py -- --epochs 5 --lr 3e-4
#
# 行为:
#   - 自动检测 GPU 数量 (nvidia-smi)
#   - 将子文件夹按字典序分配给 GPU，并发运行
#   - 子文件夹数量 > GPU 数量时，分批轮转直到全部完成
#   - 每个进程仅可见其分配的 GPU (CUDA_VISIBLE_DEVICES)
#   - 日志输出到 logs/<子文件夹名>.log

if [[ $# -lt 2 ]]; then
  echo "用法: $0 <PARENT_DIR> <PY_SCRIPT> [-- 额外参数...]"
  exit 1
fi

PARENT_DIR="$(realpath "$1")"
PY_SCRIPT="$2"
shift 2
# 其余参数（可选）：会直接传给 python 脚本
EXTRA_ARGS=()
if [[ $# -gt 0 ]]; then
  # 允许使用 -- 分隔
  if [[ "$1" == "--" ]]; then shift; fi
  EXTRA_ARGS=("$@")
fi

if [[ ! -d "$PARENT_DIR" ]]; then
  echo "错误: 父文件夹不存在: $PARENT_DIR" >&2
  exit 1
fi

if [[ ! -f "$PY_SCRIPT" ]]; then
  # 也允许传入可解析为PATH的脚本，如 package.module:main 不在此判断
  if [[ ! -f "$(realpath -m "$PY_SCRIPT")" ]]; then
    echo "警告: 找不到本地脚本文件 $PY_SCRIPT ，将直接按参数传给 python 执行。"
  fi
fi

# 收集子文件夹（仅一层）
mapfile -t SUBDIRS < <(find "$PARENT_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
if [[ ${#SUBDIRS[@]} -eq 0 ]]; then
  echo "错误: 父文件夹下未找到任何子文件夹: $PARENT_DIR" >&2
  exit 1
fi

# 检测 GPU 数量
if command -v nvidia-smi >/dev/null 2>&1; then
  # 获取 GPU 索引列表，例如: 0\n1\n2...
  mapfile -t GPU_IDX < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | sed 's/ //g')
  NUM_GPU=${#GPU_IDX[@]}
else
  echo "警告: 未检测到 nvidia-smi，将按 CPU 模式串行运行。"
  NUM_GPU=1
  GPU_IDX=(0)  # 占位
fi

if [[ $NUM_GPU -le 0 ]]; then
  echo "警告: 未检测到可用 GPU，将按 CPU 模式串行运行。"
  NUM_GPU=1
  GPU_IDX=(0)
fi

mkdir -p logs

echo "父目录: $PARENT_DIR"
echo "脚本:    $PY_SCRIPT"
echo "子目录数: ${#SUBDIRS[@]}"
echo "GPU 数量: $NUM_GPU"
echo "额外参数: ${EXTRA_ARGS[*]:-(无)}"
echo

# 轮转调度：每一批最多 NUM_GPU 个任务
batch_start=0
total=${#SUBDIRS[@]}

while [[ $batch_start -lt $total ]]; do
  echo "启动一批任务: $(($batch_start + 1)) ~ $(min() { echo $(( $1 < $2 ? $1 : $2 )); }; min $total $((batch_start + NUM_GPU)))"
  pids=()

  for i in $(seq 0 $((NUM_GPU - 1))); do
    idx=$((batch_start + i))
    if [[ $idx -ge $total ]]; then
      break
    fi

    subdir="${SUBDIRS[$idx]}"
    subname="$(basename "$subdir")"

    # 选取该进程的 GPU（CPU 模式时仍设置变量但不生效）
    gpu="${GPU_IDX[$((i % ${#GPU_IDX[@]}))]}"

    # 日志文件
    logfile="logs/${subname}.log"

    echo "  -> [GPU $gpu] 处理: $subname  (日志: $logfile)"

    # 为该进程限制可见 GPU
    # 如果无 GPU，这个变量不会起作用但也无害
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      set -x
      python run.py pipeline=default streams=raw_mp4_stream streams.base_path="${subdir}" >>"$logfile" 2>&1
    ) &

    pids+=($!)
  done

  # 等待本批次所有任务结束
  fail=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      fail=1
    fi
  done

  if [[ $fail -ne 0 ]]; then
    echo "注意: 本批次有任务失败，详见 logs/*.log"
  fi

  batch_start=$((batch_start + NUM_GPU))
done

echo
echo "全部任务完成。日志在 ./logs 下。"
