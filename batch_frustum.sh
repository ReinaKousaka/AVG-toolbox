#!/usr/bin/env bash
set -Eeuo pipefail

# 用法:
#   ./dispatch_mp4_jobs.sh <input_dir> <N> <python_script> [--timeout-secs 3600] [--gpu-list 0,1,2,3] [--extra "more args"]
#
# 说明:
#   - <input_dir>:   含有若干 .mp4 的输入文件夹
#   - <N>:           使用的 GPU 个数(整数)
#   - <python_script>: 要执行的 python 脚本路径 (例如 /path/to/xxx.py)
#   - --timeout-secs: (可选) 单条任务超时秒数，默认 3600 (1 小时)
#   - --gpu-list:     (可选) 指定要使用的 GPU ID，逗号分隔。默认使用 0..N-1
#   - --extra:        (可选) 额外透传给 python 的参数字符串（整体用引号包起来）
#
# 任务调用格式:
#   python <python_script> --assign_name=<filename> <extra_args>
#
# 产物:
#   - 在当前工作目录下创建 temp dir，里面有:
#       * N 个分片 txt:  <input_basename>_part<k>.txt (仅文件名，每行一个)
#       * N 个 log:      gpu<gpu_id>.log
#
# 备注:
#   - 均分规则: 对排序后的文件名按轮询方式分配(索引 % N)，结果可重复复现
#   - 需要核心工具: bash、find、sort、date、mktemp
#   - 如系统提供 coreutils 的 `timeout`，则启用超时强杀；如无，则降级为无超时(会在日志中注明)

if [[ $# -lt 3 ]]; then
  echo "用法: $0 <input_dir> <N> <python_script> [--timeout-secs 3600] [--gpu-list 0,1,2,3] [--extra \"...\"]"
  exit 1
fi

INPUT_DIR="$1"
N="$2"
PY_SCRIPT="$3"
shift 3

# 默认参数
TIMEOUT_SECS=3600
GPU_LIST=""
EXTRA_ARGS=""

# 解析可选参数
while (( "$#" )); do
  case "$1" in
    --timeout-secs)
      TIMEOUT_SECS="${2:-3600}"
      shift 2
      ;;
    --gpu-list)
      GPU_LIST="${2:-}"
      shift 2
      ;;
    --extra)
      EXTRA_ARGS="${2:-}"
      shift 2
      ;;
    *)
      echo "未知参数: $1"
      exit 1
      ;;
  esac
done

# 基础检查
if [[ ! -d "$INPUT_DIR" ]]; then
  echo "输入目录不存在: $INPUT_DIR"
  exit 1
fi
if ! [[ "$N" =~ ^[0-9]+$ ]] || [[ "$N" -le 0 ]]; then
  echo "N 必须是正整数: $N"
  exit 1
fi
if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "Python 脚本不存在: $PY_SCRIPT"
  exit 1
fi

# 构造 GPU 列表
if [[ -z "$GPU_LIST" ]]; then
  # 默认使用 0..N-1
  GPU_LIST=$(seq -s, 0 $((N-1)))
fi

# 解析为数组
IFS=',' read -r -a GPU_IDS <<< "$GPU_LIST"
if [[ "${#GPU_IDS[@]}" -ne "$N" ]]; then
  echo "GPU 个数与 N 不一致: N=$N, GPU_LIST=[$GPU_LIST]"
  exit 1
fi

# 收集并排序 mp4 文件(仅文件名)
mapfile -t ALL_FILES < <(find "$INPUT_DIR" -maxdepth 1 -type f -name '*.mp4' -printf '%f\n' | sort)
TOTAL="${#ALL_FILES[@]}"
if [[ "$TOTAL" -eq 0 ]]; then
  echo "未在 $INPUT_DIR 中找到任何 .mp4 文件"
  exit 0
fi

# 创建临时工作目录
TMPDIR=$(mktemp -d -p "$(pwd)" "mp4_jobs_XXXXXX")
INPUT_BASE="$(basename "$INPUT_DIR")"
echo "工作目录: $TMPDIR"
echo "文件总数: $TOTAL"
echo "GPU 列表: ${GPU_IDS[*]}"
echo "超时(秒): $TIMEOUT_SECS"
[[ -n "$EXTRA_ARGS" ]] && echo "额外参数: $EXTRA_ARGS"

# 生成 N 个分片 txt (轮询分配，稳定可复现)
for ((k=0; k< N; k++)); do
  PART_TXT="$TMPDIR/${INPUT_BASE}_part$((k+1)).txt"
  : > "$PART_TXT"   # 清空/新建
done

for i in "${!ALL_FILES[@]}"; do
  idx=$(( i % N ))
  PART_TXT="$TMPDIR/${INPUT_BASE}_part$((idx+1)).txt"
  printf "%s\n" "${ALL_FILES[$i]}" >> "$PART_TXT"
done

# 检查是否有系统 timeout
HAS_TIMEOUT=0
if command -v timeout >/dev/null 2>&1; then
  HAS_TIMEOUT=1
fi

# 定义每个 GPU 的工作函数
run_worker() {
  local gpu_id="$1"
  local part_txt="$2"
  local py_script="$3"
  local timeout_secs="$4"
  local tmpdir="$5"
  local extra_args="$6"

  local log_file="$tmpdir/gpu${gpu_id}.log"

  {
    echo "========== GPU ${gpu_id} 开始 =========="
    echo "开始时间: $(date '+%F %T')"
    echo "任务清单: ${part_txt}"
    echo "脚本路径: ${py_script}"
    echo "超时设置: ${timeout_secs} 秒 (timeout 可用: ${HAS_TIMEOUT})"
    echo

    # 逐行读取文件名
    while IFS= read -r fname || [[ -n "$fname" ]]; do
      [[ -z "$fname" ]] && continue
      echo "---- $(date '+%F %T') | 处理文件: ${fname} ----"

      export CUDA_VISIBLE_DEVICES="${gpu_id}"

      # 记录开始时间戳
      local start_ts end_ts elapsed rc
      start_ts=$(date +%s)

      if [[ "$HAS_TIMEOUT" -eq 1 ]]; then
        # 使用 coreutils timeout，超时返回码通常是 124
        set +e
        timeout --signal=SIGTERM --kill-after=30s "${timeout_secs}s" \
          python "$py_script" --assign_name="${fname}" ${extra_args} 2>&1
        rc=$?
        set -e
        if [[ $rc -eq 124 ]]; then
          echo "[超时] ${fname} 超过 ${timeout_secs}s，已强制终止并跳过。"
        elif [[ $rc -ne 0 ]]; then
          echo "[失败] ${fname} 返回码: $rc"
        else
          echo "[成功] ${fname}"
        fi
      else
        echo "[提示] 系统无 'timeout' 命令，本次不启用超时终止。"
        set +e
        python "$py_script" --assign_name="${fname}" ${extra_args} 2>&1
        rc=$?
        set -e
        if [[ $rc -ne 0 ]]; then
          echo "[失败] ${fname} 返回码: $rc"
        else
          echo "[成功] ${fname}"
        fi
      fi

      end_ts=$(date +%s)
      elapsed=$(( end_ts - start_ts ))
      echo "用时: ${elapsed} 秒"
      echo
    done < "$part_txt"

    echo "结束时间: $(date '+%F %T')"
    echo "========== GPU ${gpu_id} 完成 =========="
    echo
  } | tee -a "$log_file"
}

export -f run_worker
trap 'echo "捕获到 Ctrl+C, 正在终止所有子任务..."; kill 0; exit 130' INT
# 并行启动 N 个 worker
pids=()
for ((k=0; k< N; k++)); do
  PART_TXT="$TMPDIR/${INPUT_BASE}_part$((k+1)).txt"
  GPU_ID="${GPU_IDS[$k]}"
  # 每个 worker 后台运行
  bash -c "run_worker \"$GPU_ID\" \"$PART_TXT\" \"$PY_SCRIPT\" \"$TIMEOUT_SECS\" \"$TMPDIR\" \"$EXTRA_ARGS\"" &
  pids+=($!)
done

# 等待全部完成
fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done

echo "所有子任务已结束。结果与日志位于: $TMPDIR"
exit $fail
