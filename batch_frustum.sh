#!/usr/bin/env bash
set -Eeuo pipefail

# 用法:
#   ./dispatch_mp4_jobs.sh <input_dir> <N> <python_script> [--timeout-secs 3600] [--gpu-list 0,1,2,3] [--extra "more args"]
#
# 参数:
#   <input_dir>      含若干 .mp4 的目录
#   <N>              使用的 GPU 个数(正整数)
#   <python_script>  要执行的 Python 脚本路径 (如 /path/to/xxx.py)
#   --timeout-secs   (可选) 单条任务超时秒数，默认 3600
#   --gpu-list       (可选) 使用的 GPU ID 列表，逗号分隔。默认 0..N-1
#   --extra          (可选) 透传给 Python 的额外参数字符串(整段用引号包起来)
#
# 调用格式:
#   python <python_script> --assign_name=<filename> <extra_args>
#
# 产物:
#   在当前工作目录创建临时目录，如 ./mp4_jobs_ab12cd/
#     - 分片清单: <input_basename>_part{k}.txt (每行仅文件名)
#     - 日志    : gpu<gpu_id>.log
#
# 说明:
#   - 文件名先 sort，再按索引 % N 轮询分配 ⇒ 可复现的“均分”
#   - 对 CRLF 做了处理，避免日志被 \r 破坏
#   - 支持 Ctrl+C 传播到所有子进程 (trap + kill 0)
#   - 超时优先使用 coreutils timeout；无则降级为无超时并记录

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
  GPU_LIST
  GPU_LIST=$(seq -s, 0 $((N-1)))
fi
IFS=',' read -r -a GPU_IDS <<< "$GPU_LIST"
if [[ "${#GPU_IDS[@]}" -ne "$N" ]]; then
  echo "GPU 个数与 N 不一致: N=$N, GPU_LIST=[$GPU_LIST]"
  exit 1
fi

# 收集并排序 mp4 文件(仅文件名)
# 使用 LC_ALL=C 确保一致的字典序
mapfile -t ALL_FILES < <(LC_ALL=C find "$INPUT_DIR" -maxdepth 1 -type f -name '*.mp4' -printf '%f\n' | LC_ALL=C sort)
TOTAL="${#ALL_FILES[@]}"
if [[ "$TOTAL" -eq 0 ]]; then
  echo "未在 $INPUT_DIR 中找到任何 .mp4 文件"
  exit 0
fi

# 创建临时工作目录
TMPDIR=$(mktemp -d -p "$(pwd)" "mp4_jobs_XXXXXX")
INPUT_BASE="$(basename "$INPUT_DIR")"
echo "工作目录 : $TMPDIR"
echo "文件总数 : $TOTAL"
echo "GPU 列表 : ${GPU_IDS[*]}"
echo "超时(秒) : $TIMEOUT_SECS"
[[ -n "$EXTRA_ARGS" ]] && echo "额外参数: $EXTRA_ARGS"

# 生成 N 个分片 txt (轮询分配，稳定可复现)
for ((k=0; k< N; k++)); do
  : > "$TMPDIR/${INPUT_BASE}_part$((k+1)).txt"
done
for i in "${!ALL_FILES[@]}"; do
  idx=$(( i % N ))
  printf "%s\n" "${ALL_FILES[$i]}" >> "$TMPDIR/${INPUT_BASE}_part$((idx+1)).txt"
done

# 统一处理 CRLF -> LF，避免日志被 \r 覆盖显示
for f in "$TMPDIR"/*.txt; do
  if command -v dos2unix >/dev/null 2>&1; then
    dos2unix -q "$f"
  else
    # 仅去掉每行尾部的 \r
    sed -i $'s/\r$//' "$f"
  fi
done

# 定义每个 GPU 的工作函数
run_worker() {
  local gpu_id="$1"
  local part_txt="$2"
  local py_script="$3"
  local timeout_secs="$4"
  local tmpdir="$5"
  local extra_args_str="$6"

  # 将额外参数解析为数组，避免错误分词
  local -a extra_arr=()
  if [[ -n "$extra_args_str" ]]; then
    # shellcheck disable=SC2206
    read -r -a extra_arr <<< "$extra_args_str"
  fi

  local log_file="$tmpdir/gpu${gpu_id}.log"

  # 在函数内部再次检测 timeout 是否可用，避免环境变量传递问题
  local has_timeout=0
  if command -v timeout >/dev/null 2>&1; then
    has_timeout=1
  fi

  {
    echo "========== GPU ${gpu_id} 开始 =========="
    echo "开始时间: $(date '+%F %T')"
    echo "任务清单: ${part_txt}"
    echo "脚本路径: ${py_script}"
    echo "超时设置: ${timeout_secs} 秒 (timeout 可用: ${has_timeout})"
    echo

    # 逐行读取文件名（兼容空行与 CRLF）
    while IFS= read -r fname || [[ -n "$fname" ]]; do
      # 去掉尾部 CR (若存在)
      fname=${fname%$'\r'}
      [[ -z "$fname" ]] && continue

      echo "---- $(date '+%F %T') | 处理文件: ${fname} ----"
      export CUDA_VISIBLE_DEVICES="${gpu_id}"

      local start_ts end_ts elapsed rc
      start_ts=$(date +%s)

      if [[ "$has_timeout" -eq 1 ]]; then
        set +e
        timeout --signal=SIGTERM --kill-after=30s "${timeout_secs}s" \
          python "$py_script" --assign_name="${fname}" "${extra_arr[@]}" 2>&1
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
        python "$py_script" --assign_name="${fname}" "${extra_arr[@]}" 2>&1
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

# 捕获 Ctrl+C / SIGTERM：终止整个进程组的所有子进程
trap 'echo "捕获到中断信号，正在终止所有子任务..."; kill 0; exit 130' INT TERM

# 并行启动 N 个 worker
pids=()
for ((k=0; k< N; k++)); do
  PART_TXT="$TMPDIR/${INPUT_BASE}_part$((k+1)).txt"
  GPU_ID="${GPU_IDS[$k]}"
  # 每个 worker 后台运行；使用 bash -c 以便函数可见（依赖 export -f）
  bash -c "run_worker \"$GPU_ID\" \"$PART_TXT\" \"$PY_SCRIPT\" \"$TIMEOUT_SECS\" \"$TMPDIR\" \"$EXTRA_ARGS\"" &
  pids+=($!)
done

# 等待全部完成；只要任一失败，最终 exit 非 0
fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done

echo "所有子任务已结束。结果与日志位于: $TMPDIR"
exit $fail
