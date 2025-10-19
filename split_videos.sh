#!/bin/bash

# 用法： ./split_videos.sh <源文件夹路径> <N份>
# 示例： ./split_videos.sh /path/to/videos 4

SRC_DIR="$1"
N="$2"

# 检查参数
if [ -z "$SRC_DIR" ] || [ -z "$N" ]; then
  echo "用法: $0 <源文件夹路径> <N份>"
  exit 1
fi

# 检查文件夹是否存在
if [ ! -d "$SRC_DIR" ]; then
  echo "错误: 文件夹 $SRC_DIR 不存在"
  exit 1
fi

# 获取所有 mp4 文件
FILES=($(find "$SRC_DIR" -maxdepth 1 -type f -name "*.mp4" | sort))
TOTAL=${#FILES[@]}

if [ "$TOTAL" -eq 0 ]; then
  echo "未找到任何 mp4 文件。"
  exit 0
fi

echo "总共有 $TOTAL 个 mp4 文件，将分成 $N 份。"

# 计算每份的大小（向上取整）
PER_PART=$(( (TOTAL + N - 1) / N ))

# 创建文件夹并移动文件
for ((i=0; i<N; i++)); do
  START=$((i * PER_PART))
  END=$((START + PER_PART - 1))

  NEW_DIR="${SRC_DIR}/part_$((i+1))"
  mkdir -p "$NEW_DIR"

  echo "创建文件夹: $NEW_DIR"

  for ((j=START; j<=END && j<TOTAL; j++)); do
    FILE="${FILES[$j]}"
    echo "移动文件: $(basename "$FILE") → $NEW_DIR"
    mv "$FILE" "$NEW_DIR/"
  done
done

echo "✅ 文件已成功分配到 $N 个文件夹中。"
