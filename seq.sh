# mkdir kcd_old
# mv kcdold* kcd_old/

#!/usr/bin/env bash

# 输入和输出文件夹
INPUT_DIR="kcd_old"
OUTPUT_DIR="raw_kcd_old_301fps"

# 创建输出文件夹（如果不存在）
mkdir -p "$OUTPUT_DIR"

# 遍历所有 mp4 文件
for input_file in "$INPUT_DIR"/*.mp4; do
    # 取得文件名
    filename=$(basename "$input_file")

    # 输出路径
    output_file="$OUTPUT_DIR/$filename"

    echo "Processing: $input_file -> $output_file"

    ffmpeg -y -i "$input_file" \
        -vf "setpts=2*PTS,fps=30" \
        -an \
        "$output_file"
done

python video60forceto30.py raw_kcd_old_301fps raw_kcd_old_30fps

python split_videos.py --input_dir raw_kcd_old_30fps   --output_dir raw_kcd_old_576p   --drop_seconds 5    --resize 1120x630   --sample_ratio 1    --crop 1024x576 --interval_frames 1800  --crf 18    --preset slow --keep_temp false -j 20

export CUDA_VISIBLE_DEVICES="0,1,2,3" && python da3_batched_run_ray.py --input_dirs raw_kcd_old_576p --output_dir raw_kcd_old_576p_da3 --process_res 700 --pose_overlap 1 --chunk_size 500

export CUDA_VISIBLE_DEVICES="0,1,2,3" &&  python batch_frustum.py raw_kcd_old_576p 4 frustum_vipe_da3.py --gpu-list 0,1,2,3 --extra "--cam_dir raw_kcd_old_576p_da3 --depth_dir raw_kcd_old_576p_da3 --video_dir raw_kcd_old_576p -o raw_kcd_old_576p_frustum -or -ps 5 -pw 64 -ph 36 -hrz 1000"


# export CUDA_VISIBLE_DEVICES="4,5,6,7" &&  python video_qwen3vl_frames.py    --input_dir raw_kcd_old_576p   --out_dir  raw_kcd_old_576p_prompt   --model_id Qwen/Qwen3-VL-30B-A3B-Instruct --downscale_ratio 0.4   --num_gpus 4