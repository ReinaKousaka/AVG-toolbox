export CUDA_VISIBLE_DEVICES=4,5,6,7

python video_qwen3vl_segments.py    --input_dir raw_2077-12-2-2_432p   --out_dir raw_2077-12-2-2_432p_prompt   --model_id Qwen/Qwen3-VL-30B-A3B-Instruct   --segment_size 160   --downscale_ratio 0.5   --num_gpus 4
python video_qwen3vl_segments.py    --input_dir raw_2077-12-2-3_432p   --out_dir raw_2077-12-2-3_432p_prompt   --model_id Qwen/Qwen3-VL-30B-A3B-Instruct   --segment_size 160   --downscale_ratio 0.5   --num_gpus 4