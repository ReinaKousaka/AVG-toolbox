

export CUDA_VISIBLE_DEVICES=4,5,6,7

# python video_qwen3vl_segments.py    --input_dir raw_osmo_448p   --out_dir  raw_osmo_448p_prompt   --model_id Qwen/Qwen3-VL-30B-A3B-Instruct   --segment_size 80   --downscale_ratio 0.5   --num_gpus 4

# python video_qwen3vl_segments.py    --input_dir raw_jersey_448p   --out_dir  raw_jersey_448p_prompt   --model_id Qwen/Qwen3-VL-30B-A3B-Instruct   --segment_size 80   --downscale_ratio 0.5   --num_gpus 4

# python da3_batched_run_ray.py --input_dirs raw_osmo_448p --output_dir raw_osmo_448p_da3 --process_res 700 --pose_overlap 1 --chunk_size 500

# python da3_batched_run_ray.py --input_dirs raw_jersey_448p --output_dir raw_jersey_448p_da3 --process_res 700 --pose_overlap 1 --chunk_size 500

python batch_frustum.py raw_osmo_448p 4 frustum_vipe_da3.py --gpu-list 4,5,6,7 --extra "--cam_dir raw_osmo_448p_da3 --depth_dir raw_osmo_448p_da3 --video_dir raw_osmo_448p -o raw_osmo_448p_frustum -or -ps 5 "

python batch_frustum.py raw_jersey_448p 4 frustum_vipe_da3.py --gpu-list 4,5,6,7 --extra "--cam_dir raw_jersey_448p_da3 --depth_dir raw_jersey_448p_da3 --video_dir raw_jersey_448p -o raw_jersey_448p_frustum -or -ps 5 "