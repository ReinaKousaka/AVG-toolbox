export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# python da3_batched_run_ray.py --input_dirs raw_kcdp1_432p --output_dir raw_kcdp1_432p_da3 --process_res 640 --pose_overlap 1 --chunk_size 500
# python da3_batched_run_ray.py --input_dirs raw_kcdp2_432p --output_dir raw_kcdp2_432p_da3 --process_res 640 --pose_overlap 1 --chunk_size 500
# python da3_batched_run_ray.py --input_dirs raw_osaka-u_432p --output_dir raw_osaka-u_432p_da3 --process_res 640 --pose_overlap 1 --chunk_size 500

# python video_qwen3vl_segments.py    --input_dir raw_kcdp1_432p   --out_dir raw_kcdp1_432p_prompt   --model_id Qwen/Qwen3-VL-30B-A3B-Instruct   --segment_size 80   --downscale_ratio 0.5   --num_gpus 7
python video_qwen3vl_segments.py    --input_dir raw_kcdp2_432p   --out_dir raw_kcdp2_432p_prompt   --model_id Qwen/Qwen3-VL-30B-A3B-Instruct   --segment_size 80   --downscale_ratio 0.5   --num_gpus 8
python video_qwen3vl_segments.py    --input_dir raw_osaka-u_432p   --out_dir raw_osaka-u_432p_prompt   --model_id Qwen/Qwen3-VL-30B-A3B-Instruct   --segment_size 80   --downscale_ratio 0.5   --num_gpus 8
python video_qwen3vl_segments.py    --input_dir raw_2077-11-29_432p   --out_dir raw_2077-11-29_432p_prompt   --model_id Qwen/Qwen3-VL-30B-A3B-Instruct   --segment_size 80   --downscale_ratio 0.5   --num_gpus 8

python batch_frustum.py raw_kcdp1_432p 8 frustum_vipe_da3.py --extra "--cam_dir raw_kcdp1_432p_da3 --depth_dir raw_kcdp1_432p_da3 --video_dir raw_kcdp1_432p -o raw_kcdp1_432p_frustum -or -ps 5 "
python batch_frustum.py raw_kcdp2_432p 8 frustum_vipe_da3.py --extra "--cam_dir raw_kcdp2_432p_da3 --depth_dir raw_kcdp2_432p_da3 --video_dir raw_kcdp2_432p -o raw_kcdp2_432p_frustum -or -ps 5 "
python batch_frustum.py raw_osaka-u_432p 8 frustum_vipe_da3.py --extra "--cam_dir raw_osaka-u_432p_da3 --depth_dir raw_osaka-u_432p_da3 --video_dir raw_osaka-u_432p -o raw_osaka-u_432p_frustum -or -ps 5 "