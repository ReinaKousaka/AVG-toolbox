# python video60forceto30.py V_sekai_ed1 V_sekai_ed1_30fps -w 64

# python split_videos.py --input_dir V_sekai_ed1   --output_dir V_sekai-ed1_576p   --drop_seconds 6    --resize 1280x720   --sample_ratio 2    --crop 1024x576 --interval_frames 2000  --crf 18    --preset slow --keep_temp false -j 64 && python 576pto448p.py --folders V_sekai-ed1_576p --workers 64 --overwrite

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && python da3_batched_run_ray.py --input_dirs V_2077-ad2 --output_dir V_2077-ad2_da3 --process_res 700 --pose_overlap 1 --chunk_size 501
# frustum_vipe_FASTCHECK.py

# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" &&  python batch_frustum.py V_sim_448p 8 frustum_vipe_FASTCHECK.py --gpu-list 0,1,2,3,4,5,6,7 --extra "--cam_dir vipe_camparams --depth_dir V_sim_344p_da3 --video_dir V_sim_448p -o V_sim_448p_frustum_FASTCHECK -or -ps 6 -pw 52 -ph 28 -hrz 200 -v"


export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" &&  python batch_frustum.py V_2077-ad2 8 frustum_vipe_da3.py --gpu-list 0,1,2,3,4,5,6,7 --extra "--cam_dir V_2077-ad2_da3 --depth_dir V_2077-ad2_da3 --video_dir V_2077-ad2 -o V_2077-ad2_frustum -or -ps 6 -pw 52 -ph 28 -hrz 1000 -v"

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" &&  python video_qwen3vl_frames.py    --input_dir V_2077-ad2   --out_dir V_2077-ad2_prompt   --model_id Qwen/Qwen3-VL-30B-A3B-Instruct --downscale_ratio 1  --num_gpus 8 --detail_chunk 11 --frame_interval 8

# dbxcli-linux-amd64 get "V_sim_344p_da3.zip.part_aa" &
# dbxcli-linux-amd64 get "V_sim_344p_da3.zip.part_ab" &
# dbxcli-linux-amd64 get "V_sim_344p_da3.zip.part_ac" &
# dbxcli-linux-amd64 get "V_sim_344p_da3.zip.part_ad" &
# dbxcli-linux-amd64 get "V_sim_344p_da3.zip.part_ae" &
# dbxcli-linux-amd64 get "V_sim_344p_da3.zip.part_af" &
# dbxcli-linux-amd64 get "V_sim_344p_da3.zip.part_ag" &

# wait

# cat V_sim_344p_da3.zip.part_* > V_sim_344p_da3.zip

# unzip V_sim_344p_da3.zip