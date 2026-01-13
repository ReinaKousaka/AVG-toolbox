# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && python da3_batched_run_ray.py --input_dirs V_real_344p --output_dir V_real_344p_da3 --process_res 504 --pose_overlap 1 --chunk_size 501

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && python da3_batched_run_ray.py --input_dirs sekai-real-walking_448p --output_dir sekai-real-walking_448p_da3 --process_res 504 --pose_overlap 1 --chunk_size 501

# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" &&  python batch_frustum.py V_sim_448p 8 frustum_vipe_da3.py --gpu-list 0,1,2,3,4,5,6,7 --extra "--cam_dir V_sim_344p_da3 --depth_dir V_sim_344p_da3 --video_dir V_sim_448p -o V_sim_448p_frustum -or -ps 5 -pw 52 -ph 28 -hrz 1000 -v"

# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" &&  python video_qwen3vl_frames.py    --input_dir V_real_344p   --out_dir V_real_344p_prompt   --model_id Qwen/Qwen3-VL-30B-A3B-Instruct --downscale_ratio 1   --num_gpus 8 --detail_chunk 11 --frame_interval 8

# zip -r V_real_344p_da3.zip V_real_344p_da3 &
# zip -r V_sim_344p_da3.zip V_sim_344p_da3 &
# zip -r sekai-real-walking_448p_da3.zip sekai-real-walking_448p_da3 &

# wait

# split -b 20G V_real_344p_da3.zip V_real_344p_da3.zip.part_ &
# split -b 20G V_sim_344p_da3.zip V_sim_344p_da3.zip.part_ &
# split -b 20G sekai-real-walking_448p_da3.zip sekai-real-walking_448p_da3.zip.part_ &

# wait

dbxcli-linux-amd64 put "sekai-real-walking_448p_da3.zip.part_aa" &
dbxcli-linux-amd64 put "sekai-real-walking_448p_da3.zip.part_ab" &
dbxcli-linux-amd64 put "sekai-real-walking_448p_da3.zip.part_ac" &
dbxcli-linux-amd64 put "sekai-real-walking_448p_da3.zip.part_ad" &
dbxcli-linux-amd64 put "sekai-real-walking_448p_da3.zip.part_ae" &
dbxcli-linux-amd64 put "sekai-real-walking_448p_da3.zip.part_af" &
wait

dbxcli-linux-amd64 put "V_real_344p_da3.zip.part_aa" &
dbxcli-linux-amd64 put "V_real_344p_da3.zip.part_ab" &
dbxcli-linux-amd64 put "V_real_344p_da3.zip.part_ac" &
dbxcli-linux-amd64 put "V_real_344p_da3.zip.part_ad" &
dbxcli-linux-amd64 put "V_real_344p_da3.zip.part_ae" &
dbxcli-linux-amd64 put "V_real_344p_da3.zip.part_af" &
dbxcli-linux-amd64 put "V_real_344p_da3.zip.part_ag" &
dbxcli-linux-amd64 put "V_real_344p_da3.zip.part_ah" &
wait

dbxcli-linux-amd64 put "V_sim_344p_da3.zip.part_aa" &
dbxcli-linux-amd64 put "V_sim_344p_da3.zip.part_ab" &
dbxcli-linux-amd64 put "V_sim_344p_da3.zip.part_ac" &
dbxcli-linux-amd64 put "V_sim_344p_da3.zip.part_ad" &
dbxcli-linux-amd64 put "V_sim_344p_da3.zip.part_ae" &
dbxcli-linux-amd64 put "V_sim_344p_da3.zip.part_af" &
dbxcli-linux-amd64 put "V_sim_344p_da3.zip.part_ag" &
wait