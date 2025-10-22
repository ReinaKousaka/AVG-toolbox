# 我们为了处理数据特殊调整过的vipe： #
## 1. 首先按照vipe自己说的环境配置安装vipe，这个几乎就是全自动的 ## 
### 注意！ cuda12.6+，最好是cuda12.8，使用 ```nvcc -V``` 确认 ###
```bash
# Create a new conda environment and install 3rd-party dependencies
conda env create -f envs/base.yml
conda activate vipe
# You can switch to your own PyPI index if you want.
pip install -r envs/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128

# Build the project and install it into the current environment
# Omit the -e flag to install the project as a regular package
pip install --no-build-isolation -e .
```

## 2. 【仅记录，不用操作】configs/streams/raw_mp4_stream.yaml 这个里面的frame_end 我们从1000修改到了8000，default.yaml里面save_artifacts改成了true，而save_viz改成了false ##
## 3. 数据处理流程 ##
在一切之前，从dropbox下载视频并且解压，例如
```bash
dbxcli-linux-amd64 get 1018_video_data.zip
```
如果文件名有问题，一些改名的小命令：（记得先sudo apt install rename
# 先试运行看看会改哪些（不真正改名）
rename -n 's/_mp4$/.mp4/' -- *_mp4

# 确认无误后执行并显示改名详情
rename -v 's/_mp4$/.mp4/' -- *_mp4

### a. 准备给vipe的视频 ###
先把视频切片成8000帧一段，640*360，一般情况下我们录屏的视频都是60fps的，抽帧到30
```bash
python split_videos.py \
--input_dir 1018_video_data   \
--output_dir raw_video_vipe_1018   \
--drop_seconds 5    \
--resize 800x450   \
--sample_ratio 2    \
--crop 640x360 \
--interval_frames 8000  \
--crf 18    \
--preset slow   \
--keep_temp false
```

### b. 把视频分成不同的文件夹并且vipe ###
在多gpu的情况下。将视频移动到不同的子文件夹下面：
```bash
bash split_videos.sh [刚刚split_videos.py的输出目录] [机器上的gpu数量，如果是八卡就填8]
【例如】： bash split_videos.sh raw_video_vipe_1018 8
```
然后批量的去跑vipe
```bash
bash batch_vipe.sh [刚刚split_videos.py的输出目录] run.py
【例如】bash batch_vipe.sh raw_video_vipe_1018 run.py
```
这个会在本地生成logs文件夹，下面会有log，可以看一眼，出现
```
Caching:   0%|          | 0/30 [00:00<?, ?it/s]
Caching:  40%|████      | 12/30 [00:00<00:00, 118.13it/s]
Caching:  80%|████████  | 24/30 [00:00<00:00, 117.89it/s]
Caching: 100%|██████████| 30/30 [00:00<00:00, 105.67it/s]

SLAM Pass (1/2):  43%|████▎     | 128/300 [00:18<01:16,  2.25it/s]
SLAM Pass (1/2):  43%|████▎     | 130/300 [00:18<01:40,  1.70it/s]
SLAM Pass (1/2):  44%|████▎     | 131/300 [00:18<01:31,  1.85it/s]
SLAM Pass (1/2):  44%|████▍     | 133/300 [00:18<01:09,  2.40it/s]
SLAM Pass (1/2):  45%|████▌     | 135/300 [00:18<00:54,  3.04it/s]
SLAM Pass (1/2):  46%|████▌     | 137/300 [00:19<00:45,  3.61it/s]
SLAM Pass (1/2):  47%|████▋     | 140/300 [00:19<00:33,  4.80it/s]
SLAM Pass (1/2):  47%|████▋     | 142/300 [00:19<00:28,  5.49it/s]
SLAM Pass (1/2):  48%|████▊     | 144/300 [00:20<00:26,  5.84it/s]
SLAM Pass (1/2):  49%|████▊     | 146/300 [00:20<00:23,  6.45it/s]
SLAM Pass (1/2):  49%|████▉     | 148/300 [00:20<00:22,  6.64it/s]
SLAM Pass (1/2):  50%|█████     | 151/300 [00:20<00:19,  7.58it/s]
```
之类的信息就代表在正常vipe了

### c. 计算frustum ###
vipe的话如果我们不改vipe的代码它的输出路径一般都是在当前目录下的vipe_results里面。
所以我们可以直接起一个batch化的frustum计算把frustum计算任务分配到每个gpu里面：
```bash
# 用法:
python batch_frustum.py /path/to/inputs 8 frustum_vipe.py \
--timeout-secs 7200 \
--gpu-list 0,2,4,7 \
--extra "--flag_x=1 --mode=train"
# 说明:
#   - <input_dir>:   含有若干 .mp4 的输入文件夹
#   - <N>:           使用的 GPU 个数(整数)
#   - <python_script>: 要执行的 python 脚本路径 (例如 /path/to/xxx.py)
#   - --gpu-list:     (可选) 指定要使用的 GPU ID，逗号分隔。默认使用 0..N-1
#   - --extra:        (可选) 额外透传给 python 的参数字符串（整体用引号包起来）
# 对应八卡情况:
python batch_frustum.py vipe_results/rgb 8 frustum_vipe.py --gpu-list 0,1,2,3,4,5,6,7
# 默认的可视化概率是10%，如果希望查看的话可以把可视化概率改成100%来看frustum是否正常工作
python batch_frustum.py vipe_results/rgb 8 frustum_vipe.py --gpu-list 0,1,2,3,4,5,6,7 --extra "--verbose_prob 1"
 ```