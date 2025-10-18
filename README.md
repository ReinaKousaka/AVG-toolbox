我们为了处理数据特殊调整过的vipe：
1. 首先按照vipe自己说的环境配置安装vipe，这个几乎就是全自动的
2. configs/streams/raw_mp4_stream.yaml 这个里面的frame_end 我们从1000修改到了8000
3. default.yaml里面save_artifacts改成了true，而save_viz改成了false

数据处理流程
先把视频切片成1024*576，指定帧数，30fps的切片
```bash
python split_videos.py \
--input_dir in    \
--output_dir raw_video   \
--drop_seconds 5    \
--resize 1280x720   \
--sample_ratio 2    \
--crop 1024x576 \
--interval_frames 1500  \
--crf 18    \
--preset slow   \
--keep_temp false
```
最后视频保存分辨率1024*576，然后因为cam的视频都是7601帧的，所以我们就切8000一个片

然后去raw_video或者指定的切片好的文件夹里面，直接run

```bash
python run.py pipeline=default streams=raw_mp4_stream streams.base_path=raw_video
```

TBD:
后续处理frustum的内容
