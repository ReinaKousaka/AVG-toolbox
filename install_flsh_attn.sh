# 可选：清理旧缓存
pip uninstall -y flash-attn

git clone --recursive https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS=64 && pip install flash-attn==2.7.4.post1 --no-build-isolation
