# 可选：清理旧缓存
pip uninstall -y flash-attn

git clone --recursive https://github.com/Dao-AILab/flash-attention.git
cd flash-attention

export MAX_JOBS=64
export OMP_NUM_THREADS=1
pip install . --no-build-isolation --no-cache-dir -v
