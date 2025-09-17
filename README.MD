# 0) 建议装环境（Python 3.10+；PyTorch >= 2.x；matplotlib）
pip install torch torchvision torchaudio  # 选你本机 CUDA 版本的安装命令
pip install matplotlib numpy

# 1) 生成一个“合成异方差”数据集（只是用来冒烟测）
python gen_data.py --out data/synth --N 1200 --T 50 --seed 0

# 2) 训练加速度计（acc）
python train.py \
  --route acc \
  --train_npz data/synth/train.npz \
  --val_npz   data/synth/val.npz   \
  --test_npz  data/synth/test.npz  \
  --run_dir   runs/acc \
  --x_mode both \
  --epochs 10 --batch_size 32 --lr 1e-3 \
  --logv_min -12 --logv_max 6

# 3) 训练陀螺仪（gyr）
python train.py \
  --route gyr \
  --train_npz data/synth/train.npz \
  --val_npz   data/synth/val.npz   \
  --test_npz  data/synth/test.npz  \
  --run_dir   runs/gyr \
  --x_mode both \
  --epochs 10 --batch_size 32 --lr 1e-3 \
  --logv_min -16 --logv_max 4

# 4) 单独评测
python eval.py --route acc --npz data/synth/test.npz --model runs/acc/best.pt
python eval.py --route gyr --npz data/synth/test.npz --model runs/gyr/best.pt

# 5) 画诊断图（z²直方图、err² vs var散点、logvar时序）
python analyze.py --route acc --npz data/synth/test.npz --model runs/acc/best.pt --out plots_acc
python analyze.py --route gyr --npz data/synth/test.npz --model runs/gyr/best.pt --out plots_gyr
