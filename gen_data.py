from __future__ import annotations
import argparse, os
from pathlib import Path
from utils import load_config_file
import numpy as np


def make_synth(N: int = 1200, T: int = 50, seed: int = 0):
    """
    Create synthetic (X, E2, MASK) with heteroscedastic variance patterns.
    X  : (N,T,6) [acc3, gyr3]
    E2 : (N,T,2) pooled squared error sums for acc / gyr
    MASK: (N,T) all ones (used for potential padding scenarios)
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, T, 6)).astype(np.float32)

    # Smooth each channel to mimic low-pass sensor dynamics
    for d in range(6):
        X[:, :, d] = np.convolve(X[:, :, d].reshape(-1), np.ones(3, dtype=np.float32) / 3.0,
                                 mode="same").reshape(N, T)

    base_acc = rng.uniform(0.02, 0.2, size=N).astype(np.float32)
    base_gyr = rng.uniform(0.01, 0.5, size=N).astype(np.float32)

    E2 = np.zeros((N, T, 2), dtype=np.float32)
    Y_ACC = np.zeros((N, T), dtype=np.float32)
    Y_GYR = np.zeros((N, T), dtype=np.float32)

    for i in range(N):
        a_var = base_acc[i] * (1.0 + 0.8 * np.sin(np.linspace(0, 2 * np.pi, T))
                               + 0.2 * rng.normal(size=T))
        g_var = base_gyr[i] * (1.0 + 0.8 * np.cos(np.linspace(0, 2 * np.pi, T))
                               + 0.2 * rng.normal(size=T))
        a_var = np.clip(a_var, 1e-4, 1e2).astype(np.float32)
        g_var = np.clip(g_var, 1e-4, 1e2).astype(np.float32)

        Y_ACC[i] = a_var
        Y_GYR[i] = g_var

        acc_std = np.sqrt(a_var).astype(np.float32)
        gyr_std = np.sqrt(g_var).astype(np.float32)
        acc_err = rng.normal(scale=acc_std[:, None], size=(T, 3)).astype(np.float32)
        gyr_err = rng.normal(scale=gyr_std[:, None], size=(T, 3)).astype(np.float32)

        X[i, :, :3] += acc_err
        X[i, :, 3:6] += gyr_err
        E2[i, :, 0] = np.sum(acc_err**2, axis=-1)
        E2[i, :, 1] = np.sum(gyr_err**2, axis=-1)

    MASK = np.ones((N, T), dtype=np.float32)
    return X.astype(np.float32), E2, MASK, Y_ACC, Y_GYR


def split_and_save(X, E2, MASK, Y_ACC, Y_GYR, out_dir, ratios=(0.7, 0.15, 0.15)):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    N = X.shape[0]
    n_train = int(N * ratios[0])
    n_val = int(N * ratios[1])
    n_test = N - n_train - n_val
    idx = np.arange(N)
    np.random.shuffle(idx)
    parts = {
        "train": idx[:n_train],
        "val": idx[n_train:n_train + n_val],
        "test": idx[n_train + n_val:]
    }
    for name, ids in parts.items():
        np.savez(out / f"{name}.npz",
                 X=X[ids],
                 E2=E2[ids],
                 MASK=MASK[ids],
                 Y_ACC=Y_ACC[ids],
                 Y_GYR=Y_GYR[ids])


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None, help="YAML/JSON 配置文件")
    args_pre, _ = pre.parse_known_args()

    cfg = load_config_file(args_pre.config)
    gd = cfg.get("gen_data", {})

    ap = argparse.ArgumentParser("Generate toy heteroscedastic IMU dataset", parents=[pre])
    ap.add_argument("--out", required=(gd.get("out") is None), default=gd.get("out"))
    ap.add_argument("--N", type=int, default=gd.get("N", 1200))
    ap.add_argument("--T", type=int, default=gd.get("T", 50))
    ap.add_argument("--seed", type=int, default=gd.get("seed", 0))
    args = ap.parse_args()

    X, E2, M, Y_ACC, Y_GYR = make_synth(N=args.N, T=args.T, seed=args.seed)
    split_and_save(X, E2, M, Y_ACC, Y_GYR, args.out)


if __name__ == "__main__":
    main()
