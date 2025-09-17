from __future__ import annotations
import argparse, os
from pathlib import Path
import numpy as np

def _get(arrs, keys, default=None):
    for k in keys:
        if k in arrs: 
            return arrs[k]
    return default

def convert_one(src_npz: Path, dst_npz: Path, split_routes: bool=False):
    d = np.load(src_npz, allow_pickle=True)
    X = _get(d, ["X","X_imu_seq","imu_seq","imu"])
    E2 = _get(d, ["E2","E2_sum","E2sum"])
    E = _get(d, ["E","E_imu","err","errors"])
    M = _get(d, ["MASK","y_mask","mask"])
    Y_ACC = _get(d, ["Y_ACC","Yacc","Y_acc"])
    Y_GYR = _get(d, ["Y_GYR","Ygyr","Y_gyr"])

    if X is None or M is None or (E2 is None and E is None):
        raise ValueError(f"{src_npz}: missing required keys")

    X = X.astype(np.float32)
    M = (M>0.5).astype(np.float32)

    if E2 is not None:
        E2 = E2.astype(np.float32)
        if E2.ndim == 2:
            E2 = E2[..., None]
    else:
        E = E.astype(np.float32)
        if E.shape[-1] >= 6:
            acc_e2 = np.sum(E[..., :3]**2, axis=-1, keepdims=True)
            gyr_e2 = np.sum(E[..., 3:6]**2, axis=-1, keepdims=True)
            E2 = np.concatenate([acc_e2, gyr_e2], axis=-1)
        else:
            E2 = np.sum(E**2, axis=-1, keepdims=True)

    out_dir = dst_npz.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(dst_npz, X=X, E2=E2, MASK=M,
             **({"Y_ACC":Y_ACC} if Y_ACC is not None else {}),
             **({"Y_GYR":Y_GYR} if Y_GYR is not None else {}))

    if split_routes and X.shape[-1] >= 6:
        acc_npz = dst_npz.with_name(dst_npz.stem + "_acc.npz")
        X_acc = X[..., :3]
        Y_acc = Y_ACC if Y_ACC is not None else None
        E2_acc = E2[..., 0:1] if E2.shape[-1] >= 1 else E2
        np.savez(acc_npz, X=X_acc, E2=E2_acc.astype(np.float32), MASK=M,
                 **({"Y_ACC":Y_acc} if Y_acc is not None else {}))

        gyr_npz = dst_npz.with_name(dst_npz.stem + "_gyr.npz")
        X_gyr = X[..., 3:6]
        Y_gyr = Y_GYR if Y_GYR is not None else None
        idx = 1 if E2.shape[-1] >= 2 else 0
        E2_gyr = E2[..., idx:idx+1]
        np.savez(gyr_npz, X=X_gyr, E2=E2_gyr.astype(np.float32), MASK=M,
                 **({"Y_GYR":Y_gyr} if Y_gyr is not None else {}))

def main():
    ap = argparse.ArgumentParser("Convert old NPZ to flat IMU-route format")
    ap.add_argument("--src", required=True, help="source .npz or directory")
    ap.add_argument("--dst", required=True, help="output .npz or directory")
    ap.add_argument("--split_routes", action="store_true", help="also write *_acc.npz and *_gyr.npz")
    args = ap.parse_args()

    src = Path(args.src); dst = Path(args.dst)
    if src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.glob("*.npz"):
            convert_one(f, dst / f.name, args.split_routes)
    else:
        if dst.is_dir():
            convert_one(src, dst / src.name, args.split_routes)
        else:
            convert_one(src, dst, args.split_routes)

if __name__ == "__main__":
    main()
