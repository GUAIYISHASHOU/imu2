#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_vis_health.py
验证 *_vis.npz 的视觉数据健康程度与“有效数据覆盖度”。

输入：
  --dir <folder>     扫描目录下的 train_vis.npz / val_vis.npz / test_vis.npz
  或
  --file <npz>       指定单个 npz

输出：
  - 终端打印汇总表
  - 若加 --plot，则在 <folder>/health_plots/ 下生成：
      coverage_hist_<split>.png
      e2_hist_<split>.png
"""

from __future__ import annotations
import argparse, os, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

FEATURE_NAMES = [
    "num_inlier_norm","flow_mag_mean","flow_mag_std","baseline_m",
    "yaw_rate","speed_proxy","roll","pitch"
]

def load_vis_npz(path: Path):
    """兼容两类命名：{X,E2,MASK} 或 {X_vis,E_vis,mask_vis}"""
    data = np.load(path, allow_pickle=False)
    keys = data.files
    if {"X","E2","MASK"}.issubset(keys):
        X  = data["X"];   E2 = data["E2"];   M  = data["MASK"]
    elif {"X_vis","E_vis","mask_vis"}.issubset(keys):
        X  = data["X_vis"];   E2 = data["E_vis"];   M  = data["mask_vis"]
    else:
        raise KeyError(f"{path.name}: 未找到需要的键，实际包含 {keys}")
    return X, E2, M

def pct(x): return 100.0*float(x)

def status_from_cov(overall_cov: float, zero_win_ratio: float):
    """给出 PASS/ATTN/FAIL 级别"""
    if overall_cov < 0.30 or zero_win_ratio > 0.15:
        return "FAIL"
    if overall_cov < 0.65 or zero_win_ratio > 0.05:
        return "ATTN"
    return "PASS"

def approx_unit_check_flow(flow_px_valid: np.ndarray):
    """返回告警文本（空字符串表示通过）"""
    if flow_px_valid.size == 0:
        return "flow_mean: no valid frames"
    med = float(np.median(flow_px_valid))
    if med < 0.05:
        return f"flow_mean median≈{med:.4f} px (too small?)"
    if med > 20.0:
        return f"flow_mean median≈{med:.2f} px (too large?)"
    return ""

def plot_hist(arr, title, out_png, bins=50, logx=False):
    arr = np.asarray(arr).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0: return
    plt.figure(figsize=(6,4))
    if logx:
        arr = arr[arr>0]
        plt.hist(arr, bins=bins, log=True)
        plt.xscale("log")
    else:
        plt.hist(arr, bins=bins)
    plt.title(title); plt.xlabel(title); plt.ylabel("count")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()

def try_load_seg(dirpath: Path, split: str):
    """尽量载入分段标签（优先 seg_id_*.npy；其次 synth_vis/*.npz）"""
    npy = dirpath / f"{split}_seg_id.npy"
    if npy.exists():
        return np.load(npy)
    npz = dirpath / "synth_vis" / f"{split}.npz"
    if npz.exists():
        d = np.load(npz, allow_pickle=False)
        if "seg_id" in d.files:
            return d["seg_id"]
    return None

def analyze_one(npz_path: Path, outdir_plots: Path | None = None):
    split = npz_path.stem.replace("_vis","")  # train/val/test
    X, E2, M = load_vis_npz(npz_path)

    # 基本形状
    N, W = X.shape[0], X.shape[1]
    D = X.shape[2] if X.ndim == 3 else 0

    # 覆盖度（窗口 & 全局）
    cov_win = M.mean(axis=1)                       # 每窗覆盖度
    cov_all = float(M.mean())                      # 全体帧覆盖度
    zero_win = np.sum(cov_win == 0.0)
    zero_win_ratio = float(zero_win) / max(1, N)

    # NaN/Inf 检查
    n_nan = int(np.isnan(X).sum() + np.isnan(E2).sum() + np.isnan(M).sum())
    n_inf = int(np.isinf(X).sum() + np.isinf(E2).sum() + np.isinf(M).sum())

    # 视觉量纲自检（光流均值）
    flow_warn = ""
    if D >= 2:
        flow_mean = X[..., 1]   # feature[1] = flow_mag_mean
        flow_valid = flow_mean[M > 0.5]
        flow_warn = approx_unit_check_flow(flow_valid)

    # E2 基本分布（只看有效帧）
    E2_valid = E2[..., 0][M > 0.5] if (E2.ndim == 3 and E2.shape[-1] == 1) else E2[M > 0.5]
    E2_stats = dict(
        median=float(np.median(E2_valid)) if E2_valid.size else float("nan"),
        p75=float(np.percentile(E2_valid, 75)) if E2_valid.size else float("nan"),
        p95=float(np.percentile(E2_valid, 95)) if E2_valid.size else float("nan"),
    )

    # 分段标签（若有）
    seg = try_load_seg(npz_path.parent, split)
    seg_info = None
    if seg is not None:
        # 约定：0=normal,1=pure_rot,2=low_parallax,3=inlier_drop
        uniq, cnt = np.unique(seg, return_counts=True)
        total = int(cnt.sum())
        seg_info = {int(u): {"count": int(c), "ratio": float(c)/total} for u, c in zip(uniq, cnt)}

    # 绘图
    if outdir_plots is not None:
        plot_hist(cov_win, f"{split} coverage per window", outdir_plots / f"coverage_hist_{split}.png", bins=40)
        plot_hist(E2_valid[E2_valid>0], f"{split} E2 (valid frames, log)", outdir_plots / f"e2_hist_{split}.png", bins=60, logx=True)

    # 级别
    level = status_from_cov(cov_all, zero_win_ratio)

    # 汇总字典
    summary = {
        "split": split,
        "file": npz_path.name,
        "windows": int(N),
        "win_size": int(W),
        "features": int(D),
        "coverage_overall": cov_all,             # 所有帧的平均有效率
        "coverage_win_mean": float(cov_win.mean()),
        "coverage_win_p25": float(np.percentile(cov_win, 25)),
        "coverage_win_p50": float(np.percentile(cov_win, 50)),
        "coverage_win_p75": float(np.percentile(cov_win, 75)),
        "win_zero_count": int(zero_win),
        "win_zero_ratio": zero_win_ratio,
        "nan_count": n_nan,
        "inf_count": n_inf,
        "flow_warn": flow_warn,
        "E2_stats": E2_stats,
        "seg_info": seg_info,
        "level": level,
    }
    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir",  type=str, help="包含 *_vis.npz 的目录")
    ap.add_argument("--file", type=str, help="单个 vis npz 文件路径")
    ap.add_argument("--plot", action="store_true", help="保存覆盖度/E2 直方图")
    args = ap.parse_args()

    targets = []
    if args.file:
        targets = [Path(args.file)]
        root = Path(args.file).parent
    else:
        root = Path(args.dir) if args.dir else Path(".")
        targets = sorted([p for p in root.glob("*_vis.npz") if p.is_file()])

    if not targets:
        print("未找到 *_vis.npz")
        return

    plot_dir = (root / "health_plots") if args.plot else None
    summaries = [analyze_one(p, plot_dir) for p in targets]

    # 打印表格
    print("\n=== VIS Health Summary ===")
    header = ("split", "windows", "win", "cov_all", "zero_win%", "NaN", "Inf", "E2_med", "level")
    print("{:>6} | {:>7} | {:>3} | {:>7} | {:>9} | {:>4} | {:>3} | {:>8} | {:>5}".format(*header))
    for s in summaries:
        e2med = s["E2_stats"]["median"]
        print("{:>6} | {:>7} | {:>3} | {:>7.3f} | {:>8.2f}% | {:>4} | {:>3} | {:>8.3g} | {:>5}".format(
            s["split"], s["windows"], s["win_size"], s["coverage_overall"],
            100.0*s["win_zero_ratio"], s["nan_count"], s["inf_count"],
            (e2med if math.isfinite(e2med) else float('nan')), s["level"]
        ))

    # 详细提示
    for s in summaries:
        warn = s["flow_warn"]
        if warn:
            print(f"[{s['split']}] WARN: {warn}")
        if s["seg_info"]:
            ratio = s["seg_info"]
            def pr(k): 
                r = ratio.get(k, {"count":0,"ratio":0.0})
                return f"{r['count']}/{r['ratio']*100:.1f}%"
            print(f"[{s['split']}] seg: normal={pr(0)}, pure_rot={pr(1)}, low_parallax={pr(2)}, inlier_drop={pr(3)}")

    if args.plot:
        print(f"\n图已保存到: {plot_dir}")

if __name__ == "__main__":
    main()
