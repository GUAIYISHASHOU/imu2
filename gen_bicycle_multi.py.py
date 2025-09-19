#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_bicycle_multi.py
一次性生成 IMU + VIS + GNSS 三模态数据（严格共轨迹）。
- 100 Hz 自行车模型 master 轨迹
- IMU: 100 Hz 窗口化，导出 acc/gyr 两套 NPZ
- VIS: 10 Hz 窗口化（与你现有 vis 配置一致的风格），导出 2D 残差用的 NPZ
- GNSS: 1 Hz，支持可开关的 GARCH 异方差，导出 ENU 误差的 NPZ
- routes_meta: 可保存/读取每个 split/route 的随机种子，确保严格共轨迹
说明：
- 这里 VIS/IMU 的细节按“典型”实现给出（窗口化 + 噪声/外点）；若与你仓库已有定义略有出入，
  只需在标注位置替换为你现有的生成逻辑（变量名保持 X/Y/mask 即可）。
"""

from __future__ import annotations
import os, json, math, argparse, random
from pathlib import Path
import numpy as np

# ----------------- 小工具 -----------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def rolling_window_idx(N, T, S):
    out = []
    i = 0
    while i + T <= N:
        out.append((i, i+T))
        i += S
    return out

def windowize(X, Y, mask, T, S):
    N = X.shape[0]
    idx = rolling_window_idx(N, T, S)
    Xw = np.stack([X[a:b] for a,b in idx], axis=0) if idx else np.zeros((0,T,X.shape[1]), X.dtype)
    Yw = np.stack([Y[a:b] for a,b in idx], axis=0) if idx else np.zeros((0,T,Y.shape[1]), Y.dtype)
    Mw = np.stack([mask[a:b] for a,b in idx], axis=0) if idx else np.zeros((0,T,Y.shape[1]), bool)
    return Xw, Yw, Mw

# ----------------- 100 Hz 自行车 master 轨迹 -----------------
def sim_bicycle_traj(duration_s=500, hz=100, seed=42, wheelbase=2.7):
    set_seed(seed)
    dt = 1.0 / hz
    N = int(round(duration_s * hz))
    x = np.zeros((N, 4), dtype=np.float64)   # [px, py, yaw, v]
    x[0] = [0.0, 0.0, 0.0, 6.0]
    seg = max(1, int(8.0/dt))
    for k in range(N-1):
        if k % seg == 0:
            a = np.clip(np.random.randn()*0.3, -1.5, 1.5)
            delta = np.clip(np.random.randn()*0.15, -0.45, 0.45)
            ctrl = (a, delta)
        else:
            a, delta = ctrl
        px, py, yaw, v = x[k]
        v = max(0.5, v + a*dt)
        yaw = yaw + (v / wheelbase) * math.tan(delta) * dt
        px += v * math.cos(yaw) * dt
        py += v * math.sin(yaw) * dt
        x[k+1] = (px, py, yaw, v)
    t = np.arange(N)*dt
    U = 0.5*np.sin(0.005*t)  # 温和起伏
    gt_enu = np.column_stack([x[:,0], x[:,1], U])
    return {"t": t, "gt_enu": gt_enu, "yaw": x[:,2], "speed": x[:,3], "dt": dt, "hz": hz}

def downsample(traj, out_hz):
    step = int(round(traj["hz"]/out_hz))
    sel = np.arange(0, len(traj["t"]), step)
    out = {k: (v[sel] if isinstance(v, np.ndarray) and getattr(v, "ndim", 0)>0 else v) for k,v in traj.items()}
    out["hz"] = out_hz; out["dt"] = 1.0/out_hz; out["t"] = out["t"] - out["t"][0]
    return out

# ----------------- IMU 合成（简化版） -----------------
def imu_from_traj(tr, acc_noise=0.08, gyr_noise=0.005, hz=100, seed=0):
    """
    基于速度/航向的差分得到加速度/角速度（简化，不做重力/姿态旋转细节）
    你可以把这部分替换为你现有的 IMU 合成逻辑，以保持与原项目一致。
    """
    rng = np.random.default_rng(seed+101)
    dt = 1.0/hz
    yaw = tr["yaw"]; v = tr["speed"]
    # 角速度（z轴）
    gyr_z = np.zeros_like(yaw)
    gyr_z[1:] = (yaw[1:] - yaw[:-1]) / dt
    # 切向加速度近似
    ax = np.zeros_like(v); ay = np.zeros_like(v); az = np.zeros_like(v)
    ax[1:] = (v[1:] - v[:-1]) / dt
    # 噪声
    acc = np.stack([ax, ay, az], axis=-1) + rng.normal(0.0, acc_noise, size=(len(v),3))
    gyr = np.stack([np.zeros_like(gyr_z), np.zeros_like(gyr_z), gyr_z], axis=-1) \
          + rng.normal(0.0, gyr_noise, size=(len(v),3))
    return acc.astype(np.float32), gyr.astype(np.float32)

# ----------------- VIS 像素噪声/外点（占位，与你现有 vis 一致即可） -----------------
def vis_residuals(tr_10hz, noise_px=0.35, outlier_ratio=0.05, seed=0):
    """
    这里只给出一个 2D 残差“占位”合成：大多数小高斯，少量重尾。
    你可以把这块替换成你现有 gen_bicycle_dual_vis.py 里真实的视觉观测/匹配统计生成流程。
    """
    rng = np.random.default_rng(seed+303)
    T = tr_10hz["gt_enu"].shape[0]
    res = rng.normal(0.0, noise_px, size=(T,2)).astype(np.float32)
    mask_out = rng.random(T) < outlier_ratio
    if np.any(mask_out):
        res[mask_out] += rng.standard_t(df=3.0, size=(mask_out.sum(),2)).astype(np.float32) * (4.0*noise_px)
    return res  # (T,2)

# ----------------- GNSS：GARCH + 厂商 std -----------------
def garch_envelope(T, base_en=(0.7,0.7), base_u=1.8,
                   scene_bounds=(400,400,400,400,400),
                   scene_gain_en=(1.0,2.5,4.0,1.5,1.0),
                   scene_gain_u =(1.8,3.5,5.0,2.0,1.8),
                   omega=0.05, alpha=0.35, beta=0.45, seed=0, enable=True):
    set_seed(seed)
    sigE0, sigN0 = base_en; sigU0 = base_u
    g_en = np.zeros(T); g_u = np.zeros(T)
    idx = 0
    for L, ge, gu in zip(scene_bounds, scene_gain_en, scene_gain_u):
        r = slice(idx, min(T, idx+L)); g_en[r] = ge; g_u[r] = gu; idx += L
        if idx >= T: break
    if idx < T: g_en[idx:] = scene_gain_en[-1]; g_u[idx:] = scene_gain_u[-1]
    varE = np.zeros(T); varN = np.zeros(T); varU = np.zeros(T)
    varE[0] = (sigE0*g_en[0])**2; varN[0] = (sigN0*g_en[0])**2; varU[0] = (sigU0*g_u[0])**2
    eE_prev=eN_prev=eU_prev=0.0
    rng = np.random.default_rng(seed+123)
    for t in range(1,T):
        if enable:
            varE[t] = (omega + alpha*(eE_prev**2) + beta*varE[t-1]) * (g_en[t]**2)
            varN[t] = (omega + alpha*(eN_prev**2) + beta*varN[t-1]) * (g_en[t]**2)
            varU[t] = (omega + alpha*(eU_prev**2) + beta*varU[t-1]) * (g_u [t]**2)
        else:
            varE[t] = (sigE0*g_en[t])**2
            varN[t] = (sigN0*g_en[t])**2
            varU[t] = (sigU0*g_u [t])**2
        eE_prev = rng.normal(0.0, math.sqrt(varE[t]))
        eN_prev = rng.normal(0.0, math.sqrt(varN[t]))
        eU_prev = rng.normal(0.0, math.sqrt(varU[t]))
    return np.stack([np.sqrt(varE), np.sqrt(varN), np.sqrt(varU)], axis=-1)

def synth_vendor_std(true_sigma, bias=1.4, ln_jitter=0.2, seed=0):
    rng = np.random.default_rng(seed+999)
    ln_noise = rng.normal(0.0, ln_jitter, size=true_sigma.shape)
    return bias * true_sigma * np.exp(ln_noise)

def sample_gnss(gt_1hz, sigma_true, p_out=0.03, t_df=3.0, seed=0):
    rng = np.random.default_rng(seed+202)
    T = gt_1hz.shape[0]
    eps = rng.normal(0.0, sigma_true)
    mask_out = rng.random(T) < p_out
    if np.any(mask_out):
        scale = sigma_true[mask_out] * 6.0
        eps[mask_out] += rng.standard_t(df=t_df, size=scale.shape) * scale
    return gt_1hz + eps

def build_gns_features(tr_1hz, vendor):
    gt = tr_1hz["gt_enu"]
    dpos = np.zeros_like(gt); dpos[1:] = gt[1:] - gt[:-1]
    speed = tr_1hz["speed"]; yaw = tr_1hz["yaw"]
    dyaw = np.zeros_like(yaw); dyaw[1:] = yaw[1:] - yaw[:-1]
    base = np.column_stack([dpos, speed, dyaw])  # (T,5)
    feats = np.concatenate([vendor, base], axis=1)  # (T, 3+5) = 8
    return feats.astype(np.float32)

# ----------------- 主流程 -----------------
def main():
    ap = argparse.ArgumentParser()
    # 轨迹/split
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--traj_duration_s", type=int, default=2000)
    ap.add_argument("--rate_hz", type=int, default=100)
    ap.add_argument("--train_routes", type=int, default=6)
    ap.add_argument("--val_routes", type=int, default=2)
    ap.add_argument("--test_routes", type=int, default=2)

    # 输出目录
    ap.add_argument("--imu_out", default="data_cache")
    ap.add_argument("--vis_out", default="data_vis")
    ap.add_argument("--gns_out", default="data_gns")

    # IMU 窗口
    ap.add_argument("--imu_window", type=int, default=256)
    ap.add_argument("--imu_stride", type=int, default=128)

    # VIS 窗口与像素噪声（如果你已有实现，运行时这些参数不起决定作用）
    ap.add_argument("--vis_window", type=int, default=32)
    ap.add_argument("--vis_stride", type=int, default=16)
    ap.add_argument("--noise_px", type=float, default=0.35)
    ap.add_argument("--outlier_ratio", type=float, default=0.05)

    # GNSS 配置
    ap.add_argument("--gns_win", type=int, default=50)
    ap.add_argument("--gns_stride", type=int, default=25)
    ap.add_argument("--gns_arch_enable", action="store_true")
    ap.add_argument("--base_sigma_en", type=float, nargs=2, default=(0.7,0.7))
    ap.add_argument("--base_sigma_u", type=float, default=1.8)
    ap.add_argument("--scene_bounds", type=int, nargs="+", default=[400,400,400,400,400])
    ap.add_argument("--scene_gain_en", type=float, nargs="+", default=[1.0,2.5,4.0,1.5,1.0])
    ap.add_argument("--scene_gain_u",  type=float, nargs="+", default=[1.8,3.5,5.0,2.0,1.8])
    ap.add_argument("--omega", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--beta",  type=float, default=0.45)
    ap.add_argument("--p_out", type=float, default=0.03)
    ap.add_argument("--t_df", type=float, default=3.0)
    ap.add_argument("--vendor_bias", type=float, default=1.4)
    ap.add_argument("--vendor_ln_jitter", type=float, default=0.2)

    # 严格共轨迹
    ap.add_argument("--save_routes_meta", default=None)
    ap.add_argument("--routes_meta", default=None)
    args = ap.parse_args()

    ensure_dir(args.imu_out); ensure_dir(args.vis_out); ensure_dir(args.gns_out)

    splits = [("train", args.train_routes), ("val", args.val_routes), ("test", args.test_routes)]
    # 握手文件
    route_meta = {"seed": args.seed, "routes": {}}
    if args.routes_meta and Path(args.routes_meta).exists():
        route_meta = json.loads(Path(args.routes_meta).read_text())

    for split, R in splits:
        # 三模态容器
        ACC_Xs, ACC_Ys, ACC_Ms = [], [], []
        GYR_Xs, GYR_Ys, GYR_Ms = [], [], []
        VIS_Xs, VIS_Ys, VIS_Ms = [], [], []
        GNS_Xs, GNS_Ys, GNS_Ms = [], [], []

        for r in range(R):
            route_seed = route_meta.get("routes", {}).get(f"{split}_{r}",
                           args.seed + hash((split,r)) % 100000)
            if args.routes_meta is None:
                route_meta["routes"][f"{split}_{r}"] = route_seed

            # 100 Hz master 轨迹
            traj = sim_bicycle_traj(duration_s=args.traj_duration_s,
                                    hz=args.rate_hz, seed=route_seed)
            tr_100 = traj
            tr_10  = downsample(traj, out_hz=10)
            tr_1   = downsample(traj, out_hz=1)

            # ---------- IMU ----------
            acc, gyr = imu_from_traj(tr_100, seed=route_seed+11)
            # 这里用“预测方差学习”的思路：X 放入一些简单统计特征；Y=传感器误差（这里用噪声近似）
            # 你可以替换为你项目里 IMU 的真实标签构造。
            # 简单特征：滑窗内的均值/方差（占位）
            acc_feat = acc  # (N,3) —— 若你有更丰富的特征可在此拼接
            gyr_feat = gyr
            acc_err  = acc - np.zeros_like(acc)
            gyr_err  = gyr - np.zeros_like(gyr)
            acc_mask = np.ones_like(acc, dtype=bool)
            gyr_mask = np.ones_like(gyr, dtype=bool)

            ax, ay, am = windowize(acc_feat, acc_err, acc_mask,
                                   T=args.imu_window, S=args.imu_stride)
            gx, gy, gm = windowize(gyr_feat, gyr_err, gyr_mask,
                                   T=args.imu_window, S=args.imu_stride)
            ACC_Xs.append(ax); ACC_Ys.append(ay); ACC_Ms.append(am)
            GYR_Xs.append(gx); GYR_Ys.append(gy); GYR_Ms.append(gm)

            # ---------- VIS ----------
            # 若你已有稳定的 VIS 生成（匹配/重投影残差等），请在此替换为你现有逻辑
            vis_res = vis_residuals(tr_10, noise_px=args.noise_px,
                                    outlier_ratio=args.outlier_ratio, seed=route_seed+21)
            vis_feat = vis_res  # 简化：特征=残差本身（你可以拼接更多上下文）
            vis_err  = vis_res
            vis_mask = np.ones_like(vis_err, dtype=bool)
            vx, vy, vm = windowize(vis_feat, vis_err, vis_mask,
                                   T=args.vis_window, S=args.vis_stride)
            VIS_Xs.append(vx); VIS_Ys.append(vy); VIS_Ms.append(vm)

            # ---------- GNSS ----------
            T1 = tr_1["gt_enu"].shape[0]
            sigma_true = garch_envelope(
                T1, base_en=tuple(args.base_sigma_en), base_u=args.base_sigma_u,
                scene_bounds=tuple(args.scene_bounds),
                scene_gain_en=tuple(args.scene_gain_en),
                scene_gain_u=tuple(args.scene_gain_u),
                omega=args.omega, alpha=args.alpha, beta=args.beta,
                seed=route_seed+31, enable=bool(args.gns_arch_enable)
            )
            vendor = synth_vendor_std(sigma_true, bias=args.vendor_bias,
                                      ln_jitter=args.vendor_ln_jitter, seed=route_seed+41)
            y = sample_gnss(tr_1["gt_enu"], sigma_true,
                            p_out=args.p_out, t_df=args.t_df, seed=route_seed+51)
            gns_err = (y - tr_1["gt_enu"]).astype(np.float32)
            gns_mask = np.ones_like(gns_err, dtype=bool)
            gns_feat = build_gns_features(tr_1, vendor)
            gx_, gy_, gm_ = windowize(gns_feat, gns_err, gns_mask,
                                      T=args.gns_win, S=args.gns_stride)
            GNS_Xs.append(gx_); GNS_Ys.append(gy_); GNS_Ms.append(gm_)

        # 拼接并保存
        if ACC_Xs:
            X = np.concatenate(ACC_Xs); Y = np.concatenate(ACC_Ys); M = np.concatenate(ACC_Ms)
            np.savez_compressed(Path(args.imu_out)/f"{split}_acc.npz", X=X, Y=Y, mask=M)
            print(f"[{split}] acc  {X.shape} {Y.shape}")
        if GYR_Xs:
            X = np.concatenate(GYR_Xs); Y = np.concatenate(GYR_Ys); M = np.concatenate(GYR_Ms)
            np.savez_compressed(Path(args.imu_out)/f"{split}_gyr.npz", X=X, Y=Y, mask=M)
            print(f"[{split}] gyr  {X.shape} {Y.shape}")
        if VIS_Xs:
            X = np.concatenate(VIS_Xs); Y = np.concatenate(VIS_Ys); M = np.concatenate(VIS_Ms)
            np.savez_compressed(Path(args.vis_out)/f"{split}_vis.npz", X=X, Y=Y, mask=M)
            print(f"[{split}] vis  {X.shape} {Y.shape}")
        if GNS_Xs:
            X = np.concatenate(GNS_Xs); Y = np.concatenate(GNS_Ys); M = np.concatenate(GNS_Ms)
            np.savez_compressed(Path(args.gns_out)/f"{split}_gns.npz", X=X, Y=Y, mask=M,
                meta=json.dumps({"route":"gns","win":args.gns_win,"stride":args.gns_stride}))
            print(f"[{split}] gns  {X.shape} {Y.shape}")

    if args.save_routes_meta:
        Path(args.save_routes_meta).write_text(json.dumps(route_meta, indent=2))
        print("routes meta saved to", args.save_routes_meta)

if __name__ == "__main__":
    main()
