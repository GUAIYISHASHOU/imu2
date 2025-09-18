#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
from utils import load_config_file

# -------------------- 线性代数小工具 --------------------
def skew(v):
    x,y,z = v
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0]], dtype=np.float32)

def rot_z(yaw):
    c,s = np.cos(yaw), np.sin(yaw)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)

def rot_y(pitch):
    c,s = np.cos(pitch), np.sin(pitch)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float32)

def rot_x(roll):
    c,s = np.cos(roll), np.sin(roll)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float32)

# -------------------- 自行车/独轮车轨迹 + IMU 噪声 --------------------
def bicycle_traj(T: int, dt: float, seed: int,
                 use_slip=False, use_gravity=True, use_roll_pitch=True,
                 bank_gain=1.0, pitch_gain=1.0):
    rng = np.random.default_rng(seed)
    x = y = yaw = 0.0
    v = 5.0
    g = 9.81

    acc_true = np.zeros((T,3), np.float32)
    gyr_true = np.zeros((T,3), np.float32)
    roll = np.zeros(T, np.float32)
    pitch= np.zeros(T, np.float32)
    speed= np.zeros(T, np.float32)

    t = np.arange(T) * dt
    a_var = 0.20*(1.0 + 0.7*np.sin(0.60*t) + 0.30*rng.normal(size=T))
    g_var = 0.05*(1.0 + 0.8*np.cos(0.40*t+0.5) + 0.30*rng.normal(size=T))
    a_var = np.clip(a_var, 1e-5, 5.0).astype(np.float32)
    g_var = np.clip(g_var, 1e-6, 1.0).astype(np.float32)

    slip = 1.0
    if use_slip:
        slip = np.clip(0.9 + 0.1*np.sin(0.003*np.arange(T)), 0.8, 1.1)

    for k in range(T):
        omega = 0.20*np.sin(0.10*k)             # yaw rate
        a_cmd = 0.50*np.sin(0.07*k)             # tangential accel

        v = float(np.clip(v + a_cmd*dt, 0.1, 20.0))
        yaw = yaw + omega*dt
        x = x + v*np.cos(yaw)*dt
        y = y + v*np.sin(yaw)*dt

        ax = a_cmd
        ay = (v*omega) * (slip[k] if isinstance(slip, np.ndarray) else slip)
        az = 0.0

        # 近似 roll/pitch（小角）：roll≈ay/g, pitch≈-ax/g
        if use_roll_pitch:
            roll[k]  = bank_gain  * (ay/g)
            pitch[k] = -pitch_gain * (ax/g)

        acc_true[k] = [ax,ay,az]
        gyr_true[k] = [0.0,0.0,omega]
        speed[k]    = v

    if use_gravity:
        c_r, s_r = np.cos(roll), np.sin(roll)
        c_p, s_p = np.cos(pitch), np.sin(pitch)
        gx_b = -g * s_p
        gy_b =  g * s_r
        gz_b =  g * (c_p * np.cos(roll))  # 小角近似可直接用 1
        grav = np.stack([gx_b, gy_b, gz_b], axis=-1).astype(np.float32)
        acc_true = acc_true - grav

    return acc_true, gyr_true, a_var, g_var, roll, pitch, speed

def simulate_imu(T, dt, seed, **phys):
    rng = np.random.default_rng(seed)
    acc_true, gyr_true, a_var, g_var, roll, pitch, speed = bicycle_traj(T, dt, seed, **phys)
    acc_noise = rng.normal(scale=np.sqrt(a_var)[:,None], size=(T,3)).astype(np.float32)
    gyr_noise = rng.normal(scale=np.sqrt(g_var)[:,None], size=(T,3)).astype(np.float32)
    acc_meas = acc_true + acc_noise
    gyr_meas = gyr_true + gyr_noise

    X_imu = np.concatenate([acc_meas, gyr_meas], axis=-1).astype(np.float32)   # (T,6)
    E2_imu= np.stack([np.sum(acc_noise**2,axis=-1), np.sum(gyr_noise**2,axis=-1)], axis=-1).astype(np.float32)  # (T,2)
    return X_imu, E2_imu, a_var, g_var, roll, pitch, speed

# -------------------- 相机/视觉仿真 --------------------
def sample_landmarks(num=3000, box=(( -80, 80), ( -30, 30), (  2, 80)), seed=0):
    """
    均匀撒点（世界系），默认在车前方一个长盒子里（避免身后无意义点）
    z in [2, 80] 表示“前方距离”近远
    """
    rng = np.random.default_rng(seed)
    xs = rng.uniform(box[0][0], box[0][1], size=num)
    ys = rng.uniform(box[1][0], box[1][1], size=num)
    zs = rng.uniform(box[2][0], box[2][1], size=num)
    Pw = np.stack([xs, ys, zs], axis=-1).astype(np.float32)  # (N,3)
    return Pw

def camera_poses_from_imu(yaw, roll, pitch, trans_xy, z_height,
                          R_cb=np.eye(3,dtype=np.float32), t_cb=np.zeros(3,np.float32)):
    """
    由 IMU 的 (x,y,yaw,roll,pitch) 得到相机位姿（世界到相机的 SE3）
    简化：世界系 z 朝上；车体在 z=常数 平面行驶
    """
    T = len(yaw)
    Rc_list = []
    tc_list = []
    for k in range(T):
        Rwb = rot_z(yaw[k]) @ rot_y(pitch[k]) @ rot_x(roll[k])   # world->body
        Rwc = Rwb @ R_cb                                         # world->cam
        pwb = np.array([trans_xy[k,0], trans_xy[k,1], z_height], np.float32)
        pwc = pwb + Rwb @ t_cb                                   # cam center in world
        Rcw = Rwc.T
        tcw = -Rcw @ pwc
        Rc_list.append(Rcw)
        tc_list.append(tcw.astype(np.float32))
    return np.stack(Rc_list,0), np.stack(tc_list,0)  # (T,3,3),(T,3)

def project_points(Pw, Rcw, tcw, K, img_wh, noise_px=0.5, rng=None):
    """
    把世界点投到像素系，返回可见点及像素。
    """
    if rng is None: rng = np.random.default_rng(0)
    Pc = (Rcw @ Pw.T).T + tcw   # (N,3)
    Z = Pc[:,2]
    vis = Z > 0.3
    Pc = Pc[vis]
    if Pc.shape[0]==0:
        return np.zeros((0,2),np.float32), np.zeros((0,),bool), np.zeros((0,3),np.float32)
    uv = (K @ (Pc.T / Pc[:,2])).T[:, :2]  # 像素
    W, H = img_wh
    in_img = (uv[:,0] >= 0) & (uv[:,0] < W) & (uv[:,1] >= 0) & (uv[:,1] < H)
    uv = uv[in_img]
    Pc = Pc[in_img]
    uv_noisy = uv + rng.normal(scale=noise_px, size=uv.shape).astype(np.float32)
    return uv_noisy.astype(np.float32), in_img.nonzero()[0], Pc.astype(np.float32)

def sampson_dist(x1n, x2n, E):
    """
    x1n,x2n: (M,2) 归一化像素坐标（K^-1 u）
    返回 Sampson distance 的平方（M,）
    """
    x1 = np.concatenate([x1n, np.ones((x1n.shape[0],1),np.float32)], axis=1)  # (M,3)
    x2 = np.concatenate([x2n, np.ones((x2n.shape[0],1),np.float32)], axis=1)
    Ex1 = (E @ x1.T).T
    Etx2= (E.T @ x2.T).T
    x2tEx1 = np.sum(x2 * (E @ x1.T).T, axis=1)
    num = x2tEx1**2
    den = Ex1[:,0]**2 + Ex1[:,1]**2 + Etx2[:,0]**2 + Etx2[:,1]**2 + 1e-12
    return (num / den).astype(np.float32)  # (M,)

def simulate_vision_from_trajectory(T_cam, t_cam_idx, yaw, roll, pitch, xy,     # 轨迹（下采样到相机时刻的索引）
                                    K, img_wh, Pw, noise_px=0.5, outlier_ratio=0.1,
                                    min_match=20, seed=0):
    """
    基于轨迹位姿 + 3D 地图，仿真相机观测与相邻帧匹配，并计算每帧 E2_vis。
    返回：
      E2_vis: (T_cam,)  每帧（与上一帧）Sampson^2 的和（若匹配不足则置 0，mask=0）
      X_vis:  (T_cam, D) 每帧特征（num_inliers_norm, mean_flow_px, std_flow_px, baseline_norm, yaw_rate, speed, roll, pitch）
      MASK:   (T_cam,)  有效帧掩码（首帧或匹配不足置 0）
    """
    rng = np.random.default_rng(seed)
    # 相机外参（默认与体坐标重合，可按需改）
    R_cb = np.eye(3, dtype=np.float32)
    t_cb = np.zeros(3, dtype=np.float32)
    # 相机位姿（世界->相机）
    Rcw_all, tcw_all = camera_poses_from_imu(yaw[t_cam_idx], roll[t_cam_idx], pitch[t_cam_idx],
                                             xy[t_cam_idx], z_height=1.2, R_cb=R_cb, t_cb=t_cb)
    # 像素到归一化坐标
    Kinv = np.linalg.inv(K).astype(np.float32)

    # 对每帧投影
    UV = []
    idlists = []
    Pc_list = []
    for k in range(T_cam):
        uv, id_in_img, Pc = project_points(Pw, Rcw_all[k], tcw_all[k], K, img_wh, noise_px=noise_px, rng=rng)
        UV.append(uv)
        idlists.append(id_in_img)  # 这些是 Pw 的索引子集
        Pc_list.append(Pc)

    E2_vis = np.zeros(T_cam, np.float32)
    X_vis  = np.zeros((T_cam,8), np.float32)  # 8D 状态特征
    MASK   = np.zeros(T_cam, np.float32)

    # 临时：构造 yaw_rate/speed/roll/pitch（简单差分/带入外部）
    # 这里假设外部传进来的是 100Hz 的 yaw/speed/roll/pitch（与 IMU 同频）；我们用相机索引取子序列
    # yaw_rate: 差分（相机频率上）
    yaw_cam = yaw[t_cam_idx]; speed_cam = np.gradient(xy[t_cam_idx,0], edge_order=1) * 0  # 占位，后面替换
    xy_cam = xy[t_cam_idx]
    speed_cam = np.linalg.norm(np.diff(xy_cam, axis=0, prepend=xy_cam[:1]), axis=1)  # 粗略速度像素/帧
    yaw_rate_cam = np.diff(yaw_cam, prepend=yaw_cam[:1]) / max(1, (t_cam_idx[1]-t_cam_idx[0]))
    roll_cam = roll[t_cam_idx]; pitch_cam = pitch[t_cam_idx]

    # 相邻帧匹配与 Sampson
    for k in range(T_cam):
        if k == 0:
            MASK[k] = 0.0
            continue
        # 上一帧 / 当前帧的可见点索引（在 Pw 中的全局 id）
        ids_prev = idlists[k-1]; ids_curr = idlists[k]
        # 取交集，实现“真值匹配”
        common = np.intersect1d(ids_prev, ids_curr)
        if common.size < min_match:
            MASK[k] = 0.0
            continue
        # 从两帧里取出这些点的像素观测
        # 需要把局部列表下标映射回交集的相对位置
        def pick_uv(UV_list, idlist, common_ids):
            pos = {gid:i for i,gid in enumerate(idlist)}
            idx = [pos[g] for g in common_ids]
            return UV_list[idx]
        uv1 = pick_uv(UV[k-1], ids_prev, common)
        uv2 = pick_uv(UV[k],   ids_curr, common)

        # 注入外点
        M = uv1.shape[0]
        m_out = int(M * outlier_ratio)
        if m_out > 0:
            rnd = rng.choice(M, size=m_out, replace=False)
            uv2[rnd] += rng.normal(scale=20.0, size=(m_out,2)).astype(np.float32)

        # 归一化坐标
        x1n = (Kinv @ np.concatenate([uv1, np.ones((M,1),np.float32)], axis=1).T).T[:, :2]
        x2n = (Kinv @ np.concatenate([uv2, np.ones((M,1),np.float32)], axis=1).T).T[:, :2]

        # 用真值位姿构造本质矩阵 E = [t]_x R（相机坐标系）
        R1, t1 = Rcw_all[k-1], tcw_all[k-1]
        R2, t2 = Rcw_all[k],   tcw_all[k]
        R_rel = R2 @ R1.T
        t_rel = t2 - (R_rel @ t1)
        E = skew(t_rel) @ R_rel

        d2 = sampson_dist(x1n, x2n, E)  # (M,)
        # 统计 & 特征
        flow = np.linalg.norm(uv2 - uv1, axis=1)
        E2_vis[k] = float(np.sum(d2))
        num_inl = float(M)
        X_vis[k] = np.array([
            num_inl / 500.0,                 # 归一化匹配数（500 可按数据量调整）
            float(np.mean(flow)),
            float(np.std(flow)),
            float(np.linalg.norm(t_rel)),    # baseline_norm: 相邻帧基线的范数（相机尺度）
            float(yaw_rate_cam[k]),          # 简易 yaw_rate 代理
            float(speed_cam[k]),             # 简易速度代理（像素/帧），可改成物理速度
            float(roll_cam[k]),
            float(pitch_cam[k]),
        ], np.float32)
        MASK[k] = 1.0

    return E2_vis, X_vis, MASK

# -------------------- 滑窗 --------------------
def window_count(T: int, win: int, stride: int) -> int:
    return 0 if T < win else 1 + (T - win) // stride

def sliding_window(arr: np.ndarray, win: int, stride: int):
    T = arr.shape[0]
    n = window_count(T, win, stride)
    if n == 0:  return np.zeros((0, win) + arr.shape[1:], dtype=arr.dtype)
    return np.stack([arr[i*stride:i*stride+win] for i in range(n)], axis=0)

# -------------------- 主流程 --------------------
def make_splits(out_dir: Path,
                # 公共：长轨迹
                traj_duration_s: float, rate_hz: float, seed: int,
                train_routes:int, val_routes:int, test_routes:int,
                # 物理项
                use_slip:bool, use_gravity:bool, use_roll_pitch:bool,
                bank_gain:float, pitch_gain:float,
                # IMU 路由配置
                acc_win:int, acc_str:int, acc_preproc:str, acc_ma:int,
                gyr_win:int, gyr_str:int, gyr_preproc:str, gyr_ma:int,
                # 视觉配置
                cam_rate_hz: float, img_w:int, img_h:int, fx:float, fy:float, cx:float, cy:float,
                vis_win:int, vis_str:int, noise_px:float, outlier_ratio:float, min_match:int):

    def preprocess(x_long, mode, ma_len):
        if mode == "raw": return x_long
        if mode == "ma_residual":
            if ma_len <= 1: return x_long
            pad = ma_len//2
            xp = np.pad(x_long, ((pad,pad),(0,0)), mode="reflect")
            ker = np.ones((ma_len,1), np.float32) / ma_len
            out = np.zeros_like(xp, np.float32)
            for c in range(x_long.shape[1]):
                out[:,c] = np.convolve(xp[:,c], ker[:,0], mode="same")
            return (x_long - out[pad:-pad]).astype(np.float32)
        if mode == "diff":
            return np.diff(x_long, axis=0, prepend=x_long[:1]).astype(np.float32)
        raise ValueError(mode)

    # ---- 轻量告警/元数据 ----
    def warn(msg):
        print(f"[vis-gen][WARN] {msg}")

    def approx_unit_check_flow(flow_mean_px, tag="flow_mean"):
        arr = np.asarray(flow_mean_px)
        if arr.size == 0:
            print(f"[vis-gen][WARN] {tag}: no valid frames")
            return
        med = float(np.median(arr))
        if med < 0.05:
            print(f"[vis-gen][WARN] {tag} median≈{med:.4f} px (too small?)")
        elif med > 20.0:
            print(f"[vis-gen][WARN] {tag} median≈{med:.2f} px (too large?)")

    def write_vis_meta(out_dir_meta: Path):
        meta = {
            "unit": "px",
            "feature_names": [
                "num_inlier_norm","flow_mag_mean","flow_mag_std","baseline_m",
                "yaw_rate","speed_proxy","roll","pitch"
            ],
            "quantile_triplets": [],
            "standardize": {"enable": False, "mean": None, "std": None},
        }
        out_dir_meta.mkdir(parents=True, exist_ok=True)
        (out_dir_meta / "vis_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    def one_split(num_routes, split_name, seed_base):
        dt = 1.0 / rate_hz
        T_long = int(round(traj_duration_s * rate_hz))
        # 相机采样索引（等间隔下采样）
        cam_step = int(round(rate_hz / cam_rate_hz))
        t_cam_idx = np.arange(0, T_long, cam_step, dtype=np.int32)
        T_cam = len(t_cam_idx)

        # 地图点（共享）
        Pw = sample_landmarks(num=4000, seed=seed_base+77)

        Xa_list,Ea_list,Ma_list,YAa_list = [],[],[],[]
        Xg_list,Eg_list,Mg_list,YGg_list = [],[],[],[]
        Xv_list,Ev_list,Mv_list = [],[],[]
        seg_list = []  # (per-route) camera timeline labels

        for r in range(num_routes):
            seed_r = seed_base + r
            # 生成 IMU 长序列
            X_imu, E2_imu, Yacc, Ygyr, roll, pitch, speed = simulate_imu(T_long, dt, seed_r,
                                                                          use_slip=use_slip,
                                                                          use_gravity=use_gravity,
                                                                          use_roll_pitch=use_roll_pitch,
                                                                          bank_gain=bank_gain, pitch_gain=pitch_gain)
            # （简单）车体 xy 与 yaw：从速度/航向推演或记录；这里快速近似：
            yaw = np.cumsum(X_imu[:,5]) * dt    # 积分 gz 作为 yaw（仅用于视觉状态）
            xy  = np.cumsum(np.stack([speed*np.cos(yaw), speed*np.sin(yaw)],axis=-1), axis=0) * dt

            # 视觉：从轨迹仿真
            K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], np.float32)
            E2_vis, X_vis, M_vis = simulate_vision_from_trajectory(
                T_cam, t_cam_idx, yaw, roll, pitch, xy, K, (img_w,img_h), Pw,
                noise_px=noise_px, outlier_ratio=outlier_ratio, min_match=min_match, seed=seed_r+999
            )

            # ---- 轻量自检与段标 ----
            # 有效覆盖率与单位量级检查（基于光流均值，像素）
            valid = (M_vis > 0.5).reshape(-1)
            cov = float(valid.mean()) if valid.size else 0.0
            print(f"[{split_name}] vis_coverage={cov:.3f}")
            approx_unit_check_flow(X_vis[valid, 1] if valid.any() else np.array([]), tag=f"{split_name}/route{r}/flow_mean")
            # 段落标注（启发式）：1=纯旋(转动大/基线小)，2=弱视差(流量小&基线小)，3=内点下降(内点比小)
            seg_id = np.zeros((T_cam,), dtype=np.int32)
            baseline = X_vis[:,3]
            yaw_rate = np.abs(X_vis[:,4])
            flow_mean= X_vis[:,1]
            inlier_norm = X_vis[:,0]
            # 阈值用分位数适配
            b_small = np.quantile(baseline, 0.1) if T_cam>0 else 0.0
            f_small = np.quantile(flow_mean, 0.1) if T_cam>0 else 0.0
            ir_small= np.quantile(inlier_norm, 0.1) if T_cam>0 else 0.0
            rot_ratio = yaw_rate / (baseline + 1e-6)
            rr_big = np.quantile(rot_ratio, 0.9) if T_cam>0 else 1e9
            # 标注（优先级：内点下降>纯旋>弱视差）
            seg_id[inlier_norm <= ir_small] = 3
            mask_free = seg_id == 0
            seg_id[(rot_ratio >= rr_big) & mask_free] = 1
            mask_free = seg_id == 0
            seg_id[((baseline <= b_small) & (flow_mean <= f_small)) & mask_free] = 2
            seg_list.append(seg_id.astype(np.int32))

            # ---- IMU 分路：ACC ----
            Xa_long = preprocess(X_imu[:, :3], acc_preproc, acc_ma)
            Ea_long = E2_imu[:, [0]]
            Xa = sliding_window(Xa_long, acc_win, acc_str)
            Ea = sliding_window(Ea_long,  acc_win, acc_str)
            Ma = np.ones((Xa.shape[0], Xa.shape[1]), np.float32)
            YAa= sliding_window(Yacc,     acc_win, acc_str)
            Xa_list.append(Xa); Ea_list.append(Ea); Ma_list.append(Ma); YAa_list.append(YAa)

            # ---- IMU 分路：GYR ----
            Xg_long = preprocess(X_imu[:, 3:6], gyr_preproc, gyr_ma)
            Eg_long = E2_imu[:, [1]]
            Xg = sliding_window(Xg_long, gyr_win, gyr_str)
            Eg = sliding_window(Eg_long,  gyr_win, gyr_str)
            Mg = np.ones((Xg.shape[0], Xg.shape[1]), np.float32)
            YGg= sliding_window(Ygyr,     gyr_win, gyr_str)
            Xg_list.append(Xg); Eg_list.append(Eg); Mg_list.append(Mg); YGg_list.append(YGg)

            # ---- VIS 分路（相机频率窗口）----
            Xv = sliding_window(X_vis,         vis_win, vis_str)
            Ev = sliding_window(E2_vis[:,None],vis_win, vis_str)
            Mv = sliding_window(M_vis[:,None], vis_win, vis_str)[:, :, 0]
            Xv_list.append(Xv); Ev_list.append(Ev); Mv_list.append(Mv)

        # 拼接
        Xa  = np.concatenate(Xa_list,0).astype(np.float32)
        Ea  = np.concatenate(Ea_list,0).astype(np.float32)
        Ma  = np.concatenate(Ma_list,0).astype(np.float32)
        YAa = np.concatenate(YAa_list,0).astype(np.float32)

        Xg  = np.concatenate(Xg_list,0).astype(np.float32)
        Eg  = np.concatenate(Eg_list,0).astype(np.float32)
        Mg  = np.concatenate(Mg_list,0).astype(np.float32)
        YGg = np.concatenate(YGg_list,0).astype(np.float32)

        Xv  = np.concatenate(Xv_list,0).astype(np.float32)
        Ev  = np.concatenate(Ev_list,0).astype(np.float32)
        Mv  = np.concatenate(Mv_list,0).astype(np.float32)

        # 写出分割标签（相机时间轴，按 route 级拼接）
        seg_all = np.concatenate(seg_list, axis=0) if len(seg_list)>0 else np.zeros((0,),np.int32)
        np.save(out_dir / f"{split_name}_seg_id.npy", seg_all.astype(np.int32))

        print(f"[{split_name}] routes={num_routes} | ACC windows={Xa.shape[0]} | GYR windows={Xg.shape[0]} | VIS windows={Xv.shape[0]}")
        return (Xa,Ea,Ma,YAa),(Xg,Eg,Mg,YGg),(Xv,Ev,Mv)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Train / Val / Test
    acc_tr,gyr_tr,vis_tr = one_split(train_routes, "train", seed+1000)
    acc_va,gyr_va,vis_va = one_split(val_routes,   "val",   seed+2000)
    acc_te,gyr_te,vis_te = one_split(test_routes,  "test",  seed+3000)

    # 保存
    def savetag(prefix, acc, gyr, vis):
        Xa,Ea,Ma,YAa = acc
        Xg,Eg,Mg,YGg = gyr
        Xv,Ev,Mv     = vis
        np.savez(out_dir/f"{prefix}_acc.npz", X=Xa, E2=Ea, MASK=Ma, Y_ACC=YAa)
        np.savez(out_dir/f"{prefix}_gyr.npz", X=Xg, E2=Eg, MASK=Mg, Y_GYR=YGg)
        np.savez(out_dir/f"{prefix}_vis.npz", X=Xv, E2=Ev, MASK=Mv)

    # 写一次元数据
    write_vis_meta(out_dir)

    savetag("train", acc_tr, gyr_tr, vis_tr)
    savetag("val",   acc_va, gyr_va, vis_va)
    savetag("test",  acc_te, gyr_te, vis_te)

    # ---- 追加：VIS 端元数据与分段标签（最小自检产物） ----
    synth_vis_dir = out_dir / "synth_vis"
    synth_vis_dir.mkdir(parents=True, exist_ok=True)

    # 拆包 VIS 三路
    Xv_tr, Ev_tr, Mv_tr = vis_tr
    Xv_va, Ev_va, Mv_va = vis_va
    Xv_te, Ev_te, Mv_te = vis_te

    # 训练集统计（用于标准化/一致性校验）
    train_mean = np.mean(Xv_tr.reshape(-1, Xv_tr.shape[-1]), axis=0).astype(np.float32) if Xv_tr.size>0 else np.zeros((Xv_tr.shape[-1],), np.float32)
    train_std  = np.std( Xv_tr.reshape(-1, Xv_tr.shape[-1]), axis=0).astype(np.float32) + 1e-12

    # 读取各 split seg_id（前面已保存 per-split 标签）
    seg_id_train = np.load(out_dir / "train_seg_id.npy") if (out_dir/"train_seg_id.npy").exists() else np.zeros((Xv_tr.shape[0]*Xv_tr.shape[1],), np.int32)
    seg_id_val   = np.load(out_dir / "val_seg_id.npy")   if (out_dir/"val_seg_id.npy").exists()   else np.zeros((Xv_va.shape[0]*Xv_va.shape[1],), np.int32)
    seg_id_test  = np.load(out_dir / "test_seg_id.npy")  if (out_dir/"test_seg_id.npy").exists()  else np.zeros((Xv_te.shape[0]*Xv_te.shape[1],), np.int32)

    # vis_meta.json（按当前 X_vis 列定义）
    meta = {
        "unit": "px",
        "feature_names": [
            "num_inlier_norm","flow_mag_mean","flow_mag_std","baseline_m",
            "yaw_rate","speed_proxy","roll","pitch"
        ],
        "standardize": {
            "enable": True,
            "mean": train_mean.tolist(),
            "std":  train_std.tolist()
        },
        "random": {
            "base_seed": seed,
            "seeds": {"train": seed+1000, "val": seed+2000, "test": seed+3000},
            "target_cover": {"pure_rot":0.15,"low_parallax":0.20,"inlier_drop":0.10},
            "dur_s": [0.8, 2.0],
            "cooldown_s": 0.3
        }
    }
    (synth_vis_dir / "vis_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # 分段标签另存一份到 synth_vis
    np.save(synth_vis_dir / "seg_id_train.npy", seg_id_train.astype(np.int32))
    np.save(synth_vis_dir / "seg_id_val.npy",   seg_id_val.astype(np.int32))
    np.save(synth_vis_dir / "seg_id_test.npy",  seg_id_test.astype(np.int32))

    # 可选：把 seg_id 内嵌进单独的 VIS npz（便于单文件分析）
    np.savez(synth_vis_dir/"train.npz", X_vis=Xv_tr, E_vis=Ev_tr, mask_vis=Mv_tr, seg_id=seg_id_train)
    np.savez(synth_vis_dir/"val.npz",   X_vis=Xv_va, E_vis=Ev_va, mask_vis=Mv_va, seg_id=seg_id_val)
    np.savez(synth_vis_dir/"test.npz",  X_vis=Xv_te, E_vis=Ev_te, mask_vis=Mv_te, seg_id=seg_id_test)

def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None, help="YAML/JSON 配置文件（读取 vis 段）")
    args_pre, _ = pre.parse_known_args()

    cfg = load_config_file(args_pre.config)
    vis = cfg.get("vis", {})

    ap = argparse.ArgumentParser("One-shot sim: ACC/GYR/VIS from a shared bicycle trajectory (two-quantity supervision)", parents=[pre])
    ap.add_argument("--out", required=(vis.get("out") is None), default=vis.get("out"))
    ap.add_argument("--seed", type=int, default=vis.get("seed", 42))
    ap.add_argument("--traj_duration_s", type=float, default=vis.get("traj_duration_s", 600.0))
    ap.add_argument("--rate_hz", type=float, default=vis.get("rate_hz", 100.0))
    ap.add_argument("--train_routes", type=int, default=vis.get("train_routes", 8))
    ap.add_argument("--val_routes", type=int, default=vis.get("val_routes", 2))
    ap.add_argument("--test_routes", type=int, default=vis.get("test_routes", 2))

    # 物理
    ap.add_argument("--use_slip", action="store_true")
    ap.add_argument("--use_gravity", action="store_true")
    ap.add_argument("--use_roll_pitch", action="store_true")
    ap.add_argument("--bank_gain", type=float, default=1.0)
    ap.add_argument("--pitch_gain", type=float, default=1.0)

    # IMU 两路
    ap.add_argument("--acc_window", type=int, default=512)
    ap.add_argument("--acc_stride", type=int, default=256)
    ap.add_argument("--acc_preproc", choices=["raw","ma_residual","diff"], default="raw")
    ap.add_argument("--acc_ma", type=int, default=51)

    ap.add_argument("--gyr_window", type=int, default=512)
    ap.add_argument("--gyr_stride", type=int, default=256)
    ap.add_argument("--gyr_preproc", choices=["raw","ma_residual","diff"], default="raw")
    ap.add_argument("--gyr_ma", type=int, default=51)

    # 视觉
    ap.add_argument("--cam_rate_hz", type=float, default=vis.get("cam_rate_hz", 20.0))
    ap.add_argument("--img_w", type=int, default=vis.get("img_w", 640))
    ap.add_argument("--img_h", type=int, default=vis.get("img_h", 480))
    ap.add_argument("--fx", type=float, default=vis.get("fx", 400.0))
    ap.add_argument("--fy", type=float, default=vis.get("fy", 400.0))
    ap.add_argument("--cx", type=float, default=vis.get("cx", 320.0))
    ap.add_argument("--cy", type=float, default=vis.get("cy", 240.0))
    ap.add_argument("--vis_window", type=int, default=vis.get("vis_window", 64))
    ap.add_argument("--vis_stride", type=int, default=vis.get("vis_stride", 32))
    ap.add_argument("--noise_px", type=float, default=vis.get("noise_px", 0.5))
    ap.add_argument("--outlier_ratio", type=float, default=vis.get("outlier_ratio", 0.1))
    ap.add_argument("--min_match", type=int, default=vis.get("min_match", 20))

    args = ap.parse_args()

    make_splits(
        Path(args.out),
        args.traj_duration_s, args.rate_hz, args.seed,
        args.train_routes, args.val_routes, args.test_routes,
        args.use_slip, args.use_gravity, args.use_roll_pitch,
        args.bank_gain, args.pitch_gain,
        args.acc_window, args.acc_stride, args.acc_preproc, args.acc_ma,
        args.gyr_window, args.gyr_stride, args.gyr_preproc, args.gyr_ma,
        args.cam_rate_hz, args.img_w, args.img_h, args.fx, args.fy, args.cx, args.cy,
        args.vis_window, args.vis_stride, args.noise_px, args.outlier_ratio, args.min_match
    )
    print("Done.")

if __name__ == "__main__":
    main()
