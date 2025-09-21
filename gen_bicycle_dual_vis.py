#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import load_config_file
from engine_builtin import EngineCfg, generate_route as gen_engine

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

# -------------------- 轨迹可视化 --------------------
def plot_trajectory(traj, title="", save_path=None):
    """绘制单条轨迹的详细信息"""
    gt_enu = traj["gt_enu"]; t = traj["t"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=14)
    # 1. 2D 轨迹
    axes[0,0].plot(gt_enu[:,0], gt_enu[:,1], 'b-', linewidth=1)
    axes[0,0].plot(gt_enu[0,0], gt_enu[0,1], 'go', markersize=8, label='Start')
    axes[0,0].plot(gt_enu[-1,0], gt_enu[-1,1], 'ro', markersize=8, label='End')
    axes[0,0].set_xlabel('East (m)'); axes[0,0].set_ylabel('North (m)')
    axes[0,0].set_title('2D Trajectory (E-N)'); axes[0,0].grid(True, alpha=0.3)
    axes[0,0].legend(); axes[0,0].axis('equal')

    # 2. 高度
    axes[0,1].plot(t/60, gt_enu[:,2], 'g-', linewidth=1)
    axes[0,1].set_xlabel('Time (min)'); axes[0,1].set_ylabel('Up (m)')
    axes[0,1].set_title('Altitude Profile'); axes[0,1].grid(True, alpha=0.3)

    # 3. 速度
    speed = traj.get("speed", np.zeros(len(t)))
    axes[1,0].plot(t/60, speed, 'r-', linewidth=1)
    axes[1,0].set_xlabel('Time (min)'); axes[1,0].set_ylabel('Speed (m/s)')
    axes[1,0].set_title('Speed Profile'); axes[1,0].grid(True, alpha=0.3)

    # 4. 航向
    yaw = traj.get("yaw", np.zeros(len(t)))
    axes[1,1].plot(t/60, np.degrees(yaw), 'purple', linewidth=1)
    axes[1,1].set_xlabel('Time (min)'); axes[1,1].set_ylabel('Yaw (deg)')
    axes[1,1].set_title('Heading Profile'); axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    else:
        plt.show()

def plot_all_trajectories(trajectories, split_name, save_dir):
    """绘制所有轨迹的概览图"""
    if not trajectories: return
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{split_name.title()} Set Trajectories Overview ({len(trajectories)} routes)', fontsize=16)

    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
    # 1. All 2D
    for i, traj in enumerate(trajectories):
        gt_enu = traj["gt_enu"]
        axes[0,0].plot(gt_enu[:,0], gt_enu[:,1], color=colors[i], linewidth=1, label=f'Route {i}', alpha=0.7)
        axes[0,0].plot(gt_enu[0,0], gt_enu[0,1], 'o', color=colors[i], markersize=6)
    axes[0,0].set_xlabel('East (m)'); axes[0,0].set_ylabel('North (m)')
    axes[0,0].set_title('All 2D Trajectories'); axes[0,0].grid(True, alpha=0.3)
    axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left'); axes[0,0].axis('equal')

    # 2. 高度
    for i, traj in enumerate(trajectories):
        axes[0,1].plot(traj["t"]/60, traj["gt_enu"][:,2], color=colors[i], linewidth=1, alpha=0.7)
    axes[0,1].set_xlabel('Time (min)'); axes[0,1].set_ylabel('Up (m)')
    axes[0,1].set_title('Altitude Profiles'); axes[0,1].grid(True, alpha=0.3)

    # 3. 速度
    for i, traj in enumerate(trajectories):
        axes[1,0].plot(traj["t"]/60, traj.get("speed", np.zeros(len(traj["t"]))), color=colors[i], linewidth=1, alpha=0.7)
    axes[1,0].set_xlabel('Time (min)'); axes[1,0].set_ylabel('Speed (m/s)')
    axes[1,0].set_title('Speed Profiles'); axes[1,0].grid(True, alpha=0.3)

    # 4. 统计
    stats_text = []
    for i, traj in enumerate(trajectories):
        gt_enu = traj["gt_enu"]
        total_dist = np.sum(np.linalg.norm(np.diff(gt_enu[:,:2], axis=0), axis=1))
        max_speed = np.max(traj.get("speed", [0])); duration = traj["t"][-1] / 60
        stats_text.append(f'Route {i}: {total_dist:.1f}m, {max_speed:.1f}m/s, {duration:.1f}min')
    axes[1,1].text(0.05, 0.95, '\n'.join(stats_text), transform=axes[1,1].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[1,1].set_title('Trajectory Statistics'); axes[1,1].axis('off')

    plt.tight_layout()
    save_path = Path(save_dir) / f"{split_name}_trajectories_overview.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Saved trajectory overview: {save_path}")

# -------------------- 自行车/独轮车轨迹 + IMU 噪声 --------------------
def bicycle_traj(
    T: int, dt: float, seed: int,
    use_slip: bool = True, use_gravity: bool = True, use_roll_pitch: bool = True,
    bank_gain: float = 1.0, pitch_gain: float = 1.0,
    **kw
):
    """
    以 engine_builtin.py 生成真实/连贯的车辆轨迹，并映射到 IMU 真值（body frame）与 roll/pitch 代理。
    返回值与旧实现保持一致：acc_true, gyr_true, a_var, g_var, roll, pitch, speed
    """
    import numpy as np
    rng = np.random.default_rng(seed)

    # === 1) 调引擎生成路况 ===
    # 你也可以把这些参数做成 CLI / 配置，此处给出合理默认
    eng_cfg = EngineCfg(
        dt=dt, duration_s=T*dt,
        v_max=kw.get("eng_v_max", 20.0),       # m/s
        a_lon_max=kw.get("eng_a_lon_max", 2.5),
        a_lat_max=kw.get("eng_a_lat_max", 4.0),
        delta_max=kw.get("eng_delta_max", 0.5),       # ~28.6°
        ddelta_max=kw.get("eng_ddelta_max", 0.6),     # rad/s
        tau_delta=kw.get("eng_tau_delta", 0.25),
        sigma_max=kw.get("eng_sigma_max", 0.30),
        jerk_lat_max=kw.get("eng_jerk_lat_max", 6.0),
        grade_sigma=(kw.get("eng_grade_std", 0.01), kw.get("eng_grade_std", 0.01)),
        grade_tau_s=(60.0, 180.0),
    )
    route = gen_engine(seed=seed, cfg=eng_cfg)  # dict: t,x,y,z,yaw,v,kappa,a_lat,a_lon,jerk

    # === 2) 提取真值并构造 IMU 量 ===
    yaw = route["yaw"].astype(np.float32)           # 航向（世界系）
    v   = route["v"].astype(np.float32)             # 速度
    a_lon = route["a_lon"].astype(np.float32)       # 纵向加速度（世界->此处当作车体系 x）
    a_lat = route["a_lat"].astype(np.float32)       # 横向加速度（世界->此处当作车体系 y）

    # 真值陀螺：仅 z 轴（平面运动假设）
    yaw_rate = np.diff(yaw, prepend=yaw[:1]) / dt
    gyr_true = np.stack([
        np.zeros_like(yaw_rate, dtype=np.float32),
        np.zeros_like(yaw_rate, dtype=np.float32),
        yaw_rate.astype(np.float32)
    ], axis=-1)

    # 真值加计（先不含重力；稍后按需减重力投影）
    ax = a_lon
    ay = a_lat
    az = np.zeros_like(ax, dtype=np.float32)
    acc_true = np.stack([ax, ay, az], axis=-1).astype(np.float32)

    # roll/pitch 代理：由横/纵向加速度估计（小角近似）
    g = 9.81
    roll  = (bank_gain  * (a_lat / g)).astype(np.float32)   # 左右倾角 ~ 侧向加速度/g
    pitch = (-pitch_gain * (a_lon / g)).astype(np.float32)  # 俯仰 ~ 纵向加速度/g（前加速为低头）

    if use_gravity:
        # 将重力从 body 加速度中扣除：body系下的重力投影（小角近似下也可）
        c_r, s_r = np.cos(roll), np.sin(roll)
        c_p, s_p = np.cos(pitch), np.sin(pitch)
        gx_b = -g * s_p
        gy_b =  g * s_r
        gz_b =  g * (c_p * np.cos(roll))  # 小角可近似为 g
        grav = np.stack([gx_b, gy_b, gz_b], axis=-1).astype(np.float32)
        acc_true = acc_true - grav

    # === 3) 传感器噪声方差轨迹（保持与旧版相近的"随运动强度变化"的日程） ===
    # 你可以替换为原函数里的 schedule；这里给一个稳定且与运动强度相关的示例
    v_eps = np.clip(v / (np.max(v)+1e-6), 0.0, 1.0)
    w_eps = np.clip(np.abs(yaw_rate) / (np.max(np.abs(yaw_rate))+1e-6), 0.0, 1.0)
    a_var = (0.03 + 0.02 * v_eps)**2     # 加计 ~ 速度越快噪声略增
    g_var = (0.002 + 0.002 * w_eps)**2   # 陀螺 ~ 转向越猛噪声略增

    speed = v.astype(np.float32)
    return acc_true, gyr_true, a_var.astype(np.float32), g_var.astype(np.float32), roll, pitch, speed

def simulate_imu(T, dt, seed, **phys):
    rng = np.random.default_rng(seed)
    acc_true, gyr_true, a_var, g_var, roll, pitch, speed = bicycle_traj(T, dt, seed, **phys)
    acc_noise = rng.normal(scale=np.sqrt(a_var)[:,None], size=(T,3)).astype(np.float32)
    gyr_noise = rng.normal(scale=np.sqrt(g_var)[:,None], size=(T,3)).astype(np.float32)
    acc_meas = acc_true + acc_noise
    gyr_meas = gyr_true + gyr_noise

    X_imu = np.concatenate([acc_meas, gyr_meas], axis=-1).astype(np.float32)   # (T,6)
    E2_imu= np.stack([np.sum(acc_noise**2,axis=-1), np.sum(gyr_noise**2,axis=-1)], axis=-1).astype(np.float32)  # (T,2)
    
    # 新增：真值 yaw/xy（用于相机位姿，避免陀螺积分漂移）
    yaw_true = np.cumsum(gyr_true[:,2]) * dt
    xy_true  = np.cumsum(np.stack([speed*np.cos(yaw_true), speed*np.sin(yaw_true)], -1), 0) * dt
    
    return X_imu, E2_imu, a_var, g_var, roll, pitch, speed, yaw_true, xy_true

# -------------------- 相机/视觉仿真 --------------------
def sample_landmarks(num=4000, seed=0, x_max=80.0, y_half=30.0, z_range=(-1.0,3.0)):
    """
    均匀撒点（世界系），在车前方一个长盒子里（避免身后无意义点）
    x in [2, x_max] 表示"前方距离"（沿 +x）
    y in [-y_half, y_half] 表示"左右分布"（沿 y）
    z in z_range 表示"地面附近高度"（沿 z，上下）
    """
    rng = np.random.default_rng(seed)
    xs = rng.uniform(  2, x_max, size=num)     # ← 用 x_max 代替固定 80
    ys = rng.uniform(-y_half, y_half, size=num)
    zs = rng.uniform( z_range[0], z_range[1], size=num)
    return np.stack([xs, ys, zs], axis=-1).astype(np.float32)

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
        R_wb = rot_z(yaw[k]) @ rot_y(pitch[k]) @ rot_x(roll[k])   # body->world ✅
        R_bw = R_wb.T
        R_cw = R_cb @ R_bw                                        # world->cam ✅
        p_wb = np.array([trans_xy[k,0], trans_xy[k,1], z_height], np.float32)
        p_wc = p_wb + R_wb @ t_cb                                 # cam center (world)
        t_cw = -R_cw @ p_wc                                       # ✅
        Rc_list.append(R_cw)
        tc_list.append(t_cw.astype(np.float32))
    return np.stack(Rc_list,0), np.stack(tc_list,0)  # (T,3,3),(T,3)

def project_points(Pw, Rcw, tcw, K, img_wh, noise_px=0.5, rng=None):
    """
    把世界点投到像素系，返回：
      uv_noisy: (M,2)
      ids_global: (M,) 对应 Pw 的全局下标
      Pc: (M,3) 可见三维点在相机坐标系下的坐标
    """
    if rng is None:
        rng = np.random.default_rng(0)
    Pc_all = (Rcw @ Pw.T).T + tcw
    Z = Pc_all[:, 2]
    vis_mask = Z > 0.3
    # 对所有点投影，再用可见/在图像内的掩码过滤，确保可回到全局下标
    uv_all = (K @ (Pc_all.T / np.clip(Z, 1e-6, None))).T[:, :2]
    W, H = img_wh
    in_img = (uv_all[:, 0] >= 0) & (uv_all[:, 0] < W) & (uv_all[:, 1] >= 0) & (uv_all[:, 1] < H)
    global_mask = vis_mask & in_img
    if not np.any(global_mask):
        return np.zeros((0, 2), np.float32), np.zeros((0,), np.int32), np.zeros((0, 3), np.float32)
    ids_global = np.where(global_mask)[0]
    uv = uv_all[global_mask]
    Pc = Pc_all[global_mask]
    uv_noisy = uv + rng.normal(scale=noise_px, size=uv.shape).astype(np.float32)
    return uv_noisy.astype(np.float32), ids_global.astype(np.int32), Pc.astype(np.float32)

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

def simulate_vision_from_trajectory(T_cam, t_cam_idx, yaw, roll, pitch, xy, speed, dt_cam,   # 轨迹（下采样到相机时刻的索引）
                                    K, img_wh, Pw, noise_px=0.5, outlier_ratio=0.1,
                                    min_match=20, seed=0, 
                                    # 新增时变参数
                                    noise_tau_s=0.4, noise_ln_std=0.30, out_tau_s=0.6,
                                    burst_prob=0.03, burst_gain=(0.2, 0.6),
                                    motion_k1=0.8, motion_k2=0.4, lp_pool_p=3.0):
    """
    基于轨迹位姿 + 3D 地图，仿真相机观测与相邻帧匹配，并计算每帧 E2_vis。
    返回：
      E2_vis: (T_cam,)  每帧（与上一帧）Sampson^2 的和（若匹配不足则置 0，mask=0）
      X_vis:  (T_cam, D) 每帧特征（num_inliers_norm, mean_flow_px, std_flow_px, baseline_norm, yaw_rate, speed, roll, pitch）
      MASK:   (T_cam,)  有效帧掩码（首帧或匹配不足置 0）
    """
    rng = np.random.default_rng(seed)
    
    # ================ Lp pooling聚合函数 ================
    def aggregate_r2(r2, p=3.0):
        r2 = np.asarray(r2, np.float64)
        if r2.size == 0:
            return 0.0
        return float((np.mean(np.power(r2, p/2.0)))**(2.0/p))
    
    # ================ 时变噪声和外点率生成 ================
    dtc = dt_cam
    
    # ① 基于 log-正态抖动的像素噪声幅度（更贴近"画质/模糊"）
    alpha_n = np.exp(-dtc / max(1e-3, noise_tau_s))
    z = 0.0
    noise_px_t = np.empty(T_cam, dtype=np.float32)
    for k in range(T_cam):
        z = alpha_n*z + np.sqrt(1 - alpha_n**2) * rng.normal(0, noise_ln_std)
        noise_px_t[k] = noise_px * np.exp(z)   # 基于原 noise_px 做比例抖动

    # ② 外点率 OU + 突发项
    alpha_o = np.exp(-dtc / max(1e-3, out_tau_s))
    y = 0.0
    outlier_t = np.empty(T_cam, dtype=np.float32)
    for k in range(T_cam):
        y = alpha_o*y + np.sqrt(1 - alpha_o**2) * rng.normal(0, 0.05)
        outlier_t[k] = np.clip(outlier_ratio + y, 0.0, 0.6)
        # 随机突发（比如强遮挡/快速横摆导致错误匹配暴增）
        if rng.random() < burst_prob:
            outlier_t[k] = np.clip(outlier_t[k] + rng.uniform(*burst_gain), 0.0, 0.8)

    # ③ 运动相关的观测质量调节
    yaw_rate_cam = np.diff(yaw, prepend=yaw[:1]) / max(1e-6, dtc)
    speed_cam = speed
    yaw_ref = 0.6    # 可按数据范围调
    v_ref   = 6.0

    for k in range(T_cam):
        scale = 1.0 + motion_k1 * min(1.0, abs(yaw_rate_cam[k]) / yaw_ref) \
                     + motion_k2 * min(1.0, abs(speed_cam[k])        / v_ref)
        noise_px_t[k] *= scale
        outlier_t[k]   = np.clip(outlier_t[k] * scale, 0.0, 0.85)

    # ④ 帧内"可用内点数"波动（更真实的纹理/视差变化）
    N0 = 200  # 基础内点数
    N_scale = np.clip(np.exp(0.4 * rng.normal(size=T_cam)), 0.5, 1.5)  # lognormal
    N_inlier_target = np.maximum(min_match, (N0 * N_scale * (1.0 - outlier_t)).astype(int))
    
    # 相机外参：车体x(前进)→相机z(光轴), 车体y(左)→相机-x(右), 车体z(上)→相机-y(下)
    R_cb = np.array([[ 0, -1,  0],
                     [ 0,  0, -1],
                     [ 1,  0,  0]], dtype=np.float32)  # 让相机z沿车体+x
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
        uv, id_in_img, Pc = project_points(Pw, Rcw_all[k], tcw_all[k], K, img_wh, noise_px=noise_px_t[k], rng=rng)
        UV.append(uv)
        idlists.append(id_in_img)  # 这些是 Pw 的索引子集
        Pc_list.append(Pc)

    E2_vis = np.zeros(T_cam, np.float32)
    X_vis  = np.zeros((T_cam,8), np.float32)  # 8D 状态特征
    MASK   = np.zeros(T_cam, np.float32)

    # 构造 yaw_rate/speed/roll/pitch（按相机时刻子采样）
    yaw_cam = yaw[t_cam_idx]
    # yaw_rate in rad/s using actual camera interval
    yaw_rate_cam = np.diff(yaw_cam, prepend=yaw_cam[:1]) / max(1e-6, dt_cam)
    # true speed at camera timestamps (m/s)
    speed_cam = speed[t_cam_idx]
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
        def pick_uv(UV_list, idlist, common_ids):
            pos = {gid:i for i,gid in enumerate(idlist)}
            idx = [pos[g] for g in common_ids]
            return UV_list[idx]
        uv1 = pick_uv(UV[k-1], ids_prev, common)
        uv2 = pick_uv(UV[k],   ids_curr, common)

        # 按目标内点数截断（模拟纹理/视差变化）
        M_available = uv1.shape[0]
        M_target = min(M_available, N_inlier_target[k])
        if M_target < M_available:
            # 随机子采样到目标数量
            keep_idx = rng.choice(M_available, size=M_target, replace=False)
            uv1 = uv1[keep_idx]
            uv2 = uv2[keep_idx]

        # 注入外点（使用时变外点率）
        M = uv1.shape[0]
        m_out = int(M * outlier_t[k])
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
        # 使用 Lp pooling 聚合 + 标准化
        C = 500.0  # 标准化常数
        N_inlier_actual = max(1, M - m_out)  # 实际内点数
        E2_vis[k] = aggregate_r2(d2, p=lp_pool_p) * (C / N_inlier_actual)
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
                vis_win:int, vis_str:int, noise_px:float, outlier_ratio:float, min_match:int,
                # 新增时变参数
                noise_tau_s:float, noise_ln_std:float, out_tau_s:float,
                burst_prob:float, burst_gain:tuple, motion_k1:float, motion_k2:float, lp_pool_p:float,
                # 引擎参数（有默认值）
                eng_v_max:float=20.0, eng_a_lat_max:float=4.0, eng_a_lon_max:float=2.5,
                eng_delta_max:float=0.5, eng_ddelta_max:float=0.6, eng_tau_delta:float=0.25,
                eng_sigma_max:float=0.30, eng_jerk_lat_max:float=6.0, eng_grade_std:float=0.01,
                # 轨迹可视化参数（有默认值）
                plot_trajectories:bool=True, plot_individual:bool=False, plot_dir:str="trajectory_plots"):

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
        # 保留在 synth_vis 中的 meta（带 mean/std）；这里不再额外写一份
        pass

    def one_split(num_routes, split_name, seed_base):
        dt = 1.0 / rate_hz
        T_long = int(round(traj_duration_s * rate_hz))
        # 相机采样索引（等间隔下采样）
        cam_step = int(round(rate_hz / cam_rate_hz))
        t_cam_idx = np.arange(0, T_long, cam_step, dtype=np.int32)
        T_cam = len(t_cam_idx)

        # 地图点将在每个route内部根据行驶距离动态生成

        Xa_list,Ea_list,Ma_list,YAa_list = [],[],[],[]
        Xg_list,Eg_list,Mg_list,YGg_list = [],[],[],[]
        Xv_list,Ev_list,Mv_list = [],[],[]
        seg_list = []  # (per-route) camera timeline labels
        trajectories = []  # 收集轨迹数据用于可视化

        print(f"[{split_name}] Processing {num_routes} routes...")
        pbar = tqdm(range(num_routes), desc=f"  {split_name.capitalize()}", unit="route")
        for r in pbar:
            seed_r = seed_base + r
            # 生成 IMU 长序列
            X_imu, E2_imu, Yacc, Ygyr, roll, pitch, speed, yaw, xy = simulate_imu(T_long, dt, seed_r,
                                                                          use_slip=use_slip,
                                                                          use_gravity=use_gravity,
                                                                          use_roll_pitch=use_roll_pitch,
                                                                          bank_gain=bank_gain, pitch_gain=pitch_gain,
                                                                          eng_v_max=eng_v_max, eng_a_lat_max=eng_a_lat_max,
                                                                          eng_a_lon_max=eng_a_lon_max, eng_delta_max=eng_delta_max,
                                                                          eng_ddelta_max=eng_ddelta_max, eng_tau_delta=eng_tau_delta,
                                                                          eng_sigma_max=eng_sigma_max, eng_jerk_lat_max=eng_jerk_lat_max,
                                                                          eng_grade_std=eng_grade_std)
            # 现在使用真值 yaw/xy（避免陀螺积分漂移导致相机走出点云走廊）
            
            # 收集轨迹数据用于可视化
            if plot_trajectories:
                t = np.arange(T_long) * dt
                traj = {
                    "t": t,
                    "gt_enu": np.column_stack([xy[:, 0], xy[:, 1], np.zeros(T_long)]),  # 使用真值xy，z=0
                    "yaw": yaw,
                    "speed": speed
                }
                trajectories.append(traj)

            # 根据该route的行驶距离动态生成点云
            dist_est = float(np.sum(speed) * dt)                 # ≈ 平均速度 × 时长
            x_max   = max(100.0, dist_est * 1.2)                 # 留一点富余
            density = 40.0                                       # 每米点数（保持可见点充足）
            num_pts = int(max(4000, density * x_max))
            Pw = sample_landmarks(num=num_pts, seed=seed_r+77, x_max=x_max)
            print(f"[{split_name}/route{r}] dist_est={dist_est:.1f}m, x_max={x_max:.1f}m, landmarks={num_pts}")

            # 视觉：从轨迹仿真
            K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], np.float32)
            dt_cam = cam_step * (1.0 / rate_hz)
            E2_vis, X_vis, M_vis = simulate_vision_from_trajectory(
                T_cam, t_cam_idx, yaw, roll, pitch, xy, speed, dt_cam, K, (img_w,img_h), Pw,
                noise_px=noise_px, outlier_ratio=outlier_ratio, min_match=min_match, seed=seed_r+999,
                noise_tau_s=noise_tau_s, noise_ln_std=noise_ln_std, out_tau_s=out_tau_s,
                burst_prob=burst_prob, burst_gain=burst_gain, motion_k1=motion_k1, motion_k2=motion_k2, lp_pool_p=lp_pool_p
            )

            # ---- 轻量自检与段标 ----
            # 有效覆盖率与单位量级检查（基于光流均值，像素）
            valid = (M_vis > 0.5).reshape(-1)
            cov = float(valid.mean()) if valid.size else 0.0
            print(f"[{split_name}/route{r}] T_cam={T_cam} vis_coverage={cov:.3f}")
            approx_unit_check_flow(X_vis[valid, 1] if valid.any() else np.array([]), tag=f"{split_name}/route{r}/flow_mean")
            
            # 详细统计：可见点、公共点、内点数量
            if valid.any():
                inlier_norm = X_vis[valid, 0]  # 内点比例
                flow_mean = X_vis[valid, 1]    # 光流均值
                baseline = X_vis[valid, 3]     # 基线范数
# 调试信息移至tqdm进度条
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
            
            # 更新进度条状态信息
            pbar.set_postfix({
                'VIS_win': Xv.shape[0],
                'coverage': f"{Mv.mean():.2f}",
                'T_cam': T_cam
            })

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
        
        # 生成轨迹可视化
        if plot_trajectories and trajectories:
            plot_dir_path = Path(plot_dir)
            plot_dir_path.mkdir(parents=True, exist_ok=True)
            plot_all_trajectories(trajectories, split_name, plot_dir_path)
            
            if plot_individual:
                split_dir = plot_dir_path / split_name
                split_dir.mkdir(parents=True, exist_ok=True)
                for i, traj in tqdm(enumerate(trajectories), desc=f"    Plotting {split_name} routes", total=len(trajectories), unit="plot"):
                    title = f"{split_name.title()} Route {i} (seed={seed_base + i})"
                    save_path = split_dir / f"route_{i:02d}.png"
                    plot_trajectory(traj, title=title, save_path=save_path)
                print(f"Saved {len(trajectories)} individual trajectory plots in {split_dir}")
        
        return (Xa,Ea,Ma,YAa),(Xg,Eg,Mg,YGg),(Xv,Ev,Mv)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Train / Val / Test
    print(f"\n=== Generating Visual+IMU Data ===")
    print(f"Train: {train_routes} routes | Val: {val_routes} routes | Test: {test_routes} routes")
    print(f"Trajectory: {traj_duration_s}s @ {rate_hz}Hz | Camera: {cam_rate_hz}Hz")
    print(f"Visual windows: {vis_win}x{vis_str} | IMU windows: ACC {acc_win}x{acc_str}, GYR {gyr_win}x{gyr_str}\n")
    
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
        print(f"[{prefix}] Final VIS: {Xv.shape[0]} windows, coverage={Mv.mean():.3f}")

    # 旧行为：写一次 meta 到 out_dir；现取消，避免与 synth_vis/vis_meta.json 冲突

    print("\n=== Saving datasets ===")
    savetag("train", acc_tr, gyr_tr, vis_tr)
    savetag("val",   acc_va, gyr_va, vis_va)
    savetag("test",  acc_te, gyr_te, vis_te)
    print("=== All datasets saved ===\n")

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
    
    # 引擎参数（可选）
    ap.add_argument("--eng_v_max", type=float, default=20.0)
    ap.add_argument("--eng_a_lat_max", type=float, default=4.0)
    ap.add_argument("--eng_a_lon_max", type=float, default=2.5)
    ap.add_argument("--eng_delta_max", type=float, default=0.5)
    ap.add_argument("--eng_ddelta_max", type=float, default=0.6)
    ap.add_argument("--eng_tau_delta", type=float, default=0.25)
    ap.add_argument("--eng_sigma_max", type=float, default=0.30)
    ap.add_argument("--eng_jerk_lat_max", type=float, default=6.0)
    ap.add_argument("--eng_grade_std", type=float, default=0.01)
    
    # 轨迹可视化
    ap.add_argument("--plot_trajectories", action="store_true", default=vis.get("plot_trajectories", True),
                    help="生成轨迹可视化图")
    ap.add_argument("--plot_individual", action="store_true", default=vis.get("plot_individual", False),
                    help="为每条轨迹生成单独的图")
    ap.add_argument("--plot_dir", default=vis.get("plot_dir", "trajectory_plots"),
                    help="轨迹图保存目录")

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
    ap.add_argument("--min_match", type=int, default=vis.get("min_match", 12))
    # 新增时变参数
    ap.add_argument("--noise_tau_s", type=float, default=vis.get("noise_tau_s", 0.4))
    ap.add_argument("--noise_ln_std", type=float, default=vis.get("noise_ln_std", 0.30))
    ap.add_argument("--out_tau_s", type=float, default=vis.get("out_tau_s", 0.6))
    ap.add_argument("--burst_prob", type=float, default=vis.get("burst_prob", 0.03))
    ap.add_argument("--burst_gain", type=str, default=str(vis.get("burst_gain", [0.2, 0.6])))
    ap.add_argument("--motion_k1", type=float, default=vis.get("motion_k1", 0.8))
    ap.add_argument("--motion_k2", type=float, default=vis.get("motion_k2", 0.4))
    ap.add_argument("--lp_pool_p", type=float, default=vis.get("lp_pool_p", 3.0))

    args = ap.parse_args()
    
    # 解析 burst_gain 参数
    import ast
    try:
        args.burst_gain = ast.literal_eval(args.burst_gain) if isinstance(args.burst_gain, str) else args.burst_gain
    except:
        args.burst_gain = [0.2, 0.6]  # 默认值
    
    # 调试：确认配置被正确读取
    print(f"[cfg] cam_rate_hz={args.cam_rate_hz}  min_match={args.min_match}  outlier_ratio={args.outlier_ratio}  noise_px={args.noise_px}")
    print(f"[cfg] vis_window={args.vis_window}  vis_stride={args.vis_stride}")

    make_splits(
        Path(args.out),
        args.traj_duration_s, args.rate_hz, args.seed,
        args.train_routes, args.val_routes, args.test_routes,
        args.use_slip, args.use_gravity, args.use_roll_pitch,
        args.bank_gain, args.pitch_gain,
        args.acc_window, args.acc_stride, args.acc_preproc, args.acc_ma,
        args.gyr_window, args.gyr_stride, args.gyr_preproc, args.gyr_ma,
        args.cam_rate_hz, args.img_w, args.img_h, args.fx, args.fy, args.cx, args.cy,
        args.vis_window, args.vis_stride, args.noise_px, args.outlier_ratio, args.min_match,
        args.noise_tau_s, args.noise_ln_std, args.out_tau_s,
        args.burst_prob, args.burst_gain, args.motion_k1, args.motion_k2, args.lp_pool_p,
        args.eng_v_max, args.eng_a_lat_max, args.eng_a_lon_max,
        args.eng_delta_max, args.eng_ddelta_max, args.eng_tau_delta,
        args.eng_sigma_max, args.eng_jerk_lat_max, args.eng_grade_std,
        args.plot_trajectories, args.plot_individual, args.plot_dir
    )
    print("Done.")

if __name__ == "__main__":
    main()
