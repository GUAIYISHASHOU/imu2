#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from utils import load_config_file

def _rot_matrix_zxy(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Rz = np.array([[cy, -sy, 0.0],[sy, cy, 0.0],[0.0, 0.0, 1.0]], dtype=np.float64)
    Ry = np.array([[cp, 0.0, sp],[0.0, 1.0, 0.0],[-sp, 0.0, cp]], dtype=np.float64)
    Rx = np.array([[1.0, 0.0, 0.0],[0.0, cr, -sr],[0.0, sr, cr]], dtype=np.float64)
    return (Rz @ Ry @ Rx).astype(np.float64)


# ========== 物理仿真（同一条长轨迹，供两路共享，强化为运动学自行车模型） ==========
def bicycle_traj(T: int, dt: float, seed: int,
                 wheelbase: float = 2.5, rear_ratio: float = 0.5,
                 use_slip: bool=False, use_gravity: bool=False, use_roll_pitch: bool=False,
                 bank_gain: float=1.0, pitch_gain: float=1.0):
    """
    生成一条长轨迹的 IMU 真值与时变噪声方差。
    - 加速度真值：ax = a_cmd, ay = v*omega（向心项），az=0
    - 陀螺真值：gx=gy=0, gz=omega
    - 可选：简单引入横摆/俯仰引起的重力投影（use_gravity/use_roll_pitch）
    - 可选：横向“滑移”衰减（use_slip）

    返回:
      acc_true:(T,3), gyr_true:(T,3), a_var:(T,), g_var:(T,)
    """
    rng = np.random.default_rng(seed)
    x = y = yaw = 0.0
    roll = 0.0
    pitch = 0.0
    v = 5.0

    acc_true = np.zeros((T,3), dtype=np.float32)
    gyr_true = np.zeros((T,3), dtype=np.float32)

    t = np.arange(T) * dt
    # 时变异方差（非负与平滑）
    a_var = 0.20 * (1.0 + 0.7*np.sin(0.60*t) + 0.30*rng.normal(size=T))
    g_var = 0.05 * (1.0 + 0.8*np.cos(0.40*t+0.5) + 0.30*rng.normal(size=T))
    a_var = np.clip(a_var, 1e-5, 5.0).astype(np.float32)
    g_var = np.clip(g_var, 1e-6, 1.0).astype(np.float32)

    g = 9.81

    # 后轴距与重力常量
    lr = float(np.clip(rear_ratio, 1e-3, 1.0 - 1e-3)) * wheelbase
    g = 9.81

    prev_roll = roll
    prev_pitch = pitch
    prev_yaw = yaw

    for k in range(T):
        t_k = k * dt
        # 控制输入（基于时间的平滑变化）
        a_cmd = 0.50*np.sin(0.07*t_k)
        delta = 0.20*np.sin(0.05*t_k)  # 转向角

        # 速度更新
        v = float(np.clip(v + a_cmd*dt, 0.1, 20.0))

        # 侧偏角 beta（可选）
        if use_slip:
            beta = np.arctan((lr / wheelbase) * np.tan(delta))
        else:
            beta = 0.0

        # 运动学自行车模型偏航角速度
        if use_slip:
            yaw_rate = (v / lr) * np.sin(beta)
        else:
            yaw_rate = (v / wheelbase) * np.tan(delta)

        # 状态推进
        yaw = yaw + yaw_rate * dt
        x = x + v * np.cos(yaw + beta) * dt
        y = y + v * np.sin(yaw + beta) * dt

        # 姿态（可选，平滑）
        if use_roll_pitch:
            ay_c = v * yaw_rate
            roll_target = bank_gain * np.arctan2(ay_c, g)
            pitch_target = -pitch_gain * np.arctan2(a_cmd, g)
            alpha = 0.02
            roll = (1.0 - alpha) * roll + alpha * roll_target
            pitch = (1.0 - alpha) * pitch + alpha * pitch_target

        # 体坐标加速度（不含重力）
        ax = a_cmd
        ay = v * yaw_rate
        az = 0.0

        # 重力投影（可选，完整旋转）
        if use_gravity:
            R_bw = _rot_matrix_zxy(yaw, pitch if use_roll_pitch else 0.0, roll if use_roll_pitch else 0.0)
            g_world = np.array([0.0, 0.0, 9.81], dtype=np.float64)
            g_body = R_bw.T @ g_world
            ax += float(g_body[0]); ay += float(g_body[1]); az += float(g_body[2])

        # 陀螺输出
        if k == 0:
            roll_rate = 0.0; pitch_rate = 0.0
        else:
            roll_rate = (roll - prev_roll) / dt if use_roll_pitch else 0.0
            pitch_rate = (pitch - prev_pitch) / dt if use_roll_pitch else 0.0
        gx = roll_rate; gy = pitch_rate; gz = yaw_rate

        prev_roll = roll; prev_pitch = pitch; prev_yaw = yaw

        acc_true[k] = [ax, ay, az]
        gyr_true[k] = [gx, gy, gz]

    return acc_true, gyr_true, a_var, g_var


def simulate_long(T: int, dt: float, seed: int,
                  wheelbase: float = 2.5, rear_ratio: float = 0.5,
                  use_slip=False, use_gravity=False, use_roll_pitch=False,
                  bank_gain=1.0, pitch_gain=1.0):
    rng = np.random.default_rng(seed)
    acc_true, gyr_true, a_var, g_var = bicycle_traj(
        T, dt, seed,
        wheelbase=wheelbase, rear_ratio=rear_ratio,
        use_slip=use_slip, use_gravity=use_gravity, use_roll_pitch=use_roll_pitch,
        bank_gain=bank_gain, pitch_gain=pitch_gain
    )

    acc_noise = rng.normal(scale=np.sqrt(a_var)[:, None], size=(T,3)).astype(np.float32)
    gyr_noise = rng.normal(scale=np.sqrt(g_var)[:, None], size=(T,3)).astype(np.float32)

    acc_meas = acc_true + acc_noise
    gyr_meas = gyr_true + gyr_noise

    X_long  = np.zeros((T,6), dtype=np.float32)
    X_long[:,0:3] = acc_meas
    X_long[:,3:6] = gyr_meas

    E2_long = np.zeros((T,2), dtype=np.float32)
    E2_long[:,0] = np.sum(acc_noise**2, axis=-1)   # ACC 误差平方和
    E2_long[:,1] = np.sum(gyr_noise**2, axis=-1)   # GYR 误差平方和

    return X_long, E2_long, a_var.astype(np.float32), g_var.astype(np.float32)


# ========== 工具：滑窗 & 预处理（可分路配置） ==========
def window_count(T: int, win: int, stride: int) -> int:
    return 0 if T < win else 1 + (T - win) // stride

def sliding_window(arr: np.ndarray, win: int, stride: int) -> np.ndarray:
    T = arr.shape[0]
    n = window_count(T, win, stride)
    if n == 0:
        return np.zeros((0, win) + arr.shape[1:], dtype=arr.dtype)
    out = []
    for i in range(n):
        s, e = i*stride, i*stride + win
        out.append(arr[s:e])
    return np.stack(out, axis=0)

def moving_avg(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1: return x
    pad = k // 2
    xp = np.pad(x, ((pad,pad),(0,0)), mode="reflect")
    ker = np.ones((k,1), dtype=np.float32) / k
    out = np.zeros_like(xp, dtype=np.float32)
    for c in range(x.shape[1]):
        out[:,c] = np.convolve(xp[:,c], ker[:,0], mode="same")
    return out[pad:-pad]

def preprocess_route(x_long: np.ndarray, mode: str, ma_len: int) -> np.ndarray:
    """
    对输入进行“仅作为输入表征”的预处理（标签 E2 不变）：
      - raw: 原始信号
      - ma_residual: 移动平均低通后的残差（去缓慢趋势，突出噪声/高频）
      - diff: 一阶差分（高通特性）
    """
    if mode == "raw":
        return x_long
    elif mode == "ma_residual":
        base = moving_avg(x_long, ma_len)
        return (x_long - base).astype(np.float32)
    elif mode == "diff":
        d = np.diff(x_long, axis=0, prepend=x_long[:1])
        return d.astype(np.float32)
    else:
        raise ValueError(f"Unknown preprocess mode: {mode}")


# ========== 主流程：同一长序列 -> ACC/GYR 两路各自管线 ==========
def make_split_dual(num_routes: int, split_name: str,
                    traj_duration_s: float, rate_hz: float, seed_base: int,
                    # 物理项
                    wheelbase: float, rear_ratio: float,
                    use_slip: bool, use_gravity: bool, use_roll_pitch: bool,
                    bank_gain: float, pitch_gain: float,
                    # ACC 路由特有配置
                    acc_window: int, acc_stride: int, acc_preproc: str, acc_ma: int,
                    # GYR 路由特有配置
                    gyr_window: int, gyr_stride: int, gyr_preproc: str, gyr_ma: int,
                    also_write_combined: bool):
    dt = 1.0 / rate_hz
    T_long = int(round(traj_duration_s * rate_hz))

    # 累计器
    Xc_list, E2c_list, YA_list, YG_list = [], [], [], []   # 合并版（可选）
    Xa_list, E2a_list, Ma_list, YAa_list = [], [], [], []  # ACC 分路
    Xg_list, E2g_list, Mg_list, YGg_list = [], [], [], []  # GYR 分路

    for r in range(num_routes):
        seed = seed_base + r
        X_long, E2_long, YACC_long, YGYR_long = simulate_long(
            T_long, dt, seed,
            wheelbase=wheelbase, rear_ratio=rear_ratio,
            use_slip=use_slip, use_gravity=use_gravity, use_roll_pitch=use_roll_pitch,
            bank_gain=bank_gain, pitch_gain=pitch_gain
        )

        # --- 合并版（可选） ---
        if also_write_combined:
            Xc = sliding_window(X_long,  acc_window, acc_stride)   # 用 ACC 的窗口配置做个一致窗口（仅供对齐/可视化）
            E2c= sliding_window(E2_long, acc_window, acc_stride)
            YAc= sliding_window(YACC_long, acc_window, acc_stride)
            YGc= sliding_window(YGYR_long, acc_window, acc_stride)
            Xc_list.append(Xc); E2c_list.append(E2c); YA_list.append(YAc); YG_list.append(YGc)

        # --- ACC 路由 ---
        Xa_long = preprocess_route(X_long[:, :3], acc_preproc, acc_ma)
        Ea_long = E2_long[:, [0]]
        Xa = sliding_window(Xa_long, acc_window, acc_stride)      # (Na, Wa, 3)
        Ea = sliding_window(Ea_long,  acc_window, acc_stride)     # (Na, Wa, 1)
        Ma = np.ones((Xa.shape[0], Xa.shape[1]), dtype=np.float32)
        YAa= sliding_window(YACC_long, acc_window, acc_stride)    # (Na, Wa)

        Xa_list.append(Xa); E2a_list.append(Ea); Ma_list.append(Ma); YAa_list.append(YAa)

        # --- GYR 路由 ---
        Xg_long = preprocess_route(X_long[:, 3:6], gyr_preproc, gyr_ma)
        Eg_long = E2_long[:, [1]]
        Xg = sliding_window(Xg_long, gyr_window, gyr_stride)      # (Ng, Wg, 3)
        Eg = sliding_window(Eg_long,  gyr_window, gyr_stride)     # (Ng, Wg, 1)
        Mg = np.ones((Xg.shape[0], Xg.shape[1]), dtype=np.float32)
        YGg= sliding_window(YGYR_long, gyr_window, gyr_stride)    # (Ng, Wg)

        Xg_list.append(Xg); E2g_list.append(Eg); Mg_list.append(Mg); YGg_list.append(YGg)

    # 拼接各路
    if also_write_combined:
        Xc  = np.concatenate(Xc_list,  axis=0).astype(np.float32)
        E2c = np.concatenate(E2c_list, axis=0).astype(np.float32)
        YAc = np.concatenate(YA_list,   axis=0).astype(np.float32)
        YGc = np.concatenate(YG_list,   axis=0).astype(np.float32)
        Mc  = np.ones((Xc.shape[0], Xc.shape[1]), dtype=np.float32)
    else:
        Xc = E2c = YAc = YGc = Mc = None

    Xa  = np.concatenate(Xa_list,  axis=0).astype(np.float32)
    Ea  = np.concatenate(E2a_list, axis=0).astype(np.float32)
    Ma  = np.concatenate(Ma_list,  axis=0).astype(np.float32)
    YAa = np.concatenate(YAa_list, axis=0).astype(np.float32)

    Xg  = np.concatenate(Xg_list,  axis=0).astype(np.float32)
    Eg  = np.concatenate(E2g_list, axis=0).astype(np.float32)
    Mg  = np.concatenate(Mg_list,  axis=0).astype(np.float32)
    YGg = np.concatenate(YGg_list, axis=0).astype(np.float32)

    print(f"[{split_name}] routes={num_routes} T={T_long} "
          f"| ACC win/stride={acc_window}/{acc_stride} -> {Xa.shape[0]} windows "
          f"| GYR win/stride={gyr_window}/{gyr_stride} -> {Xg.shape[0]} windows")
    return (Xc, E2c, Mc, YAc, YGc), (Xa, Ea, Ma, YAa), (Xg, Eg, Mg, YGg)


def save_split_dual(out_dir: Path, name: str,
                    combined, acc, gyr,
                    write_combined=True):
    (Xc, E2c, Mc, YAc, YGc) = combined
    (Xa, Ea, Ma, YAa) = acc
    (Xg, Eg, Mg, YGg) = gyr

    out_dir.mkdir(parents=True, exist_ok=True)

    # 可选：合并版（主要用于一致性检查/可视化；训练时建议用分路）
    if write_combined and Xc is not None:
        np.savez(out_dir / f"{name}.npz",
                 X=Xc, E2=E2c, MASK=Mc, Y_ACC=YAc, Y_GYR=YGc)

    # 分路：ACC
    np.savez(out_dir / f"{name}_acc.npz",
             X=Xa, E2=Ea, MASK=Ma, Y_ACC=YAa)

    # 分路：GYR
    np.savez(out_dir / f"{name}_gyr.npz",
             X=Xg, E2=Eg, MASK=Mg, Y_GYR=YGg)


def main():
    # 预解析配置路径
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None, help="YAML/JSON 配置文件（读取 bicycle 段）")
    args_pre, _ = pre.parse_known_args()

    cfg = load_config_file(args_pre.config)
    by = cfg.get("bicycle", {})

    ap = argparse.ArgumentParser("Generate dual-route NPZ (ACC & GYR) from one shared long-trajectory simulation",
                                 parents=[pre])
    ap.add_argument("--out", required=(by.get("out") is None), default=by.get("out"))
    ap.add_argument("--seed", type=int, default=by.get("seed", 42))

    # 长序列
    ap.add_argument("--traj_duration_s", type=float, default=by.get("traj_duration_s", 600.0))
    ap.add_argument("--rate_hz", type=float, default=by.get("rate_hz", 100.0))

    # 物理项（可选）
    ap.add_argument("--wheelbase", type=float, default=by.get("wheelbase", 2.5))
    ap.add_argument("--rear_ratio", type=float, default=by.get("rear_ratio", 0.5))
    ap.add_argument("--use_slip", action="store_true", default=bool(by.get("use_slip", False)))
    ap.add_argument("--use_gravity", action="store_true", default=bool(by.get("use_gravity", False)))
    ap.add_argument("--use_roll_pitch", action="store_true", default=bool(by.get("use_roll_pitch", False)))
    ap.add_argument("--bank_gain", type=float, default=by.get("bank_gain", 1.0))
    ap.add_argument("--pitch_gain", type=float, default=by.get("pitch_gain", 1.0))

    # 各 split 路数
    ap.add_argument("--train_routes", type=int, default=by.get("train_routes", 8))
    ap.add_argument("--val_routes",   type=int, default=by.get("val_routes", 2))
    ap.add_argument("--test_routes",  type=int, default=by.get("test_routes", 2))

    # ACC 路由的窗口与预处理
    ap.add_argument("--acc_window_size", type=int, default=by.get("acc_window_size", by.get("window_size", 512)))
    ap.add_argument("--acc_window_stride", type=int, default=by.get("acc_window_stride", by.get("window_stride", 256)))
    ap.add_argument("--acc_preproc", choices=["raw","ma_residual","diff"], default=by.get("acc_preproc", "raw"))
    ap.add_argument("--acc_ma", type=int, default=by.get("acc_ma", 51))

    # GYR 路由的窗口与预处理
    ap.add_argument("--gyr_window_size", type=int, default=by.get("gyr_window_size", by.get("window_size", 512)))
    ap.add_argument("--gyr_window_stride", type=int, default=by.get("gyr_window_stride", by.get("window_stride", 256)))
    ap.add_argument("--gyr_preproc", choices=["raw","ma_residual","diff"], default=by.get("gyr_preproc", "raw"))
    ap.add_argument("--gyr_ma", type=int, default=by.get("gyr_ma", 51))

    # 输出控制
    ap.add_argument("--no_combined", action="store_true", default=bool(by.get("no_combined", False)), help="不写合并版 *.npz（仅写 *_acc / *_gyr）")

    args = ap.parse_args()
    out = Path(args.out)

    # train/val/test 三个 split 同源但不同 routes；不会互相泄露
    comb_tr, acc_tr, gyr_tr = make_split_dual(
        num_routes=args.train_routes, split_name="train",
        traj_duration_s=args.traj_duration_s, rate_hz=args.rate_hz, seed_base=args.seed+1000,
        wheelbase=args.wheelbase, rear_ratio=args.rear_ratio,
        use_slip=args.use_slip, use_gravity=args.use_gravity, use_roll_pitch=args.use_roll_pitch,
        bank_gain=args.bank_gain, pitch_gain=args.pitch_gain,
        acc_window=args.acc_window_size, acc_stride=args.acc_window_stride,
        acc_preproc=args.acc_preproc, acc_ma=args.acc_ma,
        gyr_window=args.gyr_window_size, gyr_stride=args.gyr_window_stride,
        gyr_preproc=args.gyr_preproc, gyr_ma=args.gyr_ma,
        also_write_combined=(not args.no_combined)
    )
    comb_va, acc_va, gyr_va = make_split_dual(
        num_routes=args.val_routes, split_name="val",
        traj_duration_s=args.traj_duration_s, rate_hz=args.rate_hz, seed_base=args.seed+2000,
        wheelbase=args.wheelbase, rear_ratio=args.rear_ratio,
        use_slip=args.use_slip, use_gravity=args.use_gravity, use_roll_pitch=args.use_roll_pitch,
        bank_gain=args.bank_gain, pitch_gain=args.pitch_gain,
        acc_window=args.acc_window_size, acc_stride=args.acc_window_stride,
        acc_preproc=args.acc_preproc, acc_ma=args.acc_ma,
        gyr_window=args.gyr_window_size, gyr_stride=args.gyr_window_stride,
        gyr_preproc=args.gyr_preproc, gyr_ma=args.gyr_ma,
        also_write_combined=(not args.no_combined)
    )
    comb_te, acc_te, gyr_te = make_split_dual(
        num_routes=args.test_routes, split_name="test",
        traj_duration_s=args.traj_duration_s, rate_hz=args.rate_hz, seed_base=args.seed+3000,
        wheelbase=args.wheelbase, rear_ratio=args.rear_ratio,
        use_slip=args.use_slip, use_gravity=args.use_gravity, use_roll_pitch=args.use_roll_pitch,
        bank_gain=args.bank_gain, pitch_gain=args.pitch_gain,
        acc_window=args.acc_window_size, acc_stride=args.acc_window_stride,
        acc_preproc=args.acc_preproc, acc_ma=args.acc_ma,
        gyr_window=args.gyr_window_size, gyr_stride=args.gyr_window_stride,
        gyr_preproc=args.gyr_preproc, gyr_ma=args.gyr_ma,
        also_write_combined=(not args.no_combined)
    )

    save_split_dual(out, "train", comb_tr, acc_tr, gyr_tr, write_combined=(not args.no_combined))
    save_split_dual(out, "val",   comb_va, acc_va, gyr_va, write_combined=(not args.no_combined))
    save_split_dual(out, "test",  comb_te, acc_te, gyr_te, write_combined=(not args.no_combined))

    print(f"Done. Saved under: {out.resolve()}")
    if not args.no_combined:
        print("Also wrote combined *.npz (for quick sanity-check/visualization). For training, prefer *_acc / *_gyr.")
if __name__ == "__main__":
    main()
