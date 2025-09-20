# sim/engine_builtin.py
# -*- coding: utf-8 -*-
"""
A+ Builtin Kinematic Bicycle Engine (curvature-continuous)
----------------------------------------------------------
- 单轨模型（kinematic single-track）
- 曲率连续：限幅 dκ/ds（等价离散 clothoid 过渡）
- 侧向/纵向/转角速率/速度等物理约束
- 随机坡度 OU 模型
- 零第三方依赖；返回 dict, 直接喂你的 IMU/GNSS 合成与窗口化

使用：
    cfg = EngineCfg(dt=0.01, duration_s=2000, v_max=30, a_lat_max=3.5)
    states = generate_route(seed=0, cfg=cfg, scenario="city")
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import math
import numpy as np


# ---------------------------- utils ---------------------------- #

def _lowpass(prev: float, target: float, dt: float, tau: float) -> float:
    """one-pole low-pass first-order response"""
    if tau <= 1e-9:
        return target
    a = dt / (tau + dt)
    return prev + a * (target - prev)

def _ou_step(prev: float, mu: float, sigma: float, dt: float, tau: float, rng: np.random.Generator) -> float:
    """Ornstein–Uhlenbeck step (mean-reverting)"""
    if tau <= 1e-9:
        return mu + sigma * rng.standard_normal()
    return prev + (-(prev - mu) / tau) * dt + sigma * math.sqrt(dt) * rng.standard_normal()

def _ou_track(N: int, dt: float, tau: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """zero-mean OU sequence"""
    z = np.zeros(N, dtype=np.float64)
    if N <= 1:
        return z
    a = dt / max(tau, 1e-9)
    s = sigma * math.sqrt(dt)
    for k in range(1, N):
        z[k] = z[k-1] + (-z[k-1]) * a + s * rng.standard_normal()
    return z


# ---------------------------- config ---------------------------- #

@dataclass
class EngineCfg:
    # time & geometry
    dt: float = 0.01                 # [s] integration step (100 Hz)
    duration_s: float = 200.0        # [s] total duration
    wheelbase: float = 2.7           # [m]
    yaw0: float = 0.0                # [rad] initial yaw
    # speed & limits
    v0: float = 6.0                  # [m/s] initial speed
    v_max: float = 30.0              # [m/s] speed cap
    a_lon_max: float = 2.0           # [m/s^2] |dv/dt|
    a_lat_max: float = 3.5           # [m/s^2] v^2*kappa
    # steering actuator
    delta_max: float = math.radians(35.0)    # [rad]
    ddelta_max: float = math.radians(30.0)   # [rad/s]
    tau_delta: float = 0.5                   # [s] first-order steering response
    # curvature continuity (A+)
    sigma_max: float = 3e-3          # [1/m^2] max |dκ/ds|，离散 clothoid 约束
    jerk_lat_max: float = 0.0        # [m/s^3] 可选侧向 jerk 上限；>0 时按 v 动态约束 σ
    # command scheduling
    seg_s: tuple = (3.0, 8.0)        # [s] piece duration for command re-sampling
    # grade model (z)
    grade_sigma: tuple = (0.01, 0.04)    # 1%~4%
    grade_tau_s: tuple = (60.0, 180.0)   # [s]
    # reproducibility
    seed_offset: int = 0             # per-route offset added to the external seed


# ---------------------------- engine ---------------------------- #

def generate_route(seed: int, cfg: EngineCfg, scenario: str = "city") -> Dict[str, np.ndarray]:
    """
    返回状态字典：
      t,x,y,z,yaw,v,delta,kappa,a_lat,a_lon,jerk
    说明：
      - 曲率连续：对 Δκ 做 |Δκ| ≤ σ * Δs 限幅，其中 σ 根据 cfg.sigma_max
        或（若 cfg.jerk_lat_max>0）按局部速度 v 动态取 σ = min(sigma_max, jerk_lat_max / max(v^3, eps))
      - 横向加速度约束：若 |v^2 κ| 超 a_lat_max，缩放 κ
    """
    # RNG
    rng = np.random.default_rng(int(seed) + int(cfg.seed_offset))

    # time base
    N = int(round(cfg.duration_s / cfg.dt))
    N = max(N, 2)
    dt = float(cfg.dt)
    t = np.arange(N, dtype=np.float64) * dt

    # state arrays
    x = np.zeros(N, dtype=np.float64)
    y = np.zeros(N, dtype=np.float64)
    z = np.zeros(N, dtype=np.float64)
    yaw = np.zeros(N, dtype=np.float64); yaw[0] = cfg.yaw0
    v = np.zeros(N, dtype=np.float64);   v[0] = cfg.v0
    delta = np.zeros(N, dtype=np.float64)
    kappa = np.zeros(N, dtype=np.float64)

    # commands
    a_cmd = 0.0
    delta_cmd = 0.0
    seg_left = 0

    # scenario-dependent randomness
    if scenario == "highway":
        a_mu, a_std = 0.0, 0.8
        d_mu, d_std = 0.0, math.radians(6.0)
    else:  # city / default
        a_mu, a_std = 0.0, 1.2
        d_mu, d_std = 0.0, math.radians(10.0)

    # grade OU process params
    g_tau = rng.uniform(*cfg.grade_tau_s)
    g_sigma = rng.uniform(*cfg.grade_sigma)
    g = _ou_track(N, dt, tau=g_tau, sigma=g_sigma, rng=rng)

    # curvature continuity state
    L = float(cfg.wheelbase)
    kappa_prev = 0.0  # previous executed curvature
    eps = 1e-9

    for i in range(1, N):
        # ---- segment resample ----
        if seg_left <= 0:
            seg_left = int(rng.uniform(*cfg.seg_s) / dt)
        seg_left -= 1

        # ---- commands OU + limits ----
        a_cmd = _ou_step(a_cmd, a_mu, a_std, dt, tau=1.0, rng=rng)
        delta_cmd = _ou_step(delta_cmd, d_mu, d_std, dt, tau=1.0, rng=rng)
        a_cmd = float(np.clip(a_cmd, -cfg.a_lon_max, cfg.a_lon_max))
        delta_cmd = float(np.clip(delta_cmd, -cfg.delta_max, cfg.delta_max))

        # ---- steering actuator (LPF + rate limit + amplitude cap) ----
        delta_raw = _lowpass(delta[i-1], delta_cmd, dt, cfg.tau_delta)
        ddelta = float(np.clip(delta_raw - delta[i-1], -cfg.ddelta_max * dt, cfg.ddelta_max * dt))
        delta_exec = float(np.clip(delta[i-1] + ddelta, -cfg.delta_max, cfg.delta_max))

        # ---- speed integration + cap ----
        v[i] = float(np.clip(v[i-1] + a_cmd * dt, 0.0, cfg.v_max))

        # ---- curvature command from steering ----
        kappa_cmd = math.tan(delta_exec) / L

        # ---- curvature continuity: |Δκ| ≤ σ * Δs ----
        ds = max(v[i-1] * dt, 1e-6)  # use previous speed for path length of this step
        sigma = cfg.sigma_max
        if cfg.jerk_lat_max and cfg.jerk_lat_max > 0.0:
            # σ ≤ J_max / v^3  （近似：常速时 ȧ_lat = v^3 σ）
            sigma_dyn = cfg.jerk_lat_max / max(v[i-1]**3, 0.3**3)
            sigma = min(cfg.sigma_max, sigma_dyn)
        dkap_lim = sigma * ds
        dkappa = float(np.clip(kappa_cmd - kappa_prev, -dkap_lim, dkap_lim))
        kappa_i = kappa_prev + dkappa

        # ---- lateral acceleration limit: |v^2 κ| ≤ a_lat_max ----
        a_lat_i = v[i] * v[i] * kappa_i
        if abs(a_lat_i) > cfg.a_lat_max and abs(kappa_i) > eps:
            scale = cfg.a_lat_max / (abs(a_lat_i) + eps)
            kappa_i *= scale
            # note: steering record will be updated from executed curvature below

        # ---- finalize steering from executed curvature ----
        delta[i] = math.atan(kappa_i * L)
        kappa[i] = kappa_i
        kappa_prev = kappa_i

        # ---- yaw & position integration ----
        yaw[i] = yaw[i-1] + v[i] * kappa[i] * dt
        x[i] = x[i-1] + v[i] * math.cos(yaw[i]) * dt
        y[i] = y[i-1] + v[i] * math.sin(yaw[i]) * dt

        # ---- altitude integrate from grade ----
        z[i] = z[i-1] + g[i] * v[i] * dt  # dz = grade * ds

    # diagnostics
    a_lon = np.gradient(v, dt)
    jerk = np.gradient(a_lon, dt)
    a_lat = v * v * kappa

    return {
        "t": t,
        "x": x, "y": y, "z": z,
        "yaw": yaw,
        "v": v,
        "delta": delta,
        "kappa": kappa,
        "a_lat": a_lat,
        "a_lon": a_lon,
        "jerk": jerk,
    }
