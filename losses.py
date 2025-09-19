from __future__ import annotations
import torch

def nll_iso3_e2(e2sum: torch.Tensor, logv: torch.Tensor, mask: torch.Tensor,
                logv_min: float=-16.0, logv_max: float=6.0) -> torch.Tensor:
    """
    Negative log-likelihood using pre-pooled squared error sum.
    e2sum: (B,T,1) or (B,T)
    logv : (B,T,1) or (B,T)
    mask : (B,T)
    """
    if logv.dim() == 3 and logv.size(-1) == 1:
        logv = logv.squeeze(-1)
    if e2sum.dim() == 3 and e2sum.size(-1) == 1:
        e2sum = e2sum.squeeze(-1)
    logv = torch.clamp(logv, min=logv_min, max=logv_max)
    v = torch.exp(logv)
    nll = 0.5 * (3.0 * logv + e2sum / v)
    m = mask.float()
    return (nll * m).sum() / torch.clamp(m.sum(), min=1.0)

def nll_iso2_e2(e2sum: torch.Tensor, logv: torch.Tensor, mask: torch.Tensor,
                logv_min: float = -16.0, logv_max: float = 6.0) -> torch.Tensor:
    """Isotropic 2D negative log-likelihood for vision route."""
    if logv.dim() == 3 and logv.size(-1) == 1:
        logv = logv.squeeze(-1)
    if e2sum.dim() == 3 and e2sum.size(-1) == 1:
        e2sum = e2sum.squeeze(-1)
    logv = torch.clamp(logv, min=logv_min, max=logv_max)
    v = torch.exp(logv)
    m = mask.float()
    nll = 0.5 * (2.0 * logv + e2sum / v)
    return (nll * m).sum() / torch.clamp(m.sum(), min=1.0)


def mse_anchor_1d(logv: torch.Tensor, y_var: torch.Tensor, mask: torch.Tensor, lam: float=1e-3) -> torch.Tensor:
    """
    Optional scale anchor on log-variance.
    y_var: (B,T) anchor variance (>=0), will be log() with clamp.
    """
    if logv.dim() == 3 and logv.size(-1) == 1:
        logv = logv.squeeze(-1)
    y = torch.clamp(y_var, min=1e-12).log()
    m = mask.float()
    se = (logv - y)**2 * m
    return lam * se.sum() / torch.clamp(m.sum(), min=1.0)

def nll_diag_axes(e2_axes: torch.Tensor, logv_axes: torch.Tensor, mask_axes: torch.Tensor,
                  logv_min: float=-16.0, logv_max: float=6.0) -> torch.Tensor:
    """
    各向异性对角高斯 NLL（逐轴）。适用于 GNSS ENU 三轴。
    e2_axes  : (B,T,3)   每轴误差平方
    logv_axes: (B,T,3)   每轴 log(σ^2)
    mask_axes: (B,T,3)   每轴有效掩码
    """
    lv = torch.clamp(logv_axes, min=logv_min, max=logv_max)
    inv_v = torch.exp(-lv)                 # (B,T,3)
    nll = 0.5 * (e2_axes * inv_v + lv)    # (B,T,3)
    m = mask_axes.float()
    num = (nll * m).sum()
    den = torch.clamp(m.sum(), min=1.0)
    return num / den