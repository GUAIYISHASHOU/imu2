from __future__ import annotations
import torch
import numpy as np


def _prepare_inputs(e2sum: torch.Tensor, logv: torch.Tensor, mask: torch.Tensor):
    if logv.dim() == 3 and logv.size(-1) == 1:
        logv = logv.squeeze(-1)
    if e2sum.dim() == 3 and e2sum.size(-1) == 1:
        e2sum = e2sum.squeeze(-1)
    if mask.dim() == 3 and mask.size(-1) == 1:
        mask = mask.squeeze(-1)
    if logv.dim() != 2 or e2sum.dim() != 2 or mask.dim() != 2:
        raise ValueError("Expected (B,T) tensors after squeeze")
    return e2sum, logv, mask


@torch.no_grad()
def _route_metrics(e2sum: torch.Tensor, logv: torch.Tensor, mask: torch.Tensor,
                  logv_min: float, logv_max: float, df: float,
                  yvar: torch.Tensor | None = None) -> dict:
    e2sum, logv, mask = _prepare_inputs(e2sum, logv, mask)
    logv = torch.clamp(logv, min=logv_min, max=logv_max)
    var = torch.clamp(torch.exp(logv), min=1e-12)
    m = mask.float()

    z2 = (e2sum / var) / float(df)
    z2 = torch.clamp(z2, min=0.0)
    msum = torch.clamp(m.sum(), min=1.0)
    z2_mean = float((z2 * m).sum() / msum)
    
    # 直接在z²空间做覆盖率（不取sqrt）
    if abs(df - 2.0) < 1e-6:
        z2_68, z2_95 = 2.27886856637673/2.0, 5.99146454710798/2.0  # χ²₂(0.68)/2, χ²₂(0.95)/2
    elif abs(df - 3.0) < 1e-6:
        z2_68, z2_95 = 3.505882355768183/3.0, 7.814727903251178/3.0  # χ²₃(0.68)/3, χ²₃(0.95)/3
    else:
        z2_68, z2_95 = 1.0, 4.0  # fallback for other df values
    
    cov68 = float((((z2 <= z2_68).float() * m).sum()) / msum)
    cov95 = float((((z2 <= z2_95).float() * m).sum()) / msum)

    # 排序相关性（err² vs var）
    v = torch.exp(torch.clamp(logv, min=logv_min, max=logv_max))
    mask_flat = (m.reshape(-1) > 0).cpu().numpy()
    v_np = v.reshape(-1).detach().cpu().numpy()[mask_flat]
    e_np = e2sum.reshape(-1).detach().cpu().numpy()[mask_flat]
    if v_np.size >= 3:
        rr = np.argsort(np.argsort(e_np))
        vv = np.argsort(np.argsort(v_np))
        spear = float(np.corrcoef(rr, vv)[0, 1])
    else:
        spear = 0.0

    # 饱和分解，便于判断是打上限还是打下限
    lv = torch.clamp(logv, min=logv_min, max=logv_max)
    sat_min = float((((lv <= logv_min).float() * m).sum()) / msum)
    sat_max = float((((lv >= logv_max).float() * m).sum()) / msum)
    sat = sat_min + sat_max

    out = {
        "z2_mean": z2_mean,
        "cov68": cov68,
        "cov95": cov95,
        "spear": spear,
        "sat": sat,
        "sat_min": sat_min,
        "sat_max": sat_max,
        "ez2": z2_mean,
    }

    if yvar is not None:
        if yvar.dim() == 3 and yvar.size(-1) == 1:
            yv = yvar.squeeze(-1)
        else:
            yv = yvar
        yv = torch.clamp(yv, min=1e-12)
        log_bias = float(((logv - yv.log()) * m).sum() / msum)
        log_rmse = float(torch.sqrt(((logv - yv.log()) ** 2 * m).sum() / msum))
        y_np = (yv * m).detach().cpu().numpy().reshape(-1)[mask_flat]
        if y_np.size >= 3:
            ry2 = np.argsort(np.argsort(y_np))
            spear_vy = float(np.corrcoef(np.argsort(np.argsort(vv)), ry2)[0, 1])
        else:
            spear_vy = 0.0
        ez2_true = float((((e2sum / yv) / float(df)) * m).sum() / msum)
        out.update({
            "log_bias": log_bias,
            "log_rmse": log_rmse,
            "spear_v_y": spear_vy,
            "ez2_true": ez2_true,
        })

    return out


@torch.no_grad()
def route_metrics_imu(e2sum: torch.Tensor, logv: torch.Tensor, mask: torch.Tensor,
                     logv_min: float, logv_max: float,
                     yvar: torch.Tensor | None = None) -> dict:
    return _route_metrics(e2sum, logv, mask, logv_min, logv_max, df=3.0, yvar=yvar)


@torch.no_grad()
def route_metrics_vis(e2sum: torch.Tensor, logv: torch.Tensor, mask: torch.Tensor,
                     logv_min: float, logv_max: float,
                     yvar: torch.Tensor | None = None) -> dict:
    return _route_metrics(e2sum, logv, mask, logv_min, logv_max, df=2.0, yvar=yvar)

@torch.no_grad()
def route_metrics_gns_axes(e2_axes: torch.Tensor, logv_axes: torch.Tensor, mask_axes: torch.Tensor,
                           logv_min: float, logv_max: float) -> dict:
    """
    GNSS 各向异性评测：逐轴 z²，再在(B*T*D)维度整体统计。
    """
    lv = torch.clamp(logv_axes, min=logv_min, max=logv_max)         # (B,T,3)
    v  = torch.clamp(torch.exp(lv), min=1e-12)
    m  = mask_axes.float()
    z2 = (e2_axes / v)                                              # 1D z²
    den = torch.clamp(m.sum(), min=1.0)
    z2_mean = float((z2 * m).sum() / den)
    cov68   = float((((z2 <= 1.0).float() * m).sum()) / den)
    cov95   = float((((z2 <= 3.841).float() * m).sum()) / den)
    # 排序相关性（err² vs var）在"逐轴展开"后算
    mask_flat = (m.reshape(-1) > 0).cpu().numpy()
    v_np  = v.reshape(-1).detach().cpu().numpy()[mask_flat]
    e_np  = e2_axes.reshape(-1).detach().cpu().numpy()[mask_flat]
    if v_np.size >= 3:
        rr = np.argsort(np.argsort(e_np))
        vv = np.argsort(np.argsort(v_np))
        spear = float(np.corrcoef(rr, vv)[0, 1])
    else:
        spear = 0.0
    # 饱和率（对所有轴）
    sat_min = float((((lv <= logv_min).float() * m).sum()) / den)
    sat_max = float((((lv >= logv_max).float() * m).sum()) / den)
    return {
        "z2_mean": z2_mean,
        "cov68": cov68,
        "cov95": cov95,
        "spear": spear,
        "sat": sat_min + sat_max,
        "sat_min": sat_min,
        "sat_max": sat_max,
        "ez2": z2_mean,
    }
