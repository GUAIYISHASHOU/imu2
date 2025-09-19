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
    rad = torch.sqrt(z2)
    
    # 使用正确的卡方分位数阈值
    from math import sqrt
    if abs(df - 2.0) < 1e-6:
        r68, r95 = 1.509596, 2.447746  # χ²₂(0.68), χ²₂(0.95)
    elif abs(df - 3.0) < 1e-6:
        r68, r95 = 1.872401, 2.795483  # χ²₃(0.68), χ²₃(0.95)
    else:
        r68, r95 = 1.0, 2.0  # fallback for other df values
    
    cov68 = float((((rad <= r68).float() * m).sum()) / msum)
    cov95 = float((((rad <= r95).float() * m).sum()) / msum)

    es = (e2sum * m).detach().cpu().numpy().reshape(-1)
    vv = (var * m).detach().cpu().numpy().reshape(-1)
    mask_flat = (m.detach().cpu().numpy().reshape(-1) > 0.5)
    es = es[mask_flat]
    vv = vv[mask_flat]
    if es.size < 3:
        spear = 0.0
    else:
        rx = np.argsort(np.argsort(es))
        ry = np.argsort(np.argsort(vv))
        spear = float(np.corrcoef(rx, ry)[0, 1])

    lv = logv.detach()
    sat = ((lv <= logv_min) | (lv >= logv_max)).float()
    sat = float((sat * m).sum() / msum)

    out = {
        "z2_mean": z2_mean,
        "cov68": cov68,
        "cov95": cov95,
        "spear": spear,
        "sat": sat,
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
