from __future__ import annotations
import torch
import numpy as np

@torch.no_grad()
def route_metrics(e2sum: torch.Tensor, logv: torch.Tensor, mask: torch.Tensor,
                  logv_min: float, logv_max: float, yvar: torch.Tensor | None = None) -> dict:
    """
    Return E[z^2] (normalized by df=3), 68% coverage, Spearman rho(e2sum, var), saturation rate.
    """
    if logv.dim() == 3 and logv.size(-1) == 1:
        logv = logv.squeeze(-1)
    if e2sum.dim() == 3 and e2sum.size(-1) == 1:
        e2sum = e2sum.squeeze(-1)
    elif e2sum.dim() != 2:
        raise ValueError("e2sum must be (B,T) or (B,T,1)")
    var = torch.clamp(torch.exp(logv), min=1e-12)
    m = mask.float()
    z2 = (e2sum / var) / 3.0            # normalized to ~1

    msum = torch.clamp(m.sum(), min=1.0)
    ez2 = float((z2 * m).sum() / msum)
    cov1 = float((((z2 <= 1.0).float() * m).sum()) / msum)

    # Spearman corr(err^2_sum, var)
    es = (e2sum * m).detach().cpu().numpy().reshape(-1)
    vv = (var * m).detach().cpu().numpy().reshape(-1)
    mask_flat = (m.detach().cpu().numpy().reshape(-1) > 0.5)
    es = es[mask_flat]; vv = vv[mask_flat]
    if es.size < 3:
        spear = 0.0
    else:
        rx = np.argsort(np.argsort(es))
        ry = np.argsort(np.argsort(vv))
        spear = float(np.corrcoef(rx, ry)[0,1])

    # Saturation: fraction of timesteps at clamp bounds
    lv = logv.detach()
    sat = ((lv <= logv_min) | (lv >= logv_max)).float()
    sat = float((sat * m).sum() / msum)

    out = {"ez2": ez2, "cov68": cov1, "spear": spear, "sat": sat}

    # Optional: compare predicted variance with oracle variance if provided
    if yvar is not None:
        if yvar.dim() == 3 and yvar.size(-1) == 1:
            yv = yvar.squeeze(-1)
        else:
            yv = yvar
        yv = torch.clamp(yv, min=1e-12)
        # Geometric mean bias in log-domain (ideal ~0)
        log_bias = float((((logv - yv.log()) * m).sum()) / msum)
        # Log RMSE between predicted and oracle variance (scale-invariant)
        log_rmse = float(torch.sqrt((((logv - yv.log())**2 * m).sum()) / msum))
        # Spearman between predicted var and oracle var
        y_np = (yv * m).detach().cpu().numpy().reshape(-1)[mask_flat]
        if y_np.size >= 3:
            ry2 = np.argsort(np.argsort(y_np))
            spear_vy = float(np.corrcoef(np.argsort(np.argsort(vv)), ry2)[0,1])
        else:
            spear_vy = 0.0
        # ez2 using oracle variance (sanity check ~1)
        ez2_true = float((((e2sum / yv) / 3.0) * m).sum() / msum)
        out.update({"log_bias": log_bias, "log_rmse": log_rmse,
                    "spear_v_y": spear_vy, "ez2_true": ez2_true})

    return out
