from __future__ import annotations
import torch
import numpy as np

@torch.no_grad()
def route_metrics(e2sum: torch.Tensor, logv: torch.Tensor, mask: torch.Tensor,
                  logv_min: float, logv_max: float) -> dict:
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

    return {"ez2": ez2, "cov68": cov1, "spear": spear, "sat": sat}
