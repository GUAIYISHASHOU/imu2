from __future__ import annotations
import argparse, os
from pathlib import Path
import torch
import matplotlib.pyplot as plt

from utils import to_device
from dataset import build_loader
from models import IMURouteModel

def parse_args():
    ap = argparse.ArgumentParser("Save diagnostics plots for a single-route model")
    ap.add_argument("--route", choices=["acc","gyr","vis","gns"], required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--x_mode", choices=["both","route_only"], default="both")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--use_loglog", action="store_true", help="使用对数坐标散点图（推荐）")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    ds, dl = build_loader(args.npz, route=args.route, x_mode=args.x_mode, batch_size=64, shuffle=False, num_workers=0)

    # 动态确定输入/输出维度
    if args.route == "gns":
        sample_batch = next(iter(dl))
        d_in = sample_batch["X"].shape[-1]
        d_out = 1                      # ← GNSS 等方差：单通道 logv
    elif args.route == "vis":
        d_in = ds.X_all.shape[-1]
        d_out = 1
    else:
        d_in = ds.X_all.shape[-1] if args.x_mode=="both" else 3
        d_out = 1
    
    ckpt = torch.load(args.model, map_location="cpu")
    md_args = ckpt.get("args", {})
    model = IMURouteModel(d_in=d_in,
                          d_out=d_out,
                          d_model=md_args.get("d_model",128),
                          n_tcn=md_args.get("n_tcn",4),
                          kernel_size=md_args.get("kernel_size",3),
                          n_layers_tf=md_args.get("n_layers_tf",2),
                          n_heads=md_args.get("n_heads",4),
                          dropout=md_args.get("dropout",0.1))
    model.load_state_dict(ckpt["model"])
    model.to(args.device).eval()

    with torch.no_grad():
        batch = next(iter(dl))
        batch = to_device(batch, args.device)
        logv = model(batch["X"])
        if logv.dim() == 3 and logv.size(-1) == 1:
            logv = logv.squeeze(-1)
        var = torch.exp(logv)
        e2sum = batch["E2"]
        if e2sum.dim() == 3 and e2sum.size(-1) == 1:
            e2sum = e2sum.squeeze(-1)
        mask = batch["MASK"]
        if mask.dim() == 3 and mask.size(-1) == 1:
            mask = mask.squeeze(-1)
        mask = mask.float()
        if args.route == "vis":
            df = 2.0
        elif args.route == "gns":
            df = 3.0  # ENU三维
        else:
            df = 3.0  # IMU三维
        z2 = (e2sum / var) / df
        mask_flat = mask > 0.5

        # Plot 1: histogram of z^2
        z2_np = z2[mask_flat].detach().cpu().numpy().reshape(-1)
        plt.figure()
        plt.hist(z2_np, bins=100)
        plt.title(f"z^2 (df={int(df)}) - route={args.route}")
        plt.xlabel("z^2")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "hist_z2.png"))
        plt.close()

        # Plot 2: scatter err^2 vs var
        if args.use_loglog:
            # 对数散点图：逐窗口散点 + 对数坐标
            m = mask.reshape(-1, mask.shape[-1])  # (B*T, D)
            e2_flat = e2sum.reshape(-1, e2sum.shape[-1])  # (B*T, D)
            var_flat = var.reshape(-1, var.shape[-1])     # (B*T, D)
            
            # 应用mask过滤
            valid_mask = m > 0.5
            e2_valid = e2_flat[valid_mask]
            var_valid = var_flat[valid_mask]
            
            # 如果是多维，取均值聚合到标量
            if e2_valid.dim() > 1 and e2_valid.shape[-1] > 1:
                e2s = e2_valid.mean(dim=-1)
                vps = var_valid.mean(dim=-1)
            else:
                e2s = e2_valid.squeeze(-1) if e2_valid.dim() > 1 else e2_valid
                vps = var_valid.squeeze(-1) if var_valid.dim() > 1 else var_valid
            
            e2s_np = e2s.detach().cpu().numpy()
            vps_np = vps.detach().cpu().numpy()
            
            plt.figure()
            plt.scatter(e2s_np, vps_np, s=6, alpha=0.35)
            plt.xscale('log'); plt.yscale('log')  # 关键：对数坐标
            plt.xlabel('err^2 (per-window, pooled)')
            plt.ylabel('pred var')
            plt.title(f'Scatter (per-window, log-log) - route={args.route}')
            plt.tight_layout()
            plt.savefig(os.path.join(args.out, "scatter_err2_vs_var_loglog.png"))
            plt.close()
        else:
            # 原始散点图
            es = e2sum[mask_flat].detach().cpu().numpy().reshape(-1)
            vv = var[mask_flat].detach().cpu().numpy().reshape(-1)
            plt.figure()
            plt.scatter(es, vv, s=4, alpha=0.5)
            plt.xlabel("pooled err^2")
            plt.ylabel("pred var")
            plt.title(f"Scatter pooled err^2 vs var - route={args.route}")
            plt.tight_layout()
            plt.savefig(os.path.join(args.out, "scatter_err2_vs_var.png"))
            plt.close()

        # Plot 3: time series of logvar (first few sequences)
        if args.route == "gns":
            # GNSS: 等方差时序图 + 逐维分析
            import json
            import numpy as np
            
            # squeeze 到 2D：(B,T)
            if logv.dim()==3 and logv.size(-1)==1: 
                logv = logv.squeeze(-1)
            if e2sum.dim()==3 and e2sum.size(-1)==1: 
                e2sum = e2sum.squeeze(-1)
            if mask.dim()==3 and mask.size(-1)==1: 
                mask = mask.squeeze(-1)

            var = torch.exp(logv).clamp_min(1e-12)      # (B,T)
            lv = logv.detach().cpu().numpy()
            
            # 等方差时序图（单条线）
            plt.figure()
            plt.plot(lv[0], label='logvar (isotropic)')  # 只显示第一个序列
            plt.legend()
            plt.title('log variance (isotropic ENU)')
            plt.xlabel("t")
            plt.ylabel("log(var)")
            plt.tight_layout()
            plt.savefig(os.path.join(args.out, "timeseries_logvar.png"))
            plt.close()
            
            # 逐维表：用逐轴误差 + 同一标量方差
            y_axes = batch["Y"].detach().cpu().numpy()                 # (B,T,3)
            m_axes = batch.get("MASK_AXES", batch["MASK"]).detach().cpu().numpy()  # (B,T,3) or (B,T,1)
            v_np   = var.detach().cpu().numpy()                        # (B,T)

            names = ['E','N','U']
            per_axis = []
            for d, nm in enumerate(names):
                m = (m_axes[..., d] > 0.5) if m_axes.ndim==3 else (m_axes.squeeze(-1) > 0.5)
                z2d = ( (y_axes[..., d]**2) / np.maximum(v_np, 1e-12) )[m]
                per_axis.append({
                    "axis": nm,
                    "Ez2": float(np.mean(z2d)),
                    "cov68": float(np.mean(z2d <= 1.0)),
                    "cov95": float(np.mean(z2d <= 3.841)),
                    "count": int(m.sum())
                })
            
            with open(os.path.join(args.out, 'per_axis.json'),'w',encoding='utf-8') as f:
                json.dump(per_axis, f, ensure_ascii=False, indent=2)
        else:
            # 原始时序图
            lv = logv.detach().cpu().numpy()
            T = lv.shape[1]
            K = min(4, lv.shape[0])
            plt.figure()
            for i in range(K):
                plt.plot(lv[i], label=f"seq{i}")
            plt.title(f"log variance (first {K} seqs) - route={args.route}")
            plt.xlabel("t")
            plt.ylabel("log(var)")
            plt.tight_layout()
            plt.savefig(os.path.join(args.out, "timeseries_logvar.png"))
            plt.close()

if __name__ == "__main__":
    main()
