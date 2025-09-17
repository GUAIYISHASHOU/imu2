from __future__ import annotations
import argparse, os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils import to_device
from dataset import build_loader
from models import IMURouteModel

def parse_args():
    ap = argparse.ArgumentParser("Save diagnostics plots for a single-route model")
    ap.add_argument("--route", choices=["acc","gyr"], required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--x_mode", choices=["both","route_only"], default="both")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    ds, dl = build_loader(args.npz, route=args.route, x_mode=args.x_mode, batch_size=64, shuffle=False, num_workers=0)

    d_in = ds.X_all.shape[-1] if args.x_mode=="both" else 3
    ckpt = torch.load(args.model, map_location="cpu")
    md_args = ckpt.get("args", {})
    model = IMURouteModel(d_in=d_in,
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
        logv = model(batch["X"]).squeeze(-1)
        var = torch.exp(logv)
        e2sum = batch["E2"].squeeze(-1)
        z2 = (e2sum / var) / 3.0
        m = batch["MASK"].float()
        mask_flat = (m > 0.5)

        # Plot 1: histogram of z^2
        z2_np = z2[mask_flat].detach().cpu().numpy().reshape(-1)
        plt.figure()
        plt.hist(z2_np, bins=100)
        plt.title(f"z^2 (normalized by df=3) — route={args.route}")
        plt.xlabel("z^2"); plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "hist_z2.png"))
        plt.close()

        # Plot 2: scatter err^2 vs var
        es = e2sum[mask_flat].detach().cpu().numpy().reshape(-1)
        vv = var[mask_flat].detach().cpu().numpy().reshape(-1)
        plt.figure()
        plt.scatter(es, vv, s=4, alpha=0.5)
        plt.xlabel("pooled err^2"); plt.ylabel("pred var")
        plt.title(f"Scatter pooled err^2 vs var — route={args.route}")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "scatter_err2_vs_var.png"))
        plt.close()

        # Plot 3: time series of logvar (first few sequences)
        lv = logv.detach().cpu().numpy()
        T = lv.shape[1]
        K = min(4, lv.shape[0])
        plt.figure()
        for i in range(K):
            plt.plot(lv[i], label=f"seq{i}")
        plt.title(f"log variance (first {K} seqs) — route={args.route}")
        plt.xlabel("t"); plt.ylabel("log(var)")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "timeseries_logvar.png"))
        plt.close()

if __name__ == "__main__":
    main()
