from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from utils import to_device, load_config_file
from dataset import build_loader
from models import IMURouteModel
from metrics import route_metrics_imu, route_metrics_vis, route_metrics_gns_axes

def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None, help="YAML/JSON 配置文件")
    args_pre, _ = pre.parse_known_args()

    cfg = load_config_file(args_pre.config)
    
    # 根据路由读取不同的配置段
    route_prefix = ""
    if args_pre.config and "gns" in str(args_pre.config).lower():
        route_prefix = "gns_"
    
    ev = cfg.get(f"eval_{route_prefix}".rstrip("_"), cfg.get("eval", {}))
    rt = cfg.get("runtime", {})

    ap = argparse.ArgumentParser("Evaluate a trained single-route model", parents=[pre])
    ap.add_argument("--route", choices=["acc","gyr","vis","gns"], default=ev.get("route","acc"))
    ap.add_argument("--npz", required=(ev.get("npz") is None), default=ev.get("npz"))
    ap.add_argument("--model", required=(ev.get("model") is None), default=ev.get("model"))
    ap.add_argument("--x_mode", choices=["both","route_only"], default=ev.get("x_mode","both"))
    ap.add_argument("--device", default=rt.get("device","cuda" if torch.cuda.is_available() else "cpu"))
    return ap.parse_args()

def main():
    args = parse_args()
    ds, dl = build_loader(args.npz, route=args.route, x_mode=args.x_mode, batch_size=64, shuffle=False, num_workers=0)

    # 动态确定输入/输出维度
    if args.route == "gns":
        sample_batch = next(iter(dl))
        d_in = sample_batch["X"].shape[-1]
        d_out = 3                      # ← 各向异性：ENU 三通道
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

    all_stats = []
    with torch.no_grad():
        for batch in dl:
            batch = to_device(batch, args.device)
            logv = model(batch["X"])
            if args.route == "vis":
                st = route_metrics_vis(batch["E2"], logv, batch["MASK"],
                                     logv_min=md_args.get("logv_min",-12.0),
                                     logv_max=md_args.get("logv_max",6.0),
                                     yvar=None)  # VIS路由不传yvar，避免异常指标
            elif args.route == "gns":
                st = route_metrics_gns_axes(batch["E2_AXES"], logv, batch["MASK_AXES"],
                                     logv_min=md_args.get("logv_min",-12.0),
                                     logv_max=md_args.get("logv_max",6.0))
            else:
                st = route_metrics_imu(batch["E2"], logv, batch["MASK"],
                                     logv_min=md_args.get("logv_min",-12.0),
                                     logv_max=md_args.get("logv_max",6.0),
                                     yvar=batch.get("Y", None))
            all_stats.append(st)

    # Average
    keys = all_stats[0].keys()
    agg = {k: float(sum(d[k] for d in all_stats)/len(all_stats)) for k in keys}
    
    # GNSS逐维分析（汇总所有批次）
    if args.route == "gns":
        import numpy as np
        from scipy.stats import chi2
        
        # 收集所有批次的逐轴数据
        all_y_axes, all_var_axes, all_mask_axes = [], [], []
        with torch.no_grad():
            for batch in dl:
                batch = to_device(batch, args.device)
                logv = model(batch["X"])                        # (B,T,3)
                var  = torch.exp(torch.clamp(logv, min=md_args.get("logv_min",-12.0),
                                              max=md_args.get("logv_max",6.0))).cpu().numpy()
                all_y_axes.append(batch["Y"].cpu().numpy())            # (B,T,3)
                all_var_axes.append(var)                               # (B,T,3)
                all_mask_axes.append(batch["MASK_AXES"].cpu().numpy()) # (B,T,3)

        y_axes = np.concatenate(all_y_axes, axis=0)
        var_axes = np.concatenate(all_var_axes, axis=0)
        mask_axes = np.concatenate(all_mask_axes, axis=0)

        D = y_axes.shape[-1]
        axis_names = ['E','N','U'] if D==3 else [f'd{i}' for i in range(D)]
        per_axis = []
        
        for d in range(D):
            m = mask_axes[..., d] > 0.5
            e2 = (y_axes[..., d]**2)[m]
            vp = var_axes[..., d][m]
            z2 = e2 / np.maximum(vp, 1e-12)
            
            per_axis.append({
                "axis": axis_names[d],
                "Ez2": float(np.mean(z2)),
                "cov68": float(np.mean(z2 <= 1.0)),     # 1D: 68%
                "cov95": float(np.mean(z2 <= 3.841)),   # 1D: 95%
                "count": int(e2.size)
            })
        
        agg["per_axis"] = per_axis
    
    print(json.dumps(agg, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
