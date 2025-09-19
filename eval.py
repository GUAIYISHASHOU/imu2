from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from utils import to_device, load_config_file
from dataset import build_loader
from models import IMURouteModel
from metrics import route_metrics_imu, route_metrics_vis

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
        d_out = sample_batch["E2"].shape[-1]
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
                st = route_metrics_imu(batch["E2"], logv, batch["MASK"],
                                     logv_min=md_args.get("logv_min",-12.0),
                                     logv_max=md_args.get("logv_max",6.0),
                                     yvar=None)  # GNSS也不传yvar
            else:
                st = route_metrics_imu(batch["E2"], logv, batch["MASK"],
                                     logv_min=md_args.get("logv_min",-12.0),
                                     logv_max=md_args.get("logv_max",6.0),
                                     yvar=batch.get("Y", None))
            all_stats.append(st)

    # Average
    keys = all_stats[0].keys()
    agg = {k: float(sum(d[k] for d in all_stats)/len(all_stats)) for k in keys}
    
    # GNSS逐维分析
    if args.route == "gns":
        import numpy as np
        from scipy.stats import chi2
        
        # 收集所有批次数据
        all_err2, all_var_pred, all_mask = [], [], []
        with torch.no_grad():
            for batch in dl:
                batch = to_device(batch, args.device)
                logv = model(batch["X"])
                var = torch.exp(torch.clamp(logv, min=md_args.get("logv_min",-12.0), max=md_args.get("logv_max",6.0)))
                all_err2.append(batch["E2"].cpu().numpy())
                all_var_pred.append(var.cpu().numpy())
                all_mask.append(batch["MASK"].cpu().numpy())
        
        err2 = np.concatenate(all_err2, axis=0)
        var_pred = np.concatenate(all_var_pred, axis=0)
        mask = np.concatenate(all_mask, axis=0)
        
        # 逐维ENU分析
        D = err2.shape[-1]
        axis_names = ['E','N','U'] if D==3 else (['x','y'] if D==2 else [f'd{i}' for i in range(D)])
        per_axis = []
        
        for d in range(D):
            m = mask[..., d] > 0.5
            e2 = err2[..., d][m]
            vp = var_pred[..., d][m]
            z2 = e2 / np.maximum(vp, 1e-12)
            
            per_axis.append({
                "axis": axis_names[d],
                "Ez2": float(np.mean(z2)),
                "cov68": float(np.mean(z2 <= 1.0)),  # 高斯1维: 68% ≈ z²<=1
                "cov95": float(np.mean(z2 <= 3.841)),  # 95% 阈值
                "count": int(e2.size)
            })
        
        agg["per_axis"] = per_axis
    
    print(json.dumps(agg, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
