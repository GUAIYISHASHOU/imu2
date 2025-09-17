from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from utils import to_device, load_config_file
from dataset import build_loader
from models import IMURouteModel
from metrics import route_metrics

def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None, help="YAML/JSON 配置文件")
    args_pre, _ = pre.parse_known_args()

    cfg = load_config_file(args_pre.config)
    ev = cfg.get("eval", {})
    rt = cfg.get("runtime", {})

    ap = argparse.ArgumentParser("Evaluate a trained single-route model", parents=[pre])
    ap.add_argument("--route", choices=["acc","gyr"], required=(ev.get("route") is None), default=ev.get("route"))
    ap.add_argument("--npz", required=(ev.get("npz") is None), default=ev.get("npz"))
    ap.add_argument("--model", required=(ev.get("model") is None), default=ev.get("model"))
    ap.add_argument("--x_mode", choices=["both","route_only"], default=ev.get("x_mode","both"))
    ap.add_argument("--device", default=rt.get("device","cuda" if torch.cuda.is_available() else "cpu"))
    return ap.parse_args()

def main():
    args = parse_args()
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

    all_stats = []
    with torch.no_grad():
        for batch in dl:
            batch = to_device(batch, args.device)
            logv = model(batch["X"])
            st = route_metrics(batch["E2"], logv, batch["MASK"], 
                               logv_min=md_args.get("logv_min",-12.0), 
                               logv_max=md_args.get("logv_max",6.0))
            all_stats.append(st)

    # Average
    keys = all_stats[0].keys()
    agg = {k: float(sum(d[k] for d in all_stats)/len(all_stats)) for k in keys}
    print(json.dumps(agg, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
