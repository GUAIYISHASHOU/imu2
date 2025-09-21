from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from utils import to_device, load_config_file
from dataset import build_loader
from models import IMURouteModel
from metrics import route_metrics_imu, route_metrics_vis, route_metrics_gns_axes

def parse_args():
    # 先解析 --route 参数来确定配置段
    pre_route = argparse.ArgumentParser(add_help=False)
    pre_route.add_argument("--route", choices=["acc","gyr","vis","gns"], default=None)
    args_route, _ = pre_route.parse_known_args()
    
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None, help="YAML/JSON 配置文件")
    args_pre, _ = pre.parse_known_args()

    cfg = load_config_file(args_pre.config)
    
    # 根据 --route 参数读取对应的配置段
    route = args_route.route or "acc"
    if route == "gns":
        ev = cfg.get("eval_gns", cfg.get("eval", {}))
    else:
        ev = cfg.get("eval", {})
    rt = cfg.get("runtime", {})

    ap = argparse.ArgumentParser("Evaluate a trained single-route model", parents=[pre])
    ap.add_argument("--route", choices=["acc","gyr","vis","gns"], default=ev.get("route","acc"))
    ap.add_argument("--npz", required=(ev.get("npz") is None), default=ev.get("npz"))
    ap.add_argument("--model", required=(ev.get("model") is None), default=ev.get("model"))
    ap.add_argument("--x_mode", choices=["both","route_only"], default=ev.get("x_mode","both"))
    ap.add_argument("--device", default=rt.get("device","cuda" if torch.cuda.is_available() else "cpu"))
    # 增加新参数
    ap.add_argument("--nu", type=float, default=0.0, help="Student-t 自由度（评测口径）；0 表示用高斯口径")
    ap.add_argument("--post_scale_json", type=str, default=None, help="按轴温度缩放系数 JSON（评测时应用）")
    ap.add_argument("--est_post_scale_from", type=str, default=None, help="从此 npz（一般是val集）估计按轴温度缩放")
    ap.add_argument("--save_post_scale_to", type=str, default=None, help="把估计的缩放系数保存到 JSON")
    return ap.parse_args()

def _apply_post_scale(logv: torch.Tensor, c_axis: torch.Tensor | None) -> torch.Tensor:
    if c_axis is None:
        return logv
    return logv + c_axis.log().view(1,1,-1).to(logv.device, logv.dtype)

@torch.no_grad()
def _estimate_post_scale_gns_axes(model, dl, device, logv_min, logv_max, nu: float) -> torch.Tensor:
    """在验证集估 c_axis = E[z²]/target（target=t:nu/(nu-2), 否则=1）"""
    num = torch.zeros(3, device=device)
    den = torch.zeros(3, device=device)
    target = nu/(nu-2.0) if (nu and nu>2.0) else 1.0
    for batch in dl:
        b = to_device(batch, device)
        logv = model(b["X"])
        lv = torch.clamp(logv, min=logv_min, max=logv_max)
        v  = torch.exp(lv).clamp_min(1e-12)
        e2 = b["E2_AXES"]; m = b["MASK_AXES"].float()
        z2 = e2 / v
        num += (z2 * m).sum(dim=(0,1))
        den += m.sum(dim=(0,1)).clamp_min(1.0)
    ez2 = num / den
    c = (ez2 / target).clamp_min(1e-6)  # (3,)
    return c.detach()

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

    # Average - 只对数值类型进行聚合
    if all_stats:
        keys = all_stats[0].keys()
        agg = {}
        for k in keys:
            values = [d[k] for d in all_stats]
            # 只对数值类型进行平均
            if all(isinstance(v, (int, float)) for v in values):
                agg[k] = float(sum(values) / len(values))
            else:
                # 对于非数值类型，取第一个值（如列表、字符串等）
                agg[k] = values[0]
    else:
        agg = {}
    
    # ==== GNSS逐维分析（支持t口径+post_scale）====
    if args.route == "gns":
        import numpy as np, json
        # —— 可选：从 val 集估计 post_scale 并存盘 ——
        c_axis = None
        if args.est_post_scale_from:
            ds_val, dl_val = build_loader(args.est_post_scale_from, route="gns", x_mode=args.x_mode,
                                          batch_size=64, shuffle=False, num_workers=0)
            c_axis = _estimate_post_scale_gns_axes(model, dl_val, args.device,
                                                   md_args.get("logv_min",-12.0),
                                                   md_args.get("logv_max",6.0),
                                                   args.nu)
            if args.save_post_scale_to:
                Path(args.save_post_scale_to).write_text(json.dumps({
                    "axis": ["E","N","U"], "c_axis": c_axis.cpu().tolist(),
                    "nu": args.nu, "target": (args.nu/(args.nu-2.0) if (args.nu and args.nu>2.0) else 1.0)
                }, ensure_ascii=False, indent=2))
                print("[post_scale] saved to:", args.save_post_scale_to)
        # —— 可选：从 json 载入 post_scale ——
        if (c_axis is None) and args.post_scale_json:
            js = json.loads(Path(args.post_scale_json).read_text())
            c_axis = torch.tensor(js["c_axis"], dtype=torch.float32, device=args.device)

        # —— 汇总所有批次：注意应用 post_scale 到 logv ——
        all_e2, all_v, all_m = [], [], []
        with torch.no_grad():
            for batch in dl:
                b = to_device(batch, args.device)
                logv = model(b["X"])                       # (B,T,3)
                if c_axis is not None:
                    logv = _apply_post_scale(logv, c_axis) # 应用按轴温度缩放
                lv = torch.clamp(logv, min=md_args.get("logv_min",-12.0), max=md_args.get("logv_max",6.0))
                var = torch.exp(lv).clamp_min(1e-12)
                all_e2.append(b["E2_AXES"].cpu()); all_v.append(var.cpu()); all_m.append(b["MASK_AXES"].cpu())

        e2_axes = torch.cat(all_e2, 0); var_axes = torch.cat(all_v, 0); mask_axes = torch.cat(all_m, 0)

        # 指标（t 口径 + 高斯口径对照 + 可靠性）
        st_axes = route_metrics_gns_axes(e2_axes, var_axes.log(), mask_axes,
                                         md_args.get("logv_min",-12.0), md_args.get("logv_max",6.0),
                                         nu=args.nu)
        # 保存
        out_dir = Path(args.model).parent
        (out_dir/"per_axis.json").write_text(json.dumps(st_axes, ensure_ascii=False, indent=2))
        print("[gns] per-axis metrics saved to", out_dir/"per_axis.json")
        
        agg["per_axis"] = st_axes
    
    print(json.dumps(agg, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
