from __future__ import annotations
import argparse, os
from pathlib import Path
import torch
import matplotlib.pyplot as plt

from utils import to_device, load_config_file
from dataset import build_loader
from models import IMURouteModel

def parse_args():
    # 先解析 --config 和 --route 参数
    pre_route = argparse.ArgumentParser(add_help=False)
    pre_route.add_argument("--route", choices=["acc","gyr","vis","gns"], default=None)
    args_route, _ = pre_route.parse_known_args()
    
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None, help="YAML/JSON 配置文件")
    args_pre, _ = pre.parse_known_args()

    # 加载配置文件
    cfg = load_config_file(args_pre.config) if args_pre.config else {}
    
    # 根据 --route 参数读取对应的配置段
    route = args_route.route
    if route == "gns" and args_pre.config:
        ev = cfg.get("eval_gns", cfg.get("eval", {}))
        an = cfg.get("analyze_gns", cfg.get("analyze", {}))
    else:
        ev = cfg.get("eval", {})
        an = cfg.get("analyze", {})
    rt = cfg.get("runtime", {})

    ap = argparse.ArgumentParser("Save diagnostics plots for a single-route model", parents=[pre])
    ap.add_argument("--route", choices=["acc","gyr","vis","gns"], 
                    required=(route is None), default=route or ev.get("route"))
    ap.add_argument("--npz", required=(ev.get("npz") is None), default=ev.get("npz"))
    ap.add_argument("--model", required=(ev.get("model") is None), default=ev.get("model"))
    ap.add_argument("--x_mode", choices=["both","route_only"], default=ev.get("x_mode", "both"))
    ap.add_argument("--out", required=(an.get("out") is None), default=an.get("out"))
    ap.add_argument("--device", default=rt.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--use_loglog", action="store_true", default=an.get("use_loglog", False),
                    help="使用对数坐标散点图（推荐）")
    ap.add_argument("--nu", type=float, default=0.0, help="Student-t 自由度（作图口径）；0 表示只画高斯口径")
    ap.add_argument("--post_scale_json", type=str, default=None, help="评图时应用按轴温度缩放 JSON")
    return ap.parse_args()

def _t_thr(nu, coverages=(0.68,0.95)):
    if not (nu and nu>2.0): 
        return {}
    import torch
    t = torch.distributions.StudentT(df=nu)
    th = {}
    for p in coverages:
        q = float(t.icdf(torch.tensor([(1+p)/2.0]))[0])
        th[p] = q*q  # z2 阈值
    return th

def _apply_post_scale(logv: torch.Tensor, c_axis: torch.Tensor | None) -> torch.Tensor:
    if c_axis is None:
        return logv
    return logv + c_axis.log().view(1,1,-1).to(logv.device, logv.dtype)

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    ds, dl = build_loader(args.npz, route=args.route, x_mode=args.x_mode, batch_size=64, shuffle=False, num_workers=0)

    # 动态确定输入/输出维度
    if args.route == "gns":
        sample_batch = next(iter(dl))
        d_in = sample_batch["X"].shape[-1]
        d_out = 3                      # ← 各向异性 ENU
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

    # 载入post_scale（可选）
    c_axis = None
    if args.post_scale_json:
        import json
        js = json.loads(Path(args.post_scale_json).read_text())
        c_axis = torch.tensor(js["c_axis"], dtype=torch.float32, device=args.device)

    with torch.no_grad():
        batch = next(iter(dl))
        batch = to_device(batch, args.device)
        logv = model(batch["X"])
        
        # 应用post_scale（如果有）
        if c_axis is not None and args.route == "gns":
            logv = _apply_post_scale(logv, c_axis)
            
        if args.route == "vis":
            df = 2.0
        else:
            df = 3.0  # IMU三维

        # --- GNSS: 逐轴 ---
        if args.route == "gns":
            # 与eval.py保持一致的clamp范围
            logv_clamped = torch.clamp(logv, min=-12.0, max=6.0)
            var_axes = torch.exp(logv_clamped)              # (B,T,3)
            e2_axes  = batch["E2_AXES"]                     # (B,T,3)
            m_axes   = batch["MASK_AXES"].float()           # (B,T,3)
            z2_axes  = e2_axes / torch.clamp(var_axes, 1e-12)
            
            # 为每个轴分别提取z2数据
            z2_E = z2_axes[:,:,0][m_axes[:,:,0] > 0.5].detach().cpu().numpy()
            z2_N = z2_axes[:,:,1][m_axes[:,:,1] > 0.5].detach().cpu().numpy()
            z2_U = z2_axes[:,:,2][m_axes[:,:,2] > 0.5].detach().cpu().numpy()
            
            mask_flat = (m_axes > 0.5)
            z2_np = z2_axes[mask_flat].detach().cpu().numpy().reshape(-1)
        else:
            if logv.dim() == 3 and logv.size(-1) == 1:
                logv = logv.squeeze(-1)
            # 与eval.py保持一致的clamp范围
            logv_clamped = torch.clamp(logv, min=-12.0, max=6.0)
            var = torch.exp(logv_clamped)
            e2sum = batch["E2"]
            if e2sum.dim() == 3 and e2sum.size(-1) == 1:
                e2sum = e2sum.squeeze(-1)
            mask = batch["MASK"]
            if mask.dim() == 3 and mask.size(-1) == 1:
                mask = mask.squeeze(-1)
            mask = mask.float()
            z2 = (e2sum / var) / df
            mask_flat = mask > 0.5
            z2_np = z2[mask_flat].detach().cpu().numpy().reshape(-1)

        # Plot 1: histogram of z^2
        if args.route == "gns":
            # GNSS: 为每个轴绘制直方图，支持t口径和高斯口径对照
            thr_t = _t_thr(args.nu)
            thr_g = {0.68:1.0, 0.95:3.841}
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for axis_name, z2_vec, ax in zip(["E","N","U"], [z2_E, z2_N, z2_U], axes):
                ax.hist(z2_vec, bins=100, alpha=0.6, density=True)
                
                # t口径阈值线（如果有）
                if thr_t:
                    for p,v in thr_t.items(): 
                        cov_t = np.mean(z2_vec <= v)
                        ax.axvline(v, linestyle="--", label=f"t(ν={args.nu}) {int(p*100)}%: {cov_t:.3f}")
                
                # 高斯口径阈值线
                for p,v in thr_g.items(): 
                    cov_g = np.mean(z2_vec <= v)
                    ax.axvline(v, linestyle=":", label=f"Gaussian {int(p*100)}%: {cov_g:.3f}")
                
                ax.set_title(f"{axis_name}: z^2 histogram (t & Gaussian thresholds)")
                ax.set_xlabel("z^2"); ax.set_ylabel("density"); ax.legend(); ax.grid(True, alpha=0.3)
            
            plt.suptitle("GNSS z^2 Histograms with Coverage Thresholds")
            plt.tight_layout()
            plt.savefig(os.path.join(args.out, "hist_z2.png"))
            plt.close()
        else:
            # 其他路由：原有逻辑
            hist_df = int(df)
            plt.figure()
            plt.hist(z2_np, bins=100)
            plt.title(f"z^2 (df={hist_df}) - route={args.route}")
            plt.xlabel("z^2")
            plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(os.path.join(args.out, "hist_z2.png"))
            plt.close()

        # Plot 2: scatter err^2 vs var
        if args.use_loglog:
            # 对数散点图：逐窗口散点 + 对数坐标
            if args.route == "gns":
                # GNSS: 为每个轴分别绘制散点图
                m = m_axes.reshape(-1, m_axes.shape[-1])      # (B*T,3)
                e2_flat = e2_axes.reshape(-1, e2_axes.shape[-1])
                var_flat = var_axes.reshape(-1, var_axes.shape[-1])
                
                axis_names = ['E', 'N', 'U']
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                for i, (ax, axis_name) in enumerate(zip(axes, axis_names)):
                    # 每个轴单独处理
                    valid_mask = m[:, i] > 0.5
                    e2_axis = e2_flat[valid_mask, i].detach().cpu().numpy()
                    var_axis = var_flat[valid_mask, i].detach().cpu().numpy()
                    
                    ax.scatter(e2_axis, var_axis, s=4, alpha=0.4)
                    ax.set_xscale('log'); ax.set_yscale('log')
                    ax.set_xlabel(f'err^2 ({axis_name} axis)')
                    ax.set_ylabel(f'pred var ({axis_name} axis)')
                    ax.set_title(f'{axis_name} Axis')
                    ax.grid(True, alpha=0.3)
                
                plt.suptitle(f'GNSS Scatter (per-axis, log-log) - route={args.route}')
                plt.tight_layout()
                plt.savefig(os.path.join(args.out, "scatter_err2_vs_var_loglog.png"))
                plt.close()
            else:
                # 其他路由：单轴处理
                m = mask.reshape(-1, mask.shape[-1])          # (B*T,1)
                e2_flat = e2sum.reshape(-1, e2sum.shape[-1])
                var_flat = var.reshape(-1, var.shape[-1])
                
                # 应用mask过滤
                valid_mask = m > 0.5
                e2_valid = e2_flat[valid_mask]
                var_valid = var_flat[valid_mask]
                
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
            if args.route == "gns":
                # GNSS: 为每个轴分别绘制散点图
                axis_names = ['E', 'N', 'U']
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                for i, (ax, axis_name) in enumerate(zip(axes, axis_names)):
                    # 每个轴单独处理
                    valid_mask = m_axes[:, :, i] > 0.5
                    e2_axis = e2_axes[valid_mask, i].detach().cpu().numpy()
                    var_axis = var_axes[valid_mask, i].detach().cpu().numpy()
                    
                    ax.scatter(e2_axis, var_axis, s=4, alpha=0.5)
                    ax.set_xlabel(f'err^2 ({axis_name} axis)')
                    ax.set_ylabel(f'pred var ({axis_name} axis)')
                    ax.set_title(f'{axis_name} Axis')
                    ax.grid(True, alpha=0.3)
                
                plt.suptitle(f'GNSS Scatter (per-axis) - route={args.route}')
                plt.tight_layout()
                plt.savefig(os.path.join(args.out, "scatter_err2_vs_var.png"))
                plt.close()
            else:
                # 其他路由：使用聚合数据
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
            # GNSS: 三轴 logvar 时序 + 逐维指标表
            import json
            import numpy as np
            lv = logv_clamped.detach().cpu().numpy()    # (B,T,3) 使用clamped版本

            # 三轴 logvar（展示第一个序列）
            plt.figure()
            for d, name in enumerate(['E','N','U']):
                plt.plot(lv[0,:,d], label=f'logvar {name}')
            plt.legend()
            plt.title('GNSS log variance (anisotropic ENU)')
            plt.xlabel("t")
            plt.ylabel("log(var)")
            plt.tight_layout()
            plt.savefig(os.path.join(args.out, "timeseries_logvar.png"))
            plt.close()
            
            # 逐维表：用逐轴误差 + 逐轴方差
            y_axes = batch["Y"].detach().cpu().numpy()                 # (B,T,3)
            m_axes = batch["MASK_AXES"].detach().cpu().numpy()         # (B,T,3)
            v_np   = var_axes.detach().cpu().numpy()                   # (B,T,3) 使用clamped版本

            names = ['E','N','U']
            per_axis = []
            for d, nm in enumerate(names):
                m = (m_axes[..., d] > 0.5)
                z2d = ( (y_axes[..., d]**2) / np.maximum(v_np[..., d], 1e-12) )[m]
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
            lv = logv_clamped.detach().cpu().numpy()  # 使用clamped版本
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
