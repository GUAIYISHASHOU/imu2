from __future__ import annotations
import argparse, os, json, time
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW

from utils import seed_everything, to_device, count_params, load_config_file
from dataset import build_loader
from models import IMURouteModel
from losses import nll_iso3_e2, nll_iso2_e2, mse_anchor_1d
from metrics import route_metrics_imu, route_metrics_vis

def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None, help="YAML/JSON 配置文件")
    args_pre, _ = pre.parse_known_args()

    cfg = load_config_file(args_pre.config)
    tr = cfg.get("train", {})
    md = cfg.get("model", {})
    rt = cfg.get("runtime", {})

    ap = argparse.ArgumentParser("Train single-route IMU variance model", parents=[pre])
    ap.add_argument("--route", choices=["acc","gyr","vis"], default=tr.get("route","acc"), help="Which route to train")
    ap.add_argument("--train_npz", required=(tr.get("train_npz") is None), default=tr.get("train_npz"))
    ap.add_argument("--val_npz", required=(tr.get("val_npz") is None), default=tr.get("val_npz"))
    ap.add_argument("--test_npz", required=(tr.get("test_npz") is None), default=tr.get("test_npz"))
    ap.add_argument("--x_mode", choices=["both","route_only"], default=tr.get("x_mode","both"))
    ap.add_argument("--run_dir", required=(tr.get("run_dir") is None), default=tr.get("run_dir"))
    ap.add_argument("--epochs", type=int, default=tr.get("epochs",20))
    ap.add_argument("--batch_size", type=int, default=tr.get("batch_size",32))
    ap.add_argument("--lr", type=float, default=tr.get("lr",1e-3))
    ap.add_argument("--seed", type=int, default=tr.get("seed",0))
    ap.add_argument("--d_model", type=int, default=md.get("d_model",128))
    ap.add_argument("--n_tcn", type=int, default=md.get("n_tcn",4))
    ap.add_argument("--kernel_size", type=int, default=md.get("kernel_size",3))
    ap.add_argument("--n_heads", type=int, default=md.get("n_heads",4))
    ap.add_argument("--n_layers_tf", type=int, default=md.get("n_layers_tf",2))
    ap.add_argument("--dropout", type=float, default=md.get("dropout",0.1))
    ap.add_argument("--num_workers", type=int, default=rt.get("num_workers",0))
    ap.add_argument("--logv_min", type=float, default=tr.get("logv_min",-12.0))
    ap.add_argument("--logv_max", type=float, default=tr.get("logv_max",6.0))
    ap.add_argument("--z2_center", type=float, default=tr.get("z2_center",0.0), help="z²居中正则化权重")
    ap.add_argument("--z2_center_target", type=str, default=tr.get("z2_center_target","auto"), help="z²目标值: 'auto' 或数字")
    ap.add_argument("--anchor_weight", type=float, default=tr.get("anchor_weight",0.0))
    ap.add_argument("--early_patience", type=int, default=tr.get("early_patience", 10))
    ap.add_argument("--device", default=rt.get("device","cuda" if torch.cuda.is_available() else "cpu"))
    return ap.parse_args()

def main():
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.run_dir, exist_ok=True)

    # Data
    train_ds, train_dl = build_loader(args.train_npz, route=args.route, x_mode=args.x_mode,
                                      batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers)
    val_ds,   val_dl   = build_loader(args.val_npz,   route=args.route, x_mode=args.x_mode,
                                      batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers)
    test_ds,  test_dl  = build_loader(args.test_npz,  route=args.route, x_mode=args.x_mode,
                                      batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers)

    if args.route == "vis":
        d_in = train_ds.X_all.shape[-1]
    else:
        d_in = train_ds.X_all.shape[-1] if args.x_mode=="both" else 3
    model = IMURouteModel(d_in=d_in, d_model=args.d_model, n_tcn=args.n_tcn, kernel_size=args.kernel_size,
                          n_layers_tf=args.n_layers_tf, n_heads=args.n_heads, dropout=args.dropout).to(args.device)
    print(f"[model] params={count_params(model):,}  d_in={d_in}")

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_val = 1e9
    epochs_since_improve = 0
    best_path = str(Path(args.run_dir) / "best.pt")

    def run_epoch(loader, training: bool):
        model.train(training)
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            batch = to_device(batch, args.device)
            x, m, y = batch["X"], batch["MASK"], batch["Y"]
            logv = model(x)
            if args.route == "vis":
                loss = nll_iso2_e2(batch["E2"], logv, m,
                                   logv_min=args.logv_min, logv_max=args.logv_max)
            else:
                loss = nll_iso3_e2(batch["E2"], logv, m,
                                   logv_min=args.logv_min, logv_max=args.logv_max)
                if args.anchor_weight > 0:
                    loss = loss + mse_anchor_1d(logv, y, m, lam=args.anchor_weight)
            
            # z²居中正则化（通用于VIS和IMU）
            if args.z2_center > 0:
                # 与 NLL 一致地 clamp，再求方差
                lv = torch.clamp(logv, min=args.logv_min, max=args.logv_max)
                v = torch.exp(lv).clamp_min(1e-12)
                df = 2.0 if args.route == "vis" else 3.0
                e2 = batch["E2"].squeeze(-1)
                m_float = m.float().squeeze(-1)
                
                z2 = (e2 / v.squeeze(-1)) / df
                mean_z2 = (z2 * m_float).sum() / m_float.clamp_min(1.0).sum()
                
                # 目标值：高斯=1；若是 Student-t 则 nu/(nu-2)
                if args.z2_center_target == "auto":
                    target = 1.0  # 默认高斯目标
                else:
                    target = float(args.z2_center_target)
                
                loss = loss + args.z2_center * (mean_z2 - target).pow(2)
            if training:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total_loss += float(loss.detach().cpu())
            n_batches += 1
        return total_loss / max(n_batches, 1)

    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        tr_loss = run_epoch(train_dl, True)
        val_loss = run_epoch(val_dl, False)

        # Validation metrics（抽一批看 ez2/coverage/Spearman/饱和率）
        with torch.no_grad():
            model.eval()
            val_batch = next(iter(val_dl))
            val_batch = to_device(val_batch, args.device)
            logv = model(val_batch["X"])
            if args.route == "vis":
                stats = route_metrics_vis(val_batch["E2"], logv, val_batch["MASK"], args.logv_min, args.logv_max)
            else:
                stats = route_metrics_imu(val_batch["E2"], logv, val_batch["MASK"], args.logv_min, args.logv_max)

        print(f"[epoch {epoch:03d}] train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  "
              f"z2_mean={stats['z2_mean']:.3f} cov68={stats['cov68']:.3f} cov95={stats['cov95']:.3f} "
              f"spear={stats['spear']:.3f} sat={stats['sat']:.3f}  time={time.time()-t0:.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            epochs_since_improve = 0
            torch.save({"model": model.state_dict(), "args": vars(args)}, best_path)
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= args.early_patience:
                print(f"[early-stop] No improvement for {args.early_patience} epochs. Stopping at epoch {epoch}.")
                break

    # Final test - iterate over all batches like eval.py
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(args.device).eval()
    
    agg, n = None, 0
    with torch.no_grad():
        for batch in test_dl:
            batch = to_device(batch, args.device)
            logv = model(batch["X"])
            if args.route == "vis":
                st = route_metrics_vis(batch["E2"], logv, batch["MASK"], args.logv_min, args.logv_max)
            else:
                st = route_metrics_imu(batch["E2"], logv, batch["MASK"], args.logv_min, args.logv_max)
            if agg is None: 
                agg = {k: 0.0 for k in st}
            for k, v in st.items(): 
                agg[k] += float(v)
            n += 1
    tst = {k: v/n for k, v in agg.items()}
    
    with open(Path(args.run_dir)/"final_test_metrics.json","w",encoding="utf-8") as f:
        json.dump(tst, f, ensure_ascii=False, indent=2)
    print("[test]", tst)

if __name__ == "__main__":
    main()
