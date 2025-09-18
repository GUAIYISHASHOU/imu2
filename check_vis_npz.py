#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, numpy as np
from pathlib import Path

def load_meta(meta_path):
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {"unit": "px", "feature_names": None, "quantile_triplets": []}

def coverage(z, thr):
    return float(np.mean(np.abs(z) <= thr))

def acorr_lag1(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 3: return np.nan
    x = x - x.mean()
    c0 = np.dot(x, x)
    c1 = np.dot(x[:-1], x[1:])
    return float(c1 / (c0 + 1e-12))

def approx_unit_check(err_abs_med):
    warn = []
    if err_abs_med < 0.02:
        warn.append(f"像素中位数≈{err_abs_med:.4f} px，疑似‘归一化坐标’而非像素；请统一到 px。")
    if err_abs_med > 20.0:
        warn.append(f"像素中位数≈{err_abs_med:.2f} px，偏大；检查焦距/尺度。")
    return warn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="train/val/test npz 路径")
    ap.add_argument("--meta", default=None, help="vis_meta.json 路径（可选）")
    ap.add_argument("--has_label_var", action="store_true", help="若 npz 内含 Y_vis 或 label_var_px2")
    ap.add_argument("--err_key", default="E_vis", help="误差键名 (T,2)")
    ap.add_argument("--x_key", default="X_vis", help="特征键名 (T,D)")
    ap.add_argument("--mask_key", default="mask_vis", help="掩码键名 (T,2)")
    ap.add_argument("--y_key_candidates", nargs="*", default=["Y_vis","label_var_px2"], help="标签方差键名候选")
    args = ap.parse_args()

    p = Path(args.npz)
    data = np.load(p, allow_pickle=True)
    X, E = data[args.x_key], data[args.err_key]
    M = data[args.mask_key] if args.mask_key in data else np.ones_like(E, dtype=bool)

    # 形状 & 缺失
    T, D = X.shape
    assert E.shape == (T,2), f"E_vis shape={E.shape}, 期望 (T,2)"
    assert M.shape == (T,2), f"mask_vis shape={M.shape}, 期望 (T,2)"
    nan_any = np.isnan(X).any() or np.isnan(E).any()
    print(f"[shape] T={T}, D={D}, NaN? {nan_any}")
    print(f"[mask]  有效覆盖率 u={M[:,0].mean():.3f}, v={M[:,1].mean():.3f}")

    # 单位启发式
    err_abs = np.abs(E[M])
    med = float(np.median(err_abs)) if err_abs.size else np.nan
    warns = approx_unit_check(med)
    if warns:
        for w in warns: print(f"[unit][WARN] {w}")
    else:
        print(f"[unit] 像素中位数≈{med:.3f} px（量级合理）")

    # 分位数单调性（如 meta 提供了 quantile_triplets）
    meta = load_meta(Path(args.meta)) if args.meta else {"quantile_triplets":[]}
    if meta.get("quantile_triplets") and meta.get("feature_names"):
        names = meta["feature_names"]
        for trip in meta["quantile_triplets"]:
            idxs = [names.index(n) for n in trip if n in names]
            if len(idxs) >= 3:
                p10, p50, p90 = X[:, idxs[0]], X[:, idxs[1]], X[:, idxs[2]]
                bad = np.sum(~(p10 <= p50) | ~(p50 <= p90))
                rate = bad / T
                msg = "OK" if bad==0 else f"{bad}帧({rate:.1%})不单调"
                print(f"[quantile] {trip}: {msg}")
    else:
        print("[quantile] 未提供 feature_names/quantile_triplets，跳过单调性检查")

    # 簇状噪声（|e| 的 lag-1 自相关）
    e_u = np.abs(E[:,0][M[:,0]])
    e_v = np.abs(E[:,1][M[:,1]])
    ac_u = acorr_lag1(e_u); ac_v = acorr_lag1(e_v)
    print(f"[cluster] |e_u| lag1 acorr={ac_u:.3f}, |e_v| lag1 acorr={ac_v:.3f}  (>=0.2 通常意味着段落性/相关性存在)")

    # 覆盖率/NIS（若有 label var）
    var_label = None
    if args.has_label_var:
        for k in args.y_key_candidates:
            if k in data:
                var_label = data[k]; break
        if var_label is None:
            print("[NIS] 未找到标签方差键名，跳过")
        else:
            assert var_label.shape == (T,2)
            z2 = np.zeros_like(E); z2[:] = np.nan
            valid = M & (var_label>1e-12)
            z2[valid] = (E[valid]**2) / var_label[valid]
            z2_u = z2[:,0][~np.isnan(z2[:,0])]; z2_v = z2[:,1][~np.isnan(z2[:,1])]
            def rep(name, z):
                if z.size==0: 
                    print(f"[NIS] {name}: 无有效样本"); return
                print(f"[NIS] {name}: E[z²]={np.mean(z):.3f}, cov68={coverage(np.sqrt(z),1.0):.3f}, cov95={coverage(np.sqrt(z),2.0):.3f}, N={z.size}")
            rep('u', z2_u); rep('v', z2_v)
    else:
        print("[NIS] 未声明 has_label_var；若你是监督式，请带上 --has_label_var")

if __name__ == "__main__":
    main()
