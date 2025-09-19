from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict

def _get(arrs: dict, keys, default=None):
    for k in keys:
        if k in arrs: 
            return arrs[k]
    return default

def _ensure_bool_mask(m):
    m = m.astype(np.float32)
    m = (m > 0.5).astype(np.float32)
    return m

class IMURouteDataset(Dataset):
    def __init__(self, npz_path: str | Path, route: str = "acc", x_mode: str = "both"):
        self.npz_path = str(npz_path)
        self.route = route
        self.x_mode = x_mode
        assert route in ("acc","gyr","vis")
        assert x_mode in ("both","route_only")
        if self.route == "vis" and self.x_mode != "both":
            raise ValueError("Vision route only supports x_mode='both'")

        data = np.load(self.npz_path, allow_pickle=True)
        X = _get(data, ["X","X_imu_seq","imu_seq","imu"], None)
        if X is None:
            raise ValueError(f"{self.npz_path}: missing X")
        E2 = _get(data, ["E2","E2_sum","E2sum"], None)
        if E2 is None:
            E = _get(data, ["E","E_imu","err","errors"], None)
            if E is None:
                raise ValueError(f"{self.npz_path}: missing E2/E")
            E = E.astype(np.float32)
            if E.shape[-1] >= 6:
                acc_e2 = np.sum(E[..., :3]**2, axis=-1, keepdims=True)
                gyr_e2 = np.sum(E[..., 3:6]**2, axis=-1, keepdims=True)
                E2 = np.concatenate([acc_e2, gyr_e2], axis=-1)
            else:
                E2 = np.sum(E**2, axis=-1, keepdims=True)
        M = _get(data, ["MASK","y_mask","mask"], None)
        if M is None:
            raise ValueError(f"{self.npz_path}: missing MASK/y_mask")

        assert X.ndim == 3 and X.shape[-1] >= 3
        if E2.ndim == 2:
            E2 = E2[..., None]
        assert E2.ndim == 3
        
        # 容错3D mask并合并为2D
        M = M.astype(np.float32)
        if M.ndim == 3:
            # 与标签对齐：若任一轴无效则该时刻无效（AND）
            M = (M > 0.5).all(axis=-1).astype(np.float32)  # (N,T)
        assert M.ndim == 2 and M.shape[0] == X.shape[0] and M.shape[1] == X.shape[1], "Expected MASK of shape (N,T) after collapsing"

        self.X_all = X.astype(np.float32)
        self.E2_all = E2.astype(np.float32)
        self.M_all = _ensure_bool_mask(M)

        self.Y_acc = _get(data, ["Y_ACC","Y_acc","Yacc"], None)
        self.Y_gyr = _get(data, ["Y_GYR","Y_gyr","Ygyr"], None)
        if self.Y_acc is not None:
            self.Y_acc = self.Y_acc.astype(np.float32)
        if self.Y_gyr is not None:
            self.Y_gyr = self.Y_gyr.astype(np.float32)

        self.N, self.T, self.D = self.X_all.shape

    def __len__(self):
        return self.N

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        X = self.X_all[idx]
        E2 = self.E2_all[idx]
        M = self.M_all[idx]

        if self.route == "acc":
            E2_route = E2[..., 0:1] if E2.shape[-1] > 1 else E2
            Y = self.Y_acc[idx] if self.Y_acc is not None else None
            if self.x_mode == "route_only" and X.shape[-1] >= 6:
                X = X[..., :3]
        elif self.route == "gyr":
            if E2.shape[-1] == 1:
                E2_route = E2
            elif E2.shape[-1] >= 2:
                E2_route = E2[..., 1:2]
            else:
                raise ValueError("E2 must have >=1 channels")
            Y = self.Y_gyr[idx] if self.Y_gyr is not None else None
            if self.x_mode == "route_only" and X.shape[-1] >= 6:
                X = X[..., 3:6]
        else:
            E2_route = E2[..., :1] if E2.shape[-1] >= 1 else E2
            Y = None

        out = {
            "X": torch.from_numpy(X),
            "MASK": torch.from_numpy(M),
        }
        out["E2"] = torch.from_numpy(E2_route.astype(np.float32))
        if Y is not None:
            out["Y"] = torch.from_numpy(Y)
        else:
            out["Y"] = torch.zeros_like(out["MASK"])
        return out

# === GNSS 数据集（ENU三维） ===
class GNSDataset(Dataset):
    def __init__(self, npz_path: str):
        z = np.load(npz_path, allow_pickle=True)
        self.X = z['X'].astype(np.float32)     # (N, T, Din)
        self.Y = z['Y'].astype(np.float32)     # (N, T, 3)  ENU误差
        self.mask = z['mask'].astype(bool)     # (N, T, 3)
        self.meta = z.get('meta', None)
        assert self.X.shape[0] == self.Y.shape[0] == self.mask.shape[0]
        assert self.Y.shape[-1] == 3, "GNS Y should be (..,3) for ENU"
    
    def __len__(self):  
        return self.X.shape[0]
    
    def __getitem__(self, i):
        y_axes = self.Y[i].astype(np.float32)            # (T,3)
        e2_axes = (y_axes ** 2).astype(np.float32)       # (T,3)
        e2_sum  = e2_axes.sum(axis=-1, keepdims=True)    # (T,1)  ← 训练/评测用
        m_axes  = self.mask[i].astype(np.float32)        # (T,3)
        m_any   = (m_axes > 0.5).all(axis=-1, keepdims=True).astype(np.float32)  # (T,1)

        return {
            "X": torch.from_numpy(self.X[i]),            # (T,Din)
            "E2": torch.from_numpy(e2_sum),              # (T,1)  ← 配合 nll_iso3_e2
            "MASK": torch.from_numpy(m_any),             # (T,1)  ← 与上对齐
            # 下面是作图/逐维统计需要的"富信息"
            "Y": torch.from_numpy(y_axes),               # (T,3)
            "MASK_AXES": torch.from_numpy(m_axes),       # (T,3)
            "E2_AXES": torch.from_numpy(e2_axes),        # (T,3)
        }

def build_dataset(route: str, npz_path: str):
    """数据集工厂函数"""
    route = route.lower()
    if route in ('acc', 'gyr', 'vis'):
        return IMURouteDataset(npz_path, route=route, x_mode="both")
    elif route == 'gns':
        return GNSDataset(npz_path)
    else:
        raise ValueError(f"Unknown route {route}")

def build_loader(npz_path, route="acc", x_mode="both",
                 batch_size=32, shuffle=True, num_workers=0):
    if route.lower() == 'gns':
        ds = build_dataset(route, npz_path)
    else:
        ds = IMURouteDataset(npz_path, route=route, x_mode=x_mode)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return ds, dl
