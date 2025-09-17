import math, os, random, json
from pathlib import Path
import numpy as np
import torch

def seed_everything(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_config_file(path: str | None):
    """
    Load a config file (YAML or JSON). Returns {} if path is None.
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    suffix = p.suffix.lower()
    text = p.read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(text)
    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError("Reading YAML requires PyYAML: pip install pyyaml") from e
        return yaml.safe_load(text) or {}
    raise ValueError(f"Unsupported config format: {suffix}. Use .json or .yaml")
