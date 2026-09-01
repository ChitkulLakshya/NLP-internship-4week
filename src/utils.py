"""Shared utilities: seeding, IO, plotting."""
import os, random, re, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def seed_everything(seed: int = 42):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p

def savefig(path: str, dpi: int = 150):
    ensure_dir(os.path.dirname(path))
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return path

def dump_json(obj, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return path

def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
