import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
import numpy as np
import torch


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def append_csv(path, row):
    path = Path(path)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            w.writeheader()
        w.writerow(row)


def provenance():
    root = Path(__file__).resolve().parents[1]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for folder in (root / "racing", root / "tests") for p in sorted(folder.glob("*.py"))}
    try:
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        gpu = None
    return dict(platform=platform.platform(), python=platform.python_version(),
        torch=torch.__version__, cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available(),
        gpu=gpu, packages={p: importlib.metadata.version(p) for p in
            ["gymnasium", "box2d-py", "numpy", "opencv-python-headless", "pygame"]}, source_sha256=hashes)


def summary(rows):
    scores = np.array([r["return_raw"] for r in rows])
    return dict(episodes=len(rows), mean_return=float(scores.mean()),
        std_return=float(scores.std(ddof=1)) if len(scores)>1 else 0.,
        median_return=float(np.median(scores)), min_return=float(scores.min()),
        max_return=float(scores.max()), completion_rate=float(np.mean([r["completed"] for r in rows])),
        mean_coverage=float(np.mean([r["coverage"] for r in rows])),
        score_ge_900_rate=float(np.mean(scores >= 900)))
