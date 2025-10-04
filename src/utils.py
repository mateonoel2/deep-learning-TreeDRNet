# helpers generales: device, semillas, carpetas, CSVs

from __future__ import annotations
from pathlib import Path
import platform
import random
import numpy as np
import pandas as pd
import torch

# retorno cuda si está disponible; si no, uso cpu
def get_device() -> torch.device:

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# imprimo info del entorno y la GPU (útil para logs)
def print_env(device: torch.device) -> None:

    print(f"SO: {platform.system()} | PyTorch: {torch.__version__} | Dispositivo: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

# fijo semillas para reproducir resultados y activo cudnn.benchmark
def set_seed(seed: int = 42) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True

# creo las carpetas que necesito si no existen
def ensure_dirs(*paths: Path) -> None:

    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

# agrego filas a un CSV (si ya existe, concateno)
def save_append_csv(path: Path, rows: list[dict]) -> None:

    df = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)

# guardo el historial de métricas por época a CSV para graficar luego
def save_epoch_history(path: Path, hist: dict | None) -> None:

    if hist is None:
        return
    df = pd.DataFrame(hist)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
