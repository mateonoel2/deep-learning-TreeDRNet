from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

RAIZ = Path("datos")
CARPETA_DATOS = RAIZ / "ETT-small"


class DatasetVentanasETT(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        long_ventana: int,
        horizonte: int,
        col_objetivo: str,
        cols_features: list[str],
        esc_x: Optional[StandardScaler],
        esc_y: Optional[StandardScaler],
    ):
        self.long_ventana = long_ventana
        self.horizonte = horizonte
        self.col_objetivo = col_objetivo
        self.cols_features = cols_features
        self.esc_x = esc_x
        self.esc_y = esc_y
        X = df[cols_features].values.astype(np.float32)
        y = df[[col_objetivo]].values.astype(np.float32)
        if esc_x is not None:
            X = esc_x.transform(X)
        if esc_y is not None:
            y = esc_y.transform(y)
        self.X = X
        self.y = y
        ini_max = len(df) - (long_ventana + horizonte)
        if ini_max < 0:
            raise ValueError("Serie demasiado corta: no alcanza para long_ventana + horizonte.")
        self._len = ini_max + 1

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, i: int):
        L, H = self.long_ventana, self.horizonte
        x_np = self.X[i : i + L]
        y_np = self.y[i + L : i + L + H]
        return torch.from_numpy(x_np), torch.from_numpy(y_np)


def _ordenar_por_fecha_si_existe(df: pd.DataFrame) -> pd.DataFrame:
    col_date = None
    for c in df.columns:
        if c.lower() == "date":
            col_date = c
            break
    if col_date is None:
        return df.reset_index(drop=True)
    df = df.copy()
    df[col_date] = pd.to_datetime(df[col_date], errors="coerce")
    return df.sort_values(by=col_date).reset_index(drop=True)


def cargar_ett(dataset: str, long_ventana: int, horizonte: int, col_objetivo: str = "OT"):
    ruta_csv = CARPETA_DATOS / f"{dataset}.csv"
    df = pd.read_csv(ruta_csv)
    df = _ordenar_por_fecha_si_existe(df)
    cols = [c for c in df.columns if c.lower() != "date"]
    assert col_objetivo in cols, f"La columna objetivo '{col_objetivo}' no está en: {cols}"
    cols_features = list(cols)
    n = len(df)
    n_train = int(n * 0.70)
    n_val = int(n * 0.10)
    df_train = df.iloc[:n_train].reset_index(drop=True)
    df_val = df.iloc[n_train : n_train + n_val].reset_index(drop=True)
    df_test = df.iloc[n_train + n_val :].reset_index(drop=True)
    esc_x = StandardScaler().fit(df_train[cols_features].values)
    esc_y = StandardScaler().fit(df_train[[col_objetivo]].values)
    ds_train = DatasetVentanasETT(
        df_train, long_ventana, horizonte, col_objetivo, cols_features, esc_x, esc_y
    )
    ds_val = DatasetVentanasETT(df_val, long_ventana, horizonte, col_objetivo, cols_features, esc_x, esc_y)
    ds_test = DatasetVentanasETT(df_test, long_ventana, horizonte, col_objetivo, cols_features, esc_x, esc_y)
    info = {"cols_features": cols_features, "col_objetivo": col_objetivo, "esc_x": esc_x, "esc_y": esc_y}
    return ds_train, ds_val, ds_test, info


def crear_loaders(
    ds_train,
    ds_val,
    batch: int,
    workers: int,
    persistent_workers: bool = False,
    prefetch_factor: Optional[int] = None,
    drop_last_train: bool = False,
):
    usar_persist = persistent_workers and (workers > 0)
    dl_train = DataLoader(
        ds_train,
        batch_size=batch,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=usar_persist,
        prefetch_factor=(prefetch_factor if (workers > 0 and prefetch_factor) else None),
        drop_last=drop_last_train,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=usar_persist,
        prefetch_factor=(prefetch_factor if (workers > 0 and prefetch_factor) else None),
        drop_last=False,
    )
    return dl_train, dl_val
