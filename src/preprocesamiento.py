# cargo ETT, preparo splits y ventanas L→H; dejo DataLoaders listos para entrenar

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

        ini_max = len(df) - (long_ventana + horizonte)                                                # último índice que permite L de entrada y H de salida
        if ini_max < 0:
            raise ValueError("Serie demasiado corta: no alcanza para long_ventana + horizonte.")
        self._len = ini_max + 1

    def __len__(self) -> int:

        return self._len

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:

        L, H = self.long_ventana, self.horizonte

        x_np = self.X[i: i + L]

        y_np = self.y[i + L: i + L + H]

        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)

        return x, y

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
    df = df.sort_values(by=col_date).reset_index(drop=True)  # aseguro orden temporal correcto antes de ventanear
    return df

# leo el CSV del dataset ETT, ordeno por fecha si hay y armo splits; retorno datasets y escaladores
def cargar_ett(
    dataset: str,
    long_ventana: int,
    horizonte: int,
    col_objetivo: str = "OT"
) -> Tuple[DatasetVentanasETT, DatasetVentanasETT, DatasetVentanasETT, Dict]:

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
    df_val = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    df_test = df.iloc[n_train + n_val:].reset_index(drop=True)

    esc_x = StandardScaler().fit(df_train[cols_features].values)                                                    # normalizo X solo con train para no filtrar info futura
    esc_y = StandardScaler().fit(df_train[[col_objetivo]].values)                                                  # idem para y (después puedo invertir escala)

    ds_train = DatasetVentanasETT(
        df=df_train,
        long_ventana=long_ventana,
        horizonte=horizonte,
        col_objetivo=col_objetivo,
        cols_features=cols_features,
        esc_x=esc_x,
        esc_y=esc_y
    )
    ds_val = DatasetVentanasETT(
        df=df_val,
        long_ventana=long_ventana,
        horizonte=horizonte,
        col_objetivo=col_objetivo,
        cols_features=cols_features,
        esc_x=esc_x,
        esc_y=esc_y
    )
    ds_test = DatasetVentanasETT(
        df=df_test,
        long_ventana=long_ventana,
        horizonte=horizonte,
        col_objetivo=col_objetivo,
        cols_features=cols_features,
        esc_x=esc_x,
        esc_y=esc_y
    )

    info = {
        "cols_features": cols_features,
        "col_objetivo": col_objetivo,
        "esc_x": esc_x,
        "esc_y": esc_y
    }

    return ds_train, ds_val, ds_test, info

# creo DataLoaders con opciones de rendimiento (shuffle, workers, prefetch)
def crear_loaders(
    ds_train: DatasetVentanasETT,
    ds_val: DatasetVentanasETT,
    batch: int,
    workers: int,
    persistent_workers: bool = False,
    prefetch_factor: Optional[int] = None,
    drop_last_train: bool = False
) -> Tuple[DataLoader, DataLoader]:

    usar_persist = persistent_workers and (workers > 0)

    dl_train = DataLoader(
        ds_train,
        batch_size=batch,
        shuffle=True,                                                                  # en train barajo para romper patrones fijos
        num_workers=workers,
        pin_memory=True,                                                                                 # acelera el pase a GPU desde los DataLoader
        persistent_workers=usar_persist,                                                                   # mantengo workers vivos para no re-crear procesos por época
        prefetch_factor=(prefetch_factor if (workers > 0 and prefetch_factor) else None),                             # preparo lotes por adelantado (útil con varios workers)
        drop_last=drop_last_train                                                                            # a veces conviene para batchs constantes en BN (acá no es crítico)
    )

    dl_val = DataLoader(
        ds_val,
        batch_size=batch,
        shuffle=False,                                                              # en val mantengo orden y reproducibilidad
        num_workers=workers,
        pin_memory=True,                                                                 # acelera el pase a GPU desde los DataLoader
        persistent_workers=usar_persist,                                                                    # mantengo workers vivos para no re-crear procesos por época
        prefetch_factor=(prefetch_factor if (workers > 0 and prefetch_factor) else None),                            # preparo lotes por adelantado (útil con varios workers)
        drop_last=False
    )

    return dl_train, dl_val
