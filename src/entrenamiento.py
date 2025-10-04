from __future__ import annotations
import time, math
from typing import Dict, Tuple, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

def _mse(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return nn.MSELoss()(pred, y)

def _mae(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return nn.L1Loss()(pred, y)

def _rmse_from_mse(mse_value: float) -> float:
    return math.sqrt(max(mse_value, 0.0))

def _mape(pred: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    denom = torch.clamp(torch.abs(y), min=eps)
    return torch.mean(torch.abs((y - pred) / denom)) * 100.0

def _r2(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y_mean = torch.mean(y)
    ss_res = torch.sum((y - pred) ** 2)
    ss_tot = torch.sum((y - y_mean) ** 2) + 1e-12
    return 1.0 - (ss_res / ss_tot)

class EarlyStopping:
    def __init__(self, paciencia: int = 6, min_delta: float = 1e-3):
        self.paciencia = paciencia
        self.min_delta = min_delta
        self._mejor = None
        self._conteo = 0
    def step(self, valor: float) -> bool:
        if self._mejor is None:
            self._mejor = valor
            self._conteo = 0
            return False
        if (self._mejor - valor) > self.min_delta:
            self._mejor = valor
            self._conteo = 0
            return False
        self._conteo += 1
        return self._conteo >= self.paciencia

def entrenar_validar(
    modelo: nn.Module,
    dl_train: DataLoader,
    dl_val: DataLoader,
    epocas: int,
    lr: float,
    device: torch.device,
    usar_amp: bool,
    amp_dtype: str = "bf16",
    usar_scheduler: bool = True,
    factor_sched: float = 0.5,
    paciencia_sched: int = 2,
    min_lr: float = 1e-6,
    usar_earlystop: bool = True,
    paciencia_es: int = 6,
    min_delta_es: float = 1e-3
) -> Dict[str, list]:
    modelo = modelo.to(device)
    optim = torch.optim.AdamW(modelo.parameters(), lr=lr)
    autocast_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler(device=device.type, enabled=usar_amp and autocast_dtype == torch.float16)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=factor_sched, patience=paciencia_sched, min_lr=min_lr
    ) if usar_scheduler else None
    es = EarlyStopping(paciencia=paciencia_es, min_delta=min_delta_es) if usar_earlystop else None

    hist = {
        "epoch": [], "lr": [],
        "train_mse": [], "train_mae": [], "train_rmse": [], "train_mape": [], "train_r2": [],
        "val_mse": [], "val_mae": [], "val_rmse": [], "val_mape": [], "val_r2": [],
        "tiempo_epoca_s": [], "its_por_s": [], "muestras_por_s": []
    }

    t0_total = time.perf_counter()
    mejor_val = float("inf")
    sin_mejora = 0

    for ep in range(1, epocas + 1):
        modelo.train()
        t0 = time.perf_counter()
        suma_mse_tr = suma_mae_tr = suma_mape_tr = suma_r2_tr = 0.0
        n_muestras_tr = n_iters_tr = 0

        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=usar_amp, dtype=autocast_dtype):
                pred = modelo(xb)
                mse = _mse(pred, yb)
                mae = _mae(pred, yb)
                mape = _mape(pred, yb)
                r2 = _r2(pred, yb)
            if scaler.is_enabled():
                scaler.scale(mse).backward()
                scaler.step(optim)
                scaler.update()
            else:
                mse.backward()
                optim.step()
            bs = xb.size(0)
            suma_mse_tr += mse.item() * bs
            suma_mae_tr += mae.item() * bs
            suma_mape_tr += mape.item() * bs
            suma_r2_tr += r2.item() * bs
            n_muestras_tr += bs
            n_iters_tr += 1

        prom_mse_tr = suma_mse_tr / n_muestras_tr
        prom_mae_tr = suma_mae_tr / n_muestras_tr
        prom_mape_tr = suma_mape_tr / n_muestras_tr
        prom_r2_tr = suma_r2_tr / n_muestras_tr
        prom_rmse_tr = _rmse_from_mse(prom_mse_tr)

        modelo.eval()
        suma_mse_va = suma_mae_va = suma_mape_va = suma_r2_va = 0.0
        n_muestras_va = 0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                with torch.amp.autocast(device_type=device.type, enabled=usar_amp, dtype=autocast_dtype):
                    pred = modelo(xb)
                    mse = _mse(pred, yb)
                    mae = _mae(pred, yb)
                    mape = _mape(pred, yb)
                    r2 = _r2(pred, yb)
                bs = xb.size(0)
                suma_mse_va += mse.item() * bs
                suma_mae_va += mae.item() * bs
                suma_mape_va += mape.item() * bs
                suma_r2_va += r2.item() * bs
                n_muestras_va += bs

        prom_mse_va = suma_mse_va / n_muestras_va
        prom_mae_va = suma_mae_va / n_muestras_va
        prom_mape_va = suma_mape_va / n_muestras_va
        prom_r2_va = suma_r2_va / n_muestras_va
        prom_rmse_va = _rmse_from_mse(prom_mse_va)

        dur_epoca = time.perf_counter() - t0
        its_por_s = n_iters_tr / dur_epoca if dur_epoca > 0 else 0.0
        muestras_por_s = n_muestras_tr / dur_epoca if dur_epoca > 0 else 0.0
        lr_actual = optim.param_groups[0]["lr"]

        if sched is not None:
            sched.step(prom_mse_va)

        if prom_mse_va < mejor_val - min_delta_es:
            mejor_val = prom_mse_va
            sin_mejora = 0
        else:
            sin_mejora += 1

        if usar_earlystop and es is not None and es.step(prom_mse_va):
            print(f"EarlyStopping: stop en época {ep}")
            break

        hist["epoch"].append(ep); hist["lr"].append(lr_actual)
        hist["train_mse"].append(prom_mse_tr); hist["train_mae"].append(prom_mae_tr)
        hist["train_rmse"].append(prom_rmse_tr); hist["train_mape"].append(prom_mape_tr); hist["train_r2"].append(prom_r2_tr)
        hist["val_mse"].append(prom_mse_va); hist["val_mae"].append(prom_mae_va)
        hist["val_rmse"].append(prom_rmse_va); hist["val_mape"].append(prom_mape_va); hist["val_r2"].append(prom_r2_va)
        hist["tiempo_epoca_s"].append(dur_epoca); hist["its_por_s"].append(its_por_s); hist["muestras_por_s"].append(muestras_por_s)

        print(f"[Ép {ep:02d}] train MSE={prom_mse_tr:.6f} val MSE={prom_mse_va:.6f} MAE={prom_mae_va:.6f} | {its_por_s:.1f} it/s {muestras_por_s:.1f} muestras/s {dur_epoca:.2f}s")

    print(f"Tiempo total: {time.perf_counter() - t0_total:.2f} s")
    return hist

def evaluar_test(modelo: nn.Module, ds_test: Dataset, device: torch.device, batch: int = 512) -> Tuple[float, float, float, float, float]:
    dl_test = DataLoader(ds_test, batch_size=batch, shuffle=False)
    modelo = modelo.to(device); modelo.eval()
    suma_mse = suma_mae = suma_mape = suma_r2 = 0.0
    n_muestras = 0
    with torch.no_grad():
        for xb, yb in dl_test:
            xb, yb = xb.to(device), yb.to(device)
            pred = modelo(xb)
            mse = _mse(pred, yb); mae = _mae(pred, yb)
            mape = _mape(pred, yb); r2 = _r2(pred, yb)
            bs = xb.size(0)
            suma_mse += mse.item() * bs
            suma_mae += mae.item() * bs
            suma_mape += mape.item() * bs
            suma_r2 += r2.item() * bs
            n_muestras += bs
    prom_mse = suma_mse / n_muestras
    prom_mae = suma_mae / n_muestras
    prom_mape = suma_mape / n_muestras
    prom_r2 = suma_r2 / n_muestras
    prom_rmse = _rmse_from_mse(prom_mse)
    return float(prom_mse), float(prom_mae), float(prom_rmse), float(prom_mape), float(prom_r2)

def obtener_hist_y_preds_ultima_ventana(modelos: List[Tuple[str, nn.Module]], ds_test: Dataset, device: torch.device, idx_col_objetivo: int, desnormalizar: bool = False, esc_y = None):
    i = len(ds_test) - 1
    xb, yb = ds_test[i]
    if yb.ndim == 1: yb = yb.unsqueeze(-1)
    hist = xb[:, idx_col_objetivo].unsqueeze(-1); fut = yb.clone()
    xb_batch = xb.unsqueeze(0).to(device)
    preds = {}
    for nombre, modelo in modelos:
        modelo = modelo.to(device); modelo.eval()
        with torch.no_grad():
            p = modelo(xb_batch).squeeze(0)
        preds[nombre] = p.cpu()
    return hist.cpu(), fut.cpu(), preds
