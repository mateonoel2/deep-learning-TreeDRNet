from __future__ import annotations
from pathlib import Path
import time, math
from typing import Dict, Tuple, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# --------- Métricas base ---------
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

# --------- Normalización robusta de salida ---------
def _coerce_pred(pred) -> torch.Tensor:
    if pred is None:
        raise ValueError("forward() devolvió None; debe devolver un Tensor.")
    if isinstance(pred, torch.Tensor):
        return pred
    if isinstance(pred, dict):
        for k in ("pred", "y_pred", "forecast", "out", "logits"):
            v = pred.get(k, None)
            if isinstance(v, torch.Tensor):
                return v
        raise TypeError("Salida dict sin Tensor en claves conocidas.")
    if isinstance(pred, (list, tuple)):
        for v in pred:
            if isinstance(v, torch.Tensor):
                return v
        raise TypeError("Salida tuple/list sin Tensor utilizable.")
    raise TypeError(f"Tipo de salida no soportado: {type(pred)}")

def _match_shape(pred: torch.Tensor, yb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if pred.shape == yb.shape:
        return pred, yb
    if pred.ndim == yb.ndim - 1 and yb.size(-1) == 1 and pred.shape == yb.squeeze(-1).shape:
        return pred.unsqueeze(-1), yb
    if pred.ndim == yb.ndim + 1 and pred.size(-1) == 1 and pred.squeeze(-1).shape == yb.shape:
        return pred, yb.unsqueeze(-1)
    if pred.ndim == yb.ndim and pred.size(0) == yb.size(0):
        if pred.size(-1) != yb.size(-1):
            if pred.size(-1) == 1:
                return pred, yb[..., :1]
            if yb.size(-1) == 1:
                return pred[..., :1], yb
    return pred, yb

# --------- EarlyStopping ---------
class EarlyStopping:
    def __init__(self, paciencia: int = 6, min_delta: float = 1e-3):
        self.paciencia = paciencia
        self.min_delta = min_delta
        self._mejor = None
        self._conteo = 0
    def step(self, valor: float) -> bool:
        if self._mejor is None:
            self._mejor = valor; self._conteo = 0
            return False
        if (self._mejor - valor) > self.min_delta:
            self._mejor = valor; self._conteo = 0
            return False
        self._conteo += 1
        return self._conteo >= self.paciencia

# --------- Entrenar + validar ---------
def entrenar_validar(
    modelo: nn.Module,
    dl_train: DataLoader,
    dl_val: DataLoader,
    epocas: int,
    lr: float,
    device: torch.device,
    usar_amp: bool,
    ruta_checkpoint: Path,
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
    optim = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=1e-2)
    autocast_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler(device=device.type, enabled=usar_amp and amp_dtype == "fp16")
    sched = None
    if usar_scheduler:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode="min", factor=factor_sched, patience=paciencia_sched, min_lr=min_lr
        )
    es = EarlyStopping(paciencia=paciencia_es, min_delta=min_delta_es) if usar_earlystop else None

    hist = {
        "epoch": [], "lr": [],
        "train_mse": [], "train_mae": [], "train_rmse": [], "train_mape": [], "train_r2": [],
        "val_mse": [], "val_mae": [], "val_rmse": [], "val_mape": [], "val_r2": [],
        "tiempo_epoca_s": [], "its_por_s": [], "muestras_por_s": []
    }

    mejor_val_mse = float("inf")
    mejor_epoch = -1
    t0_total = time.perf_counter()

    for ep in range(1, epocas + 1):
        modelo.train()
        t0 = time.perf_counter()
        suma_mse_tr = suma_mae_tr = suma_mape_tr = suma_r2_tr = 0.0
        n_muestras_tr = n_iters_tr = 0

        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=usar_amp, dtype=autocast_dtype):
                raw = modelo(xb)
                pred = _coerce_pred(raw)
                pred, yb_n = _match_shape(pred, yb)
                mse = _mse(pred, yb_n)
                mae = _mae(pred, yb_n)
                mape = _mape(pred, yb_n)
                r2 = _r2(pred, yb_n)

            if scaler.is_enabled():
                scaler.scale(mse).backward()
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
            else:
                mse.backward()
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
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
                    raw = modelo(xb)
                    pred = _coerce_pred(raw)
                    pred, yb_n = _match_shape(pred, yb)
                    mse = _mse(pred, yb_n)
                    mae = _mae(pred, yb_n)
                    mape = _mape(pred, yb_n)
                    r2 = _r2(pred, yb_n)

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

        if prom_mse_va < mejor_val_mse:
            mejor_val_mse = prom_mse_va
            mejor_epoch = ep
            ruta_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"modelo": modelo.state_dict(), "epoch": mejor_epoch, "val_mse": mejor_val_mse}, ruta_checkpoint)

        parar = False
        if es is not None and es.step(prom_mse_va):
            print(f"EarlyStopping: sin mejora en {es.paciencia} épocas. Paramos en {ep}.")
            parar = True

        hist["epoch"].append(ep); hist["lr"].append(lr_actual)
        hist["train_mse"].append(prom_mse_tr); hist["train_mae"].append(prom_mae_tr)
        hist["train_rmse"].append(prom_rmse_tr); hist["train_mape"].append(prom_mape_tr); hist["train_r2"].append(prom_r2_tr)
        hist["val_mse"].append(prom_mse_va); hist["val_mae"].append(prom_mae_va)
        hist["val_rmse"].append(prom_rmse_va); hist["val_mape"].append(prom_mape_va); hist["val_r2"].append(prom_r2_va)
        hist["tiempo_epoca_s"].append(dur_epoca); hist["its_por_s"].append(its_por_s); hist["muestras_por_s"].append(muestras_por_s)

        print(
            f"[Época {ep:02d}] "
            f"train MSE={prom_mse_tr:.6f} MAE={prom_mae_tr:.6f} RMSE={prom_rmse_tr:.6f} MAPE={prom_mape_tr:.3f}% R2={prom_r2_tr:.4f} | "
            f"val MSE={prom_mse_va:.6f} MAE={prom_mae_va:.6f} RMSE={prom_rmse_va:.6f} MAPE={prom_mape_va:.3f}% R2={prom_r2_va:.4f} | "
            f"LR={lr_actual:.2e} | {its_por_s:.2f} it/s, {muestras_por_s:.1f} muestras/s, {dur_epoca:.2f}s"
        )
        if parar:
            break

    print(f"Tiempo total de entrenamiento: {time.perf_counter() - t0_total:.2f} s")
    return hist

# --------- Evaluación de TEST (MAPE desnormalizado opcional) ---------
def evaluar_test(
    modelo: nn.Module,
    ds_test: Dataset,
    device: torch.device,
    batch: int = 512,
    esc_y = None,
    mape_original: bool = True
) -> Tuple[float, float, float, float, float]:
    dl_test = DataLoader(ds_test, batch_size=batch, shuffle=False)
    modelo = modelo.to(device); modelo.eval()
    suma_mse = suma_mae = suma_mape = suma_r2 = 0.0
    n_muestras = 0

    with torch.no_grad():
        for xb, yb in dl_test:
            xb, yb = xb.to(device), yb.to(device)
            pred = _coerce_pred(modelo(xb))
            pred, yb_n = _match_shape(pred, yb)

            if mape_original and esc_y is not None:
                import numpy as np
                y_np = yb_n.detach().cpu().numpy().reshape(-1, 1)
                p_np = pred.detach().cpu().numpy().reshape(-1, 1)
                y_orig = torch.from_numpy(esc_y.inverse_transform(y_np).astype(np.float32)).to(device)
                p_orig = torch.from_numpy(esc_y.inverse_transform(p_np).astype(np.float32)).to(device)
                mape_val = _mape(p_orig, y_orig).item()
            else:
                mape_val = _mape(pred, yb_n).item()

            mse = _mse(pred, yb_n).item()
            mae = _mae(pred, yb_n).item()
            r2  = _r2 (pred, yb_n).item()

            bs = xb.size(0)
            suma_mse  += mse  * bs
            suma_mae  += mae  * bs
            suma_mape += mape_val * bs
            suma_r2   += r2   * bs
            n_muestras += bs

    prom_mse  = suma_mse  / n_muestras
    prom_mae  = suma_mae  / n_muestras
    prom_mape = suma_mape / n_muestras
    prom_r2   = suma_r2   / n_muestras
    prom_rmse = _rmse_from_mse(prom_mse)
    return float(prom_mse), float(prom_mae), float(prom_rmse), float(prom_mape), float(prom_r2)

# --------- Serie para gráficas ---------
def obtener_hist_y_preds_ultima_ventana(
    modelos: List[Tuple[str, nn.Module]],
    ds_test: Dataset,
    device: torch.device,
    idx_col_objetivo: int,
    desnormalizar: bool = False,
    esc_y = None
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    i = len(ds_test) - 1
    xb, yb = ds_test[i]
    if yb.ndim == 1:
        yb = yb.unsqueeze(-1)
    hist = xb[:, idx_col_objetivo].unsqueeze(-1)
    fut  = yb.clone()
    xb_batch = xb.unsqueeze(0).to(device)
    preds: Dict[str, torch.Tensor] = {}
    for nombre, modelo in modelos:
        modelo = modelo.to(device); modelo.eval()
        with torch.no_grad():
            p = _coerce_pred(modelo(xb_batch)).squeeze(0)
        preds[nombre] = p.cpu()
    if desnormalizar and esc_y is not None:
        import numpy as np
        def inv(v: torch.Tensor) -> torch.Tensor:
            arr = v.cpu().numpy()
            arr2 = esc_y.inverse_transform(arr.reshape(-1, 1))
            return torch.from_numpy(arr2.astype(np.float32))
        hist = inv(hist); fut = inv(fut)
        for k in list(preds.keys()):
            preds[k] = inv(preds[k])
    return hist.cpu(), fut.cpu(), preds
