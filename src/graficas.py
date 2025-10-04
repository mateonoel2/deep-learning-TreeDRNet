import numpy as np
import torch
import matplotlib.pyplot as plt

def _to_np_1d(x):
    # garantizo array 1D (H,) para plot
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.ravel(np.asarray(x, dtype=float))

# métricas por época (comparando 2 modelos)
def grafica_mse(dataset, H, hA, hB, nombreA, nombreB, ruta):
    plt.figure(figsize=(7,3))
    ep = hA["epoch"]
    plt.plot(ep, hA["val_mse"], label=nombreA)
    plt.plot(ep, hB["val_mse"], label=nombreB)
    plt.title(f"MSE Val | {dataset} | H={H}")
    plt.xlabel("Época"); plt.ylabel("MSE")
    plt.legend(); plt.tight_layout(); plt.savefig(ruta); plt.close()

def grafica_mae(dataset, H, hA, hB, nombreA, nombreB, ruta):
    plt.figure(figsize=(7,3))
    ep = hA["epoch"]
    plt.plot(ep, hA["val_mae"], label=nombreA)
    plt.plot(ep, hB["val_mae"], label=nombreB)
    plt.title(f"MAE Val | {dataset} | H={H}")
    plt.xlabel("Época"); plt.ylabel("MAE")
    plt.legend(); plt.tight_layout(); plt.savefig(ruta); plt.close()

def grafica_rmse(dataset, H, hA, hB, nombreA, nombreB, ruta):
    plt.figure(figsize=(7,3))
    ep = hA["epoch"]
    plt.plot(ep, hA["val_rmse"], label=nombreA)
    plt.plot(ep, hB["val_rmse"], label=nombreB)
    plt.title(f"RMSE Val | {dataset} | H={H}")
    plt.xlabel("Época"); plt.ylabel("RMSE")
    plt.legend(); plt.tight_layout(); plt.savefig(ruta); plt.close()

def grafica_mape(dataset, H, hA, hB, nombreA, nombreB, ruta):
    plt.figure(figsize=(7,3))
    ep = hA["epoch"]
    plt.plot(ep, hA["val_mape"], label=nombreA)
    plt.plot(ep, hB["val_mape"], label=nombreB)
    plt.title(f"MAPE Val | {dataset} | H={H}")
    plt.xlabel("Época"); plt.ylabel("MAPE")
    plt.legend(); plt.tight_layout(); plt.savefig(ruta); plt.close()

def grafica_r2(dataset, H, hA, hB, nombreA, nombreB, ruta):
    plt.figure(figsize=(7,3))
    ep = hA["epoch"]
    plt.plot(ep, hA["val_r2"], label=nombreA)
    plt.plot(ep, hB["val_r2"], label=nombreB)
    plt.title(f"R² Val | {dataset} | H={H}")
    plt.xlabel("Época"); plt.ylabel("R²")
    plt.legend(); plt.tight_layout(); plt.savefig(ruta); plt.close()

def grafica_lr(dataset, H, hA, hB, nombreA, nombreB, ruta):
    plt.figure(figsize=(7,3))
    ep = hA["epoch"]
    plt.plot(ep, hA["lr"], label=nombreA)
    plt.plot(ep, hB["lr"], label=nombreB)
    plt.title(f"LR | {dataset} | H={H}")
    plt.xlabel("Época"); plt.ylabel("LR")
    plt.legend(); plt.tight_layout(); plt.savefig(ruta); plt.close()

# velocidad separada (iters/s)
def grafica_velocidad_iters(dataset, H, historiales, ruta):
    plt.figure(figsize=(7,3))
    for nombre, h in historiales.items():
        plt.plot(h["epoch"], h["its_por_s"], label=nombre)
    plt.title(f"Velocidad (iteraciones/s) | {dataset} | H={H}")
    plt.xlabel("Época"); plt.ylabel("its/s")
    plt.legend(); plt.tight_layout(); plt.savefig(ruta); plt.close()

# velocidad separada (muestras/s)
def grafica_velocidad_muestras(dataset, H, historiales, ruta):
    plt.figure(figsize=(7,3))
    for nombre, h in historiales.items():
        plt.plot(h["epoch"], h["muestras_por_s"], label=nombre)
    plt.title(f"Velocidad (muestras/s) | {dataset} | H={H}")
    plt.xlabel("Época"); plt.ylabel("muestras/s")
    plt.legend(); plt.tight_layout(); plt.savefig(ruta); plt.close()

# métricas consolidadas por H (crea 5 imágenes)
def grafica_metricas_horizontes(ruta_csv, dataset, ruta_salida):
    import pandas as pd
    df = pd.read_csv(ruta_csv)
    for met in ["test_mse","test_mae","test_rmse","test_mape","test_r2"]:
        piv = df.pivot(index="H", columns="modelo", values=met).sort_index()
        plt.figure(figsize=(9,5))
        piv.plot(marker="o")
        plt.title(f"{dataset} | {met}")
        plt.xlabel("Horizonte H"); plt.ylabel(met)
        plt.tight_layout()
        plt.savefig(str(ruta_salida).replace(".png", f"_{met}.png"))
        plt.close()

# serie (historial L + futuro H + predicciones de todos los modelos)
def grafica_serie_test_multi(
    dataset, L, H, hist_real, futuro_real, preds, ruta_salida,
    offset_ratio=1.0, offset_abs=1.0, offset_min=0.0
):
    plt.figure(figsize=(10,4))
    x_hist = np.arange(L)
    x_fut  = np.arange(L, L + H)

    y_hist = _to_np_1d(hist_real)
    y_fut  = _to_np_1d(futuro_real)

    plt.plot(x_hist, y_hist, label="Historial real", linewidth=1.6)
    plt.plot(x_fut,  y_fut,  label="Futuro real",   linewidth=1.2, linestyle="--")
    plt.axvline(x=L, color="gray", linestyle="--", alpha=0.7, label="Inicio horizonte")

    # offset robusto: se nota aunque el rango sea chico
    y_range = float(np.max(y_fut) - np.min(y_fut))
    y_std   = float(np.std(y_fut))
    q25, q75 = np.percentile(y_fut, [25, 75])
    iqr = float(q75 - q25)
    max_abs = float(np.max(np.abs(y_fut)))
    escala = max(y_range, 3.0*y_std, 1.5*iqr, 0.2*max_abs, 1e-6)

    base_offset = float(offset_abs) if offset_abs is not None else float(escala) * float(offset_ratio)
    base_offset = base_offset + float(offset_min)

    for i, nombre in enumerate(sorted(preds.keys())):
        y_pred = _to_np_1d(preds[nombre])
        offset = (i + 1) * base_offset  # todas separadas
        plt.plot(x_fut, y_pred + offset, linewidth=1.2, label=f"Pred {nombre} (+{offset:.3g})")

    plt.title(f"{dataset} | L={L} | H={H} | Serie Test (preds desplazadas)")
    plt.xlabel("Tiempo"); plt.ylabel("Valor (desplazado en Y)")
    plt.legend(); plt.tight_layout(); plt.savefig(ruta_salida); plt.close()
