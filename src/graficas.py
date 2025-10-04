from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt


def _ensure_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def _plot_multi_val_epocas(dataset, horizonte, historiales, clave_metric, etiqueta_y, ruta_salida):
    plt.figure(figsize=(10, 6))
    for i, (nombre, hist) in enumerate(historiales.items()):
        if hist is None or clave_metric not in hist:
            continue
        ep = hist["epoch"]
        y = hist[clave_metric]
        estilo = "-" if i % 2 == 0 else "-."
        plt.plot(ep, y, estilo, label=nombre)
    plt.title(f"{dataset} | H={horizonte} | {etiqueta_y} (validación) por época")
    plt.xlabel("Época")
    plt.ylabel(etiqueta_y)
    plt.grid(True, alpha=0.3)
    plt.legend()
    _ensure_dir(ruta_salida)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()


def grafica_val_mse_multi(dataset, H, historiales, ruta):
    _plot_multi_val_epocas(dataset, H, historiales, "val_mse", "MSE", ruta)


def grafica_val_mae_multi(dataset, H, historiales, ruta):
    _plot_multi_val_epocas(dataset, H, historiales, "val_mae", "MAE", ruta)


def grafica_val_rmse_multi(dataset, H, historiales, ruta):
    _plot_multi_val_epocas(dataset, H, historiales, "val_rmse", "RMSE", ruta)


def grafica_val_mape_multi(dataset, H, historiales, ruta):
    _plot_multi_val_epocas(dataset, H, historiales, "val_mape", "MAPE (%)", ruta)


def grafica_val_r2_multi(dataset, H, historiales, ruta):
    _plot_multi_val_epocas(dataset, H, historiales, "val_r2", "R²", ruta)


def grafica_lr_multi(dataset, horizonte, historiales, ruta_salida):
    plt.figure(figsize=(10, 6))
    for i, (nombre, hist) in enumerate(historiales.items()):
        if hist is None or "lr" not in hist:
            continue
        ep = hist["epoch"]
        lr = hist["lr"]
        estilo = "-" if i % 2 == 0 else "-."
        plt.plot(ep, lr, estilo, label=nombre)
    plt.title(f"{dataset} | H={horizonte} | Learning Rate por época")
    plt.xlabel("Época")
    plt.ylabel("LR")
    plt.grid(True, alpha=0.3)
    plt.legend()
    _ensure_dir(ruta_salida)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()


def grafica_velocidad_multi(dataset, horizonte, historiales, ruta_salida):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    for i, (nombre, hist) in enumerate(historiales.items()):
        if hist is None or "its_por_s" not in hist:
            continue
        ep = hist["epoch"]
        y = hist["its_por_s"]
        estilo = "-" if i % 2 == 0 else "-."
        plt.plot(ep, y, estilo, label=nombre)
    plt.title("Iteraciones por segundo")
    plt.xlabel("Época")
    plt.ylabel("it/s")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.subplot(1, 2, 2)
    for i, (nombre, hist) in enumerate(historiales.items()):
        if hist is None or "muestras_por_s" not in hist:
            continue
        ep = hist["epoch"]
        y = hist["muestras_por_s"]
        estilo = "-" if i % 2 == 0 else "-."
        plt.plot(ep, y, estilo, label=nombre)
    plt.title("Muestras por segundo")
    plt.xlabel("Época")
    plt.ylabel("muestras/s")
    plt.grid(True, alpha=0.3)
    plt.legend()
    _ensure_dir(ruta_salida)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()


def grafica_metricas_horizontes(ruta_csv: Path, dataset: str, ruta_salida: Path):
    import pandas as pd

    df = pd.read_csv(ruta_csv)
    df = df[df["dataset"] == dataset].copy()

    def _plot_metric(columna, etiqueta_y, ruta_png):
        plt.figure(figsize=(10, 6))
        for nombre_modelo, dfm in df.groupby("modelos"):
            if columna not in dfm.columns:
                continue
            plt.plot(dfm["H"], dfm[columna], marker="o", label=nombre_modelo)
        plt.title(f"{dataset} | {etiqueta_y} de TEST por horizonte")
        plt.xlabel("Horizonte (H)")
        plt.ylabel(f"{etiqueta_y} (TEST)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        _ensure_dir(ruta_png)
        plt.tight_layout()
        plt.savefig(ruta_png, dpi=150)
        plt.close()

    base = Path(ruta_salida)
    stem = base.with_suffix("")
    _plot_metric("test_mse", "MSE", Path(f"{stem}_mse.png"))
    _plot_metric("test_mae", "MAE", Path(f"{stem}_mae.png"))
    _plot_metric("test_rmse", "RMSE", Path(f"{stem}_rmse.png"))
    _plot_metric("test_mape", "MAPE (%)", Path(f"{stem}_mape.png"))
    _plot_metric("test_r2", "R²", Path(f"{stem}_r2.png"))


def grafica_serie_test_multi(
    dataset, L, H, hist_real, futuro_real, preds, ruta_salida, titulo_extra: str = ""
):
    import numpy as np

    hist = hist_real.squeeze(-1)
    fut = futuro_real.squeeze(-1)
    x_hist = np.arange(L)
    x_fut = np.arange(L, L + H)
    x_zoom = np.arange(H)
    fig = plt.figure(figsize=(12, 9))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(x_hist, hist, label="Real (historial)", linewidth=2.0)
    ax1.plot(x_fut, fut, label="Real (futuro)", linewidth=2.0, linestyle=":")
    for i, (nombre, arr) in enumerate(preds.items()):
        y = arr.squeeze(-1)
        estilo = "--" if i % 2 == 0 else "-."
        ax1.plot(x_fut, y, label=f"Pred {nombre}", linestyle=estilo)
    ax1.axvline(x=L, color="gray", linestyle="--", alpha=0.6)
    titulo = f"{dataset} | L={L} H={H} | Serie test (última ventana)"
    if titulo_extra:
        titulo += f" | {titulo_extra}"
    ax1.set_title(titulo)
    ax1.set_xlabel("Paso temporal")
    ax1.set_ylabel("Valor (normalizado)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(x_zoom, fut, label="Real (futuro)", linewidth=2.0, linestyle=":")
    for i, (nombre, arr) in enumerate(preds.items()):
        y = arr.squeeze(-1)
        estilo = "--" if i % 2 == 0 else "-."
        ax2.plot(x_zoom, y, label=f"Pred {nombre}", linestyle=estilo)
    ax2.set_title("Zoom en el horizonte (H pasos)")
    ax2.set_xlabel("Paso futuro (0..H-1)")
    ax2.set_ylabel("Valor (normalizado)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    _ensure_dir(ruta_salida)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()
