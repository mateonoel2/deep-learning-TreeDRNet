from __future__ import annotations
from pathlib import Path
import torch

# ====== PARÁMETROS ======
DATASETS     = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
HORIZONTES   = [24, 48, 96, 192, 336, 720]
LONG_VENTANA = 96

EPOCAS = 60
BATCH  = 32
LR     = 1e-4
SEED   = 42

USAR_AMP  = True
AMP_DTYPE = "bf16"

WORKERS            = 6
PERSISTENT_WORKERS = True
PREFETCH_FACTOR    = 2
DROP_LAST_TRAIN    = False

USAR_SCHEDULER   = True
FACTOR_SCHED     = 0.5
PACIENCIA_SCHED  = 3
MIN_LR           = 1e-6

USAR_EARLYSTOP = True
PACIENCIA_ES   = 8
MIN_DELTA_ES   = 1e-5

TD_OCULTO   = 128
TD_PROF     = 3
TD_RAMAS    = 2

CARPETA_RESULTADOS = Path("resultados")
CARPETA_METRICAS_GLOBAL = CARPETA_RESULTADOS / "metricas"

# ====== IMPORTS ======
from src.utils import get_device, print_env, set_seed, ensure_dirs, save_append_csv, save_epoch_history
from src.preprocesamiento import cargar_ett, crear_loaders
from src.modelos.treedrnet import TreeDRNet
from src.entrenamiento import (
    entrenar_validar, evaluar_test,
    obtener_hist_y_preds_ultima_ventana
)
from src.graficas import (
    grafica_val_mse_multi, grafica_val_mae_multi, grafica_val_rmse_multi, grafica_val_mape_multi, grafica_val_r2_multi,
    grafica_lr_multi, grafica_velocidad_multi,
    grafica_metricas_horizontes, grafica_serie_test_multi
)

# Helper: carga de checkpoint sin warning de weights_only
def _safe_load(ckpt_path: Path, map_location):
    try:
        return torch.load(ckpt_path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(ckpt_path, map_location=map_location)

# ============================ MAIN ============================
if __name__ == "__main__":
    set_seed(SEED)
    disp = get_device()
    print_env(disp)
    ensure_dirs(CARPETA_METRICAS_GLOBAL)

    filas_test_global: list[dict] = []

    for dataset in DATASETS:
        CARPETA_RES_DS = CARPETA_RESULTADOS / dataset
        CARPETA_METRICAS_DS_CONS = CARPETA_RES_DS / "metricas"
        CARPETA_GRAFICAS_DS_CONS = CARPETA_RES_DS / "graficas"
        ensure_dirs(CARPETA_RES_DS, CARPETA_METRICAS_DS_CONS, CARPETA_GRAFICAS_DS_CONS)

        CSV_TEST_DS = CARPETA_METRICAS_DS_CONS / "TEST_resultados.csv"
        filas_test_ds: list[dict] = []

        for H in HORIZONTES:
            print(f"\n=== DATASET: {dataset} | L={LONG_VENTANA} | H={H} ===")

            CARPETA_RES_DSH = CARPETA_RES_DS / f"H{H}"
            CARPETA_PESOS_DSH = CARPETA_RES_DSH / "pesos"
            CARPETA_METRICAS_DSH = CARPETA_RES_DSH / "metricas"
            CARPETA_GRAFICAS_DSH = CARPETA_RES_DSH / "graficas"
            ensure_dirs(CARPETA_RES_DSH, CARPETA_PESOS_DSH, CARPETA_METRICAS_DSH, CARPETA_GRAFICAS_DSH)

            # Datos + Loaders
            ds_tr, ds_va, ds_te, info = cargar_ett(dataset, LONG_VENTANA, H, col_objetivo="OT")
            dl_tr, dl_va = crear_loaders(
                ds_train=ds_tr, ds_val=ds_va, batch=BATCH, workers=WORKERS,
                persistent_workers=PERSISTENT_WORKERS, prefetch_factor=PREFETCH_FACTOR,
                drop_last_train=DROP_LAST_TRAIN
            )
            dim_in = len(info["cols_features"])
            idx_obj = info["cols_features"].index(info["col_objetivo"])

            historiales = {}
            modelos_inst = {}

            # ---- ÚNICO MODELO: TreeDRNet ----
            nombre = "TreeDRNet"
            modelo = TreeDRNet(
                entrada_dim=dim_in, salida_dim=1, horizonte=H,
                long_ventana=LONG_VENTANA,
                profundidad_arbol=TD_PROF, num_ramas=TD_RAMAS,
                hidden_gate=TD_OCULTO, hidden_core=TD_OCULTO,
                mlp_depth=2, dropout=0.10, usar_conv_covariate=True
            )
            ckpt = CARPETA_PESOS_DSH / f"{dataset}_TreeDRNet_L{LONG_VENTANA}_H{H}.pt"
            hist = entrenar_validar(
                modelo=modelo, dl_train=dl_tr, dl_val=dl_va,
                epocas=EPOCAS, lr=LR, device=disp, usar_amp=USAR_AMP,
                ruta_checkpoint=ckpt, amp_dtype=AMP_DTYPE,
                usar_scheduler=USAR_SCHEDULER, factor_sched=FACTOR_SCHED,
                paciencia_sched=PACIENCIA_SCHED, min_lr=MIN_LR,
                usar_earlystop=USAR_EARLYSTOP, paciencia_es=PACIENCIA_ES, min_delta_es=MIN_DELTA_ES
            )
            best = _safe_load(ckpt, map_location=disp); modelo.load_state_dict(best["modelo"])

            # MAPE se reporta desnormalizado (escala original)
            mse, mae, rmse, mape, r2 = evaluar_test(
                modelo, ds_te, disp, esc_y=info["esc_y"], mape_original=True
            )
            save_epoch_history(CARPETA_METRICAS_DSH / f"{dataset}_TreeDRNet_L{LONG_VENTANA}_H{H}_hist.csv", hist)
            filas = {"dataset": dataset, "modelo": "treedrnet", "L": LONG_VENTANA, "H": H,
                     "test_mse": mse, "test_mae": mae, "test_rmse": rmse, "test_mape": mape, "test_r2": r2}
            filas_test_ds.append(filas); filas_test_global.append(filas)
            historiales[nombre] = hist; modelos_inst[nombre] = modelo
            print(f"[TreeDRNet] Test MSE={mse:.6f} MAE={mae:.6f} RMSE={rmse:.6f} MAPE={mape:.3f}% R2={r2:.4f}")

            # Gráficas por época (aunque ahora solo hay 1 modelo, mantenemos la API multi)
            grafica_val_mse_multi (dataset, H, historiales, CARPETA_GRAFICAS_DSH / f"{dataset}_L{LONG_VENTANA}_H{H}_mse.png")
            grafica_val_mae_multi (dataset, H, historiales, CARPETA_GRAFICAS_DSH / f"{dataset}_L{LONG_VENTANA}_H{H}_mae.png")
            grafica_val_rmse_multi(dataset, H, historiales, CARPETA_GRAFICAS_DSH / f"{dataset}_L{LONG_VENTANA}_H{H}_rmse.png")
            grafica_val_mape_multi(dataset, H, historiales, CARPETA_GRAFICAS_DSH / f"{dataset}_L{LONG_VENTANA}_H{H}_mape.png")
            grafica_val_r2_multi  (dataset, H, historiales, CARPETA_GRAFICAS_DSH / f"{dataset}_L{LONG_VENTANA}_H{H}_r2.png")

            grafica_lr_multi(dataset, H, historiales, CARPETA_GRAFICAS_DSH / f"{dataset}_L{LONG_VENTANA}_H{H}_lr.png")
            grafica_velocidad_multi(dataset, H, historiales, CARPETA_GRAFICAS_DSH / f"{dataset}_L{LONG_VENTANA}_H{H}_velocidad.png")

            # Serie MULTI (solo TreeDRNet, pero mantenemos interfaz)
            modelos_lista = [(n, modelos_inst[n]) for n in modelos_inst.keys()]
            hist_real, fut_real, preds = obtener_hist_y_preds_ultima_ventana(
                modelos=modelos_lista, ds_test=ds_te, device=disp,
                idx_col_objetivo=idx_obj, desnormalizar=False, esc_y=None
            )
            grafica_serie_test_multi(
                dataset=dataset, L=LONG_VENTANA, H=H,
                hist_real=hist_real, futuro_real=fut_real, preds=preds,
                ruta_salida=CARPETA_GRAFICAS_DSH / f"{dataset}_L{LONG_VENTANA}_H{H}_serie_test_MULTI.png"
            )

        # CSV por dataset y comparativas por horizonte
        save_append_csv(CSV_TEST_DS, filas_test_ds)
        grafica_metricas_horizontes(
            ruta_csv=CSV_TEST_DS, dataset=dataset,
            ruta_salida=CARPETA_GRAFICAS_DS_CONS / f"{dataset}_metricas.png"
        )

    # CSV global
    ensure_dirs(CARPETA_METRICAS_GLOBAL)
    save_append_csv(CARPETA_METRICAS_GLOBAL / "TEST_resultados.csv", filas_test_global)
    print("\nMétricas de TEST (global) →", CARPETA_METRICAS_GLOBAL / "TEST_resultados.csv")
    print("Resultados por dataset en: resultados/<DATASET>/H*/{pesos,metricas,graficas} + metricas/ + graficas/")
