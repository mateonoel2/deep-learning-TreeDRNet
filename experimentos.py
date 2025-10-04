from __future__ import annotations
from pathlib import Path
from src.utils import get_device, print_env, set_seed, ensure_dirs, save_append_csv, save_epoch_history
from src.preprocesamiento import cargar_ett, crear_loaders
from src.modelos.transformer import ModeloTransformer
from src.modelos.autoformer import Autoformer
from src.modelos.treedrnet import TreeDRNet
from src.entrenamiento import entrenar_validar, evaluar_test, obtener_hist_y_preds_ultima_ventana
from src.graficas import (
    grafica_mse,
    grafica_mae,
    grafica_rmse,
    grafica_mape,
    grafica_r2,
    grafica_lr,
    grafica_velocidad_iters,
    grafica_velocidad_muestras,
    grafica_metricas_horizontes,
    grafica_serie_test_multi,
)


DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
HORIZONTES = [24, 48, 96, 192, 336, 720]
LONG_VENTANA = 96

SEED = 42
USAR_AMP = True
AMP_DTYPE = "bf16"

WORKERS = 0
PERSISTENT_WORKERS = False
PREFETCH_FACTOR = 2
DROP_LAST_TRAIN = False

USAR_SCHEDULER = True
FACTOR_SCHED = 0.5
PACIENCIA_SCHED = 4
MIN_LR = 1e-5

USAR_EARLYSTOP = True
MIN_DELTA_ES = 0.0

ENABLE_TRANSFORMER = True
ENABLE_AUTOFORMER = True
ENABLE_TREEDRNET = True

TR_D_MODELO = 128
TR_CABEZAS = 4
TR_CAPAS = 2
TR_DIM_FF = 256
TR_DROPOUT = 0.10
AF_OCULTO = 128
AF_K_PROM = 25
TD_OCULTO = 128
TD_PROF = 2
TD_RAMAS = 2

CARPETA_RESULTADOS = Path("resultados")
CARPETA_METRICAS_GLOBAL = CARPETA_RESULTADOS / "metricas"


def hiper_por_h(H: int):
    if H <= 96:
        return dict(EPOCAS=50, BATCH=256, LR=5e-4, PACIENCIA_ES=16, AF_K_PROM=25, TD_PROF=2, TD_RAMAS=2)
    elif H <= 336:
        return dict(EPOCAS=60, BATCH=256, LR=3e-4, PACIENCIA_ES=18, AF_K_PROM=31, TD_PROF=3, TD_RAMAS=3)
    else:
        return dict(EPOCAS=80, BATCH=192, LR=2e-4, PACIENCIA_ES=22, AF_K_PROM=49, TD_PROF=4, TD_RAMAS=3)


if __name__ == "__main__":
    set_seed(SEED)
    disp = get_device()
    print_env(disp)
    ensure_dirs(CARPETA_METRICAS_GLOBAL)
    filas_test_global = []
    for dataset in DATASETS:
        CARPETA_RES_DS = CARPETA_RESULTADOS / dataset
        CARPETA_METRICAS_DS_CONS = CARPETA_RES_DS / "metricas"
        CARPETA_GRAFICAS_DS_CONS = CARPETA_RES_DS / "graficas"
        ensure_dirs(CARPETA_RES_DS, CARPETA_METRICAS_DS_CONS, CARPETA_GRAFICAS_DS_CONS)
        CSV_TEST_DS = CARPETA_METRICAS_DS_CONS / "TEST_resultados.csv"
        filas_test_ds = []
        for H in HORIZONTES:
            cfg = hiper_por_h(H)
            EPOCAS = cfg["EPOCAS"]
            BATCH = cfg["BATCH"]
            LR = cfg["LR"]
            PACIENCIA_ES = cfg["PACIENCIA_ES"]
            AF_K_PROM = cfg["AF_K_PROM"]
            TD_PROF = cfg["TD_PROF"]
            TD_RAMAS = cfg["TD_RAMAS"]
            print(f"\n=== DATASET: {dataset} | L={LONG_VENTANA} | H={H} ===")
            CARPETA_RES_DSH = CARPETA_RES_DS / f"H{H}"
            CARPETA_METRICAS_DSH = CARPETA_RES_DSH / "metricas"
            CARPETA_GRAFICAS_DSH = CARPETA_RES_DSH / "graficas"
            ensure_dirs(CARPETA_RES_DSH, CARPETA_METRICAS_DSH, CARPETA_GRAFICAS_DSH)
            ds_tr, ds_va, ds_te, info = cargar_ett(dataset, LONG_VENTANA, H, col_objetivo="OT")
            dl_tr, dl_va = crear_loaders(
                ds_train=ds_tr,
                ds_val=ds_va,
                batch=BATCH,
                workers=WORKERS,
                persistent_workers=PERSISTENT_WORKERS,
                prefetch_factor=PREFETCH_FACTOR,
                drop_last_train=DROP_LAST_TRAIN,
            )
            dim_in = len(info["cols_features"])
            idx_obj = info["cols_features"].index(info["col_objetivo"])
            historiales = {}
            modelos_inst = {}
            if ENABLE_TRANSFORMER:
                nombre = "Transformer"
                modelo = ModeloTransformer(
                    entrada_dim=dim_in,
                    salida_dim=1,
                    horizonte=H,
                    d_modelo=TR_D_MODELO,
                    num_cabezas=TR_CABEZAS,
                    num_capas=TR_CAPAS,
                    dim_ff=TR_DIM_FF,
                    dropout=TR_DROPOUT,
                )
                hist = entrenar_validar(
                    modelo,
                    dl_tr,
                    dl_va,
                    EPOCAS,
                    LR,
                    disp,
                    USAR_AMP,
                    AMP_DTYPE,
                    USAR_SCHEDULER,
                    FACTOR_SCHED,
                    PACIENCIA_SCHED,
                    MIN_LR,
                    USAR_EARLYSTOP,
                    PACIENCIA_ES,
                    MIN_DELTA_ES,
                )
                mse, mae, rmse, mape, r2 = evaluar_test(modelo, ds_te, disp)
                save_epoch_history(CARPETA_METRICAS_DSH / f"{dataset}_TR_L{LONG_VENTANA}_H{H}_hist.csv", hist)
                filas_test_ds.append(
                    {
                        "dataset": dataset,
                        "modelo": "transformer",
                        "L": LONG_VENTANA,
                        "H": H,
                        "test_mse": mse,
                        "test_mae": mae,
                        "test_rmse": rmse,
                        "test_mape": mape,
                        "test_r2": r2,
                    }
                )
                historiales[nombre] = hist
                modelos_inst[nombre] = modelo
            if ENABLE_AUTOFORMER:
                nombre = "Autoformer"
                modelo = Autoformer(
                    entrada_dim=dim_in,
                    salida_dim=1,
                    horizonte=H,
                    tam_oculto=AF_OCULTO,
                    k_promedio=AF_K_PROM,
                    dropout=0.10,
                )
                hist = entrenar_validar(
                    modelo,
                    dl_tr,
                    dl_va,
                    EPOCAS,
                    LR,
                    disp,
                    USAR_AMP,
                    AMP_DTYPE,
                    USAR_SCHEDULER,
                    FACTOR_SCHED,
                    PACIENCIA_SCHED,
                    MIN_LR,
                    USAR_EARLYSTOP,
                    PACIENCIA_ES,
                    MIN_DELTA_ES,
                )
                mse, mae, rmse, mape, r2 = evaluar_test(modelo, ds_te, disp)
                save_epoch_history(CARPETA_METRICAS_DSH / f"{dataset}_AF_L{LONG_VENTANA}_H{H}_hist.csv", hist)
                filas_test_ds.append(
                    {
                        "dataset": dataset,
                        "modelo": "autoformer",
                        "L": LONG_VENTANA,
                        "H": H,
                        "test_mse": mse,
                        "test_mae": mae,
                        "test_rmse": rmse,
                        "test_mape": mape,
                        "test_r2": r2,
                    }
                )
                historiales[nombre] = hist
                modelos_inst[nombre] = modelo
            if ENABLE_TREEDRNET:
                nombre = "TreeDRNet"
                modelo = TreeDRNet(
                    entrada_dim=dim_in,
                    salida_dim=1,
                    horizonte=H,
                    tam_oculto=TD_OCULTO,
                    profundidad=TD_PROF,
                    num_ramas=TD_RAMAS,
                    dropout=0.10,
                )
                hist = entrenar_validar(
                    modelo,
                    dl_tr,
                    dl_va,
                    EPOCAS,
                    LR,
                    disp,
                    USAR_AMP,
                    AMP_DTYPE,
                    USAR_SCHEDULER,
                    FACTOR_SCHED,
                    PACIENCIA_SCHED,
                    MIN_LR,
                    USAR_EARLYSTOP,
                    PACIENCIA_ES,
                    MIN_DELTA_ES,
                )
                mse, mae, rmse, mape, r2 = evaluar_test(modelo, ds_te, disp)
                save_epoch_history(CARPETA_METRICAS_DSH / f"{dataset}_TD_L{LONG_VENTANA}_H{H}_hist.csv", hist)
                filas_test_ds.append(
                    {
                        "dataset": dataset,
                        "modelo": "treedrnet",
                        "L": LONG_VENTANA,
                        "H": H,
                        "test_mse": mse,
                        "test_mae": mae,
                        "test_rmse": rmse,
                        "test_mape": mape,
                        "test_r2": r2,
                    }
                )
                historiales[nombre] = hist
                modelos_inst[nombre] = modelo
            nombres = list(historiales.keys())
            if len(nombres) >= 2:
                A, B = nombres[0], nombres[1]
                hA, hB = historiales[A], historiales[B]
                grafica_mse(
                    dataset, H, hA, hB, A, B, CARPETA_GRAFICAS_DSH / f"{dataset}_L{LONG_VENTANA}_H{H}_mse.png"
                )
                grafica_mae(
                    dataset, H, hA, hB, A, B, CARPETA_GRAFICAS_DSH / f"{dataset}_L{LONG_VENTANA}_H{H}_mae.png"
                )
                grafica_rmse(
                    dataset,
                    H,
                    hA,
                    hB,
                    A,
                    B,
                    CARPETA_GRAFICAS_DSH / f"{dataset}_L{LONG_VENTANA}_H{H}_rmse.png",
                )
                grafica_mape(
                    dataset,
                    H,
                    hA,
                    hB,
                    A,
                    B,
                    CARPETA_GRAFICAS_DSH / f"{dataset}_L{LONG_VENTANA}_H{H}_mape.png",
                )
                grafica_r2(
                    dataset, H, hA, hB, A, B, CARPETA_GRAFICAS_DSH / f"{dataset}_L{LONG_VENTANA}_H{H}_r2.png"
                )
                grafica_lr(
                    dataset, H, hA, hB, A, B, CARPETA_GRAFICAS_DSH / f"{dataset}_L{LONG_VENTANA}_H{H}_lr.png"
                )
            grafica_velocidad_iters(
                dataset,
                H,
                historiales,
                CARPETA_GRAFICAS_DSH / f"{dataset}_L{LONG_VENTANA}_H{H}_vel_iters.png",
            )
            grafica_velocidad_muestras(
                dataset,
                H,
                historiales,
                CARPETA_GRAFICAS_DSH / f"{dataset}_L{LONG_VENTANA}_H{H}_vel_muestras.png",
            )
            modelos_lista = [(n, modelos_inst[n]) for n in modelos_inst.keys()]
            hist_real, fut_real, preds = obtener_hist_y_preds_ultima_ventana(
                modelos_lista, ds_te, disp, idx_obj, desnormalizar=False, esc_y=None
            )
            grafica_serie_test_multi(
                dataset,
                LONG_VENTANA,
                H,
                hist_real,
                fut_real,
                preds,
                CARPETA_GRAFICAS_DSH / f"{dataset}_L{LONG_VENTANA}_H{H}_serie_test_MULTI.png",
                0.4,
            )
        save_append_csv(CARPETA_METRICAS_DS_CONS / "TEST_resultados.csv", filas_test_ds)
        grafica_metricas_horizontes(
            CSV_TEST_DS, dataset, CARPETA_GRAFICAS_DS_CONS / f"{dataset}_metricas.png"
        )
    save_append_csv(CARPETA_METRICAS_GLOBAL / "TEST_resultados.csv", filas_test_global)
