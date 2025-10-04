# TreeDRNet: Forecasting de Series de Tiempo en el Dataset ETT

Este proyecto implementa el modelo **TreeDRNet** (Tree-based Deep Residual Network) para la predicción de series de tiempo a largo plazo utilizando el dataset ETT (Electricity Transformer Temperature).

## 📋 Descripción del Proyecto

El objetivo principal es aplicar el approach propuesto en el paper **"TreeDRNet: A Robust Deep Model for Long Term Time Series Forecasting"** para predecir la temperatura del aceite (OT - Oil Temperature) en transformadores eléctricos a través de múltiples horizontes temporales.

### Dataset

Utilizamos el **ETDataset** de [zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset), que contiene datos de transformadores eléctricos registrados entre 2016/07 y 2018/07. El dataset incluye las siguientes variables:

- **HUFL**: High UseFul Load
- **HULL**: High UseLess Load  
- **MUFL**: Middle UseFul Load
- **MULL**: Middle UseLess Load
- **LUFL**: Low UseFul Load
- **LULL**: Low UseLess Load
- **OT**: Oil Temperature (variable objetivo)

Trabajamos con cuatro variantes del dataset:
- `ETTh1`: Transformador 1, muestreo horario
- `ETTh2`: Transformador 2, muestreo horario
- `ETTm1`: Transformador 1, muestreo por minuto
- `ETTm2`: Transformador 2, muestreo por minuto

## 🎯 Configuración Experimental

### Ventanas de Tiempo
- **Longitud de entrada (L)**: 96 pasos temporales
- **Horizontes de predicción (H)**: {24, 48, 96, 192, 336, 720} pasos

### Split Temporal
Los datos se dividen de forma temporal (sin shuffle) en:
- **Train**: 70% de los datos
- **Validación**: 10% de los datos  
- **Test**: 20% de los datos

### Métricas de Evaluación
Para cada horizonte se reportan las siguientes métricas en el conjunto de test:
- **MSE** (Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)
- **R²** (Coeficiente de determinación)

## 🏗️ Arquitectura del Modelo

El modelo TreeDRNet implementado incluye:

- **Estructura jerárquica tipo árbol** con residuales profundos
- **Múltiples ramas paralelas** con mecanismos de gating
- **Convoluciones 1x1** para procesamiento de covariables temporales
- **Backcast y forecast** en cada bloque residual
- **Dropout regularization** para prevenir overfitting

### Hiperparámetros Principales
```python
EPOCAS = 60
BATCH = 32
LR = 1e-4
SEED = 42

TD_OCULTO = 128        # Hidden dimension
TD_PROF = 3            # Tree depth
TD_RAMAS = 2           # Number of branches
```

### Técnicas de Optimización
- **Optimizer**: AdamW con weight decay = 1e-2
- **Learning Rate Scheduler**: ReduceLROnPlateau
- **Early Stopping**: Paciencia de 8 épocas
- **Gradient Clipping**: max_norm = 1.0
- **Mixed Precision Training**: AMP con bfloat16

## 📁 Estructura del Proyecto

```
Lab2/
├── datos/
│   └── ETT-small/
│       ├── ETTh1.csv
│       ├── ETTh2.csv
│       ├── ETTm1.csv
│       └── ETTm2.csv
├── src/
│   ├── modelos/
│   │   └── treedrnet.py          # Implementación del modelo TreeDRNet
│   ├── preprocesamiento.py       # Carga y preparación de datos
│   ├── entrenamiento.py          # Loop de entrenamiento y evaluación
│   ├── graficas.py               # Visualización de resultados
│   └── utils.py                  # Utilidades generales
├── resultados/
│   ├── metricas/                 # Métricas consolidadas
│   └── <DATASET>/                # Por cada dataset
│       ├── H<horizonte>/
│       │   ├── pesos/            # Checkpoints del modelo
│       │   ├── metricas/         # CSV con historial de entrenamiento
│       │   └── graficas/         # Gráficas de métricas y predicciones
│       ├── metricas/             # Resultados por dataset
│       └── graficas/             # Comparativas por horizonte
├── experimentos.py               # Script principal de ejecución
└── README.md
```

## 🚀 Uso

### Requisitos

```bash
pip install torch numpy pandas scikit-learn matplotlib
```

### Ejecución

Para ejecutar todos los experimentos (4 datasets × 6 horizontes):

```bash
python experimentos.py
```

El script automáticamente:
1. Carga cada dataset (ETTh1, ETTh2, ETTm1, ETTm2)
2. Para cada horizonte {24, 48, 96, 192, 336, 720}:
   - Crea ventanas deslizantes de longitud 96
   - Entrena el modelo TreeDRNet
   - Guarda el mejor checkpoint según MSE en validación
   - Evalúa en el conjunto de test
   - Genera gráficas de métricas y predicciones
3. Consolida resultados globales en CSV

### Resultados

Los resultados se guardan automáticamente en:

- **Pesos del modelo**: `resultados/<DATASET>/H<H>/pesos/`
- **Métricas de entrenamiento**: `resultados/<DATASET>/H<H>/metricas/`
- **Gráficas individuales**: `resultados/<DATASET>/H<H>/graficas/`
- **Métricas consolidadas**: `resultados/metricas/TEST_resultados.csv`

## 📊 Visualizaciones

El proyecto genera automáticamente las siguientes gráficas:

### Por Experimento (dataset + horizonte)
- Evolución de MSE, MAE, RMSE, MAPE, R² durante entrenamiento
- Learning rate por época
- Velocidad de entrenamiento (iteraciones/s, muestras/s)
- Serie temporal de la última ventana de test (predicción vs real)

### Por Dataset
- Comparativa de métricas vs horizonte de predicción

## 🔬 Detalles de Implementación

### Preprocesamiento
- Normalización usando `StandardScaler` (ajustado solo en train)
- Ventanas deslizantes sin solapamiento entre splits
- Predicción univariada (OT) con covariables multivariadas

### Entrenamiento
- Dataset loader con pin_memory y persistent_workers para eficiencia
- Checkpoint basado en mejor MSE de validación
- Soporte para CUDA con mixed precision training

### Evaluación
- MAPE calculado en escala original (desnormalizado)
- Métricas promediadas sobre todas las ventanas de test

## 📖 Referencias

### Paper Implementado

**TreeDRNet: A Robust Deep Model for Long Term Time Series Forecasting**

Zhou, T., Zhu, J., Wang, X., Ma, Z., Wen, Q., Sun, L., & Jin, R. (2022). TreeDRNet: A Robust Deep Model for Long Term Time Series Forecasting. arXiv preprint arXiv:2206.12106.

**Paper**: [https://arxiv.org/abs/2206.12106](https://arxiv.org/abs/2206.12106)

```bibtex
@article{zhou2022treedrnet,
  title={TreeDRNet: A Robust Deep Model for Long Term Time Series Forecasting},
  author={Zhou, Tian and Zhu, Jianqing and Wang, Xue and Ma, Ziqing and Wen, Qingsong and Sun, Liang and Jin, Rong},
  journal={arXiv preprint arXiv:2206.12106},
  year={2022}
}
```

**Características clave del paper**:
- Doubly Residual (DRes) link structure para predicciones más robustas
- Estructura de árbol para ensemble de modelos
- Mecanismo de gating para selección de features
- Basado completamente en MLPs (10x más eficiente que Transformers)
- Reduce errores de predicción en 20-40% comparado con SOTA

### Dataset

Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. (2021). Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. AAAI Conference on Artificial Intelligence, 35(12), 11106-11115.

**Dataset Repository**: [https://github.com/zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset)

```bibtex
@inproceedings{haoyietal-informer-2021,
  author    = {Haoyi Zhou and Shanghang Zhang and Jieqi Peng and 
               Shuai Zhang and Jianxin Li and Hui Xiong and Wancai Zhang},
  title     = {Informer: Beyond Efficient Transformer for Long Sequence 
               Time-Series Forecasting},
  booktitle = {The Thirty-Fifth {AAAI} Conference on Artificial Intelligence},
  volume    = {35},
  number    = {12},
  pages     = {11106--11115},
  publisher = {{AAAI} Press},
  year      = {2021},
}
```

## 📝 Notas

- El proyecto sigue principios de clean code sin comentarios
- La arquitectura utiliza clean architecture patterns
- Los tests siguen un approach basado en tablas de casos
- Seed fijada (42) para reproducibilidad

## 🔧 Personalización

Para modificar los experimentos, edita los parámetros en `experimentos.py`:

```python
DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
HORIZONTES = [24, 48, 96, 192, 336, 720]
LONG_VENTANA = 96

EPOCAS = 60
BATCH = 32
LR = 1e-4

TD_OCULTO = 128
TD_PROF = 3
TD_RAMAS = 2
```

