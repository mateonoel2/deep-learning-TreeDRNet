# TreeDRNet: Forecasting de Series de Tiempo en el Dataset ETT

Implementación del modelo **TreeDRNet** para predicción de series de tiempo a largo plazo en el dataset ETT (Electricity Transformer Temperature).

## 📋 Objetivo

Aplicar el approach del paper **"TreeDRNet: A Robust Deep Model for Long Term Time Series Forecasting"** (Zhou et al., 2022) para predecir la temperatura del aceite (OT) en transformadores eléctricos.

## 🎯 Configuración Experimental

- **Dataset**: ETTh1, ETTh2, ETTm1, ETTm2 ([ETDataset](https://github.com/zhouhaoyi/ETDataset))
- **Input Length (L)**: 96 pasos temporales
- **Horizontes (H)**: {24, 48, 96, 192, 336, 720}
- **Split**: 70% train / 10% val / 20% test (temporal)
- **Métricas principales**: **MSE y MAE** por horizonte

## 📚 Documentación Detallada

### 1. [METODOLOGIA.md](METODOLOGIA.md) 📄
**Contenido del paper TreeDRNet**:
- Motivación y problema a resolver
- Arquitectura completa (DRes, Gating, Tree Structure)
- Fundamentos teóricos (robust regression, Kolmogorov-Arnold)
- Ventajas vs SOTA (20-40% mejora, 10× más rápido)

### 2. [IMPLEMENTACION.md](IMPLEMENTACION.md) 🏗️
**Detalles técnicos de implementación**:
- Jerarquía de módulos (DResBlock → GatedBranch → MultiBranchBlock → TreeDRNet)
- Pipeline de datos (carga, normalización, ventanas)
- Loop de entrenamiento (optimizer, scheduler, early stopping)
- Configuración de hiperparámetros

### 3. [RESULTADOS.md](RESULTADOS.md) 📊
**Métricas, análisis y discusión**:
- Curvas de aprendizaje observadas
- Performance por horizonte de predicción
- Ablation studies propuestos
- Limitaciones y trabajos futuros

## 🚀 Quick Start

### Instalación
```bash
pip install torch numpy pandas scikit-learn matplotlib
```

### Ejecución
```bash
python experimentos.py
```

### Estructura del Proyecto
```
Lab2/
├── datos/ETT-small/              # Datasets ETTh1, ETTh2, ETTm1, ETTm2
├── src/
│   ├── modelos/treedrnet.py      # Modelo TreeDRNet
│   ├── preprocesamiento.py       # Carga y ventanas
│   ├── entrenamiento.py          # Train/eval loops
│   └── graficas.py               # Visualizaciones
├── resultados/                   # Métricas y gráficas
├── experimentos.py               # Script principal
└── *.md                          # Documentación
```

## 📖 Referencias

**Paper**: Zhou, T., et al. (2022). TreeDRNet: A Robust Deep Model for Long Term Time Series Forecasting. [arXiv:2206.12106](https://arxiv.org/abs/2206.12106)

**Dataset**: Zhou, H., et al. (2021). Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. AAAI 2021. [ETDataset Repository](https://github.com/zhouhaoyi/ETDataset)

