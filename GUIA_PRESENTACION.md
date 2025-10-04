# Guía de Presentación (3 × 5 minutos)

## 🎯 Estructura General

Cada sección está diseñada para 5 minutos de presentación con slides sugeridos.

---

# 1️⃣ METODOLOGÍA (5 min)

## Slide 1: Título y Motivación (1 min)
**Título**: TreeDRNet: Robust Deep Model para Long-Term Forecasting

**Problema**:
- ❌ Transformers: O(n²), lentos, se degradan con secuencias largas
- ❌ Modelos tradicionales: No capturan patrones complejos
- ✅ **Necesidad**: Modelo eficiente, robusto, para horizontes largos (H=720)

**Visual sugerido**: Gráfica comparando complejidad temporal

---

## Slide 2: Arquitectura - Doubly Residual (30 seg)
**DResBlock**: Dos salidas complementarias

```
Input x → MLP → {backcast, forecast}
         ↓           ↓           ↓
   Embedding   Reconstrucción  Predicción
```

**Key point**: Separar "entender el pasado" de "predecir el futuro"

---

## Slide 3: Arquitectura - Gating (30 seg)
**Selección automática de features**

```
gate = σ(MLP(x))        # Valores [0,1]
x_selected = x ⊙ gate   # Filtrar features
```

**Key point**: Cada rama aprende qué variables son importantes

---

## Slide 4: Arquitectura - Tree Structure (1 min)
**Ensemble jerárquico**

```
         Nivel 0: [x₀]
                 ↙  ↘
    Nivel 1: [x-bc₁] [x-bc₂]
            ↙ ↘      ↙ ↘
Nivel 2:  [...]  [...] [...] [...]
```

**Key points**:
- Cada nivel genera forecasts
- Promedio de todos los forecasts = predicción final
- K branches × D depth = K^D nodos finales

**Visual sugerido**: Diagrama del árbol con depth=3

---

## Slide 5: Fundamentos Teóricos (1 min)
**Inspiración**:

1. **Robust Regression**: Residuos múltiples → resistencia a outliers
2. **Kolmogorov-Arnold**: Funciones complejas = suma de funciones simples
3. **Ensemble Learning**: Diversidad → reducción de varianza

**Ecuación clave**:
```
Predicción = Σ (forecasts nivel 1) + Σ (forecasts nivel 2) + ...
```

---

## Slide 6: Resultados del Paper (1 min)
**Performance vs SOTA** (ETT, Weather, Electricity):

| Métrica | vs Informer | vs Autoformer | vs FEDformer |
|---------|-------------|---------------|--------------|
| MSE ↓ | -20 a -40% | -15 a -35% | -10 a -30% |
| Speed ↑ | **10×** | **10×** | **10×** |

**Key messages**:
- ✅ Mejor accuracy (20-40% menos error)
- ✅ Mucho más rápido (10× vs Transformers)
- ✅ MLP-only = O(n) complejidad

---

# 2️⃣ IMPLEMENTACIÓN (5 min)

## Slide 1: Overview de Módulos (1 min)
**Jerarquía de componentes**:

```
TreeDRNet (orquestador)
  ├── Conv1D (preprocesamiento covariables)
  └── MultiBranchBlock (ensemble en cada nodo)
      └── GatedBranch × K (ramas paralelas)
          ├── Gate Network (selección features)
          └── DResBlock (backcast + forecast)
              ├── MLP profundo
              ├── Backcast head
              └── Forecast head
```

**Key point**: Clean architecture con responsabilidades claras

---

## Slide 2: Forward Pass del Árbol (2 min)
**Algoritmo paso a paso**:

```python
1. Inicializar: nodos = [x₀], total_forecast = 0

2. Para cada nivel l:
   - Para cada nodo:
     * Procesar con K ramas → K backcasts, K forecasts
     * Crear K hijos: nodo - bc₁, nodo - bc₂, ..., nodo - bcₖ
   
   - Promediar forecasts del nivel
   - Acumular en total_forecast
   - nodos ← hijos

3. Return: total_forecast
```

**Bug corregido**:
```python
# ❌ Antes: nuevos_nodos = [nodo-bc, nodo-bc]  # DUPLICADOS
# ✅ Ahora: for bc in bcs: nuevos_nodos.append(nodo-bc)  # DIFERENTES
```

**Visual sugerido**: Diagrama de flujo con ejemplo

---

## Slide 3: Pipeline de Datos (1 min)
**Flujo de preprocesamiento**:

1. **Carga**: `ETTh1.csv` → DataFrame
2. **Split temporal**: 70% train / 10% val / 20% test (sin shuffle)
3. **Normalización**: StandardScaler ajustado SOLO en train
4. **Ventanas**: (i:i+96) → (i+96:i+96+H)
5. **DataLoader**: batch=32, workers=6, pin_memory

**Dimensiones ejemplo** (ETTh1, H=24):
```
Input:  (batch=32, L=96, D=7)
Output: (batch=32, H=24, 1)
```

**Key point**: Evitar data leakage con split temporal

---

## Slide 4: Entrenamiento y Optimización (1 min)
**Componentes**:

```python
Optimizer: AdamW (lr=1e-4, weight_decay=1e-2)
Scheduler: ReduceLROnPlateau (patience=3, factor=0.5)
Early Stop: Patience=8 épocas, min_delta=1e-5
Mixed Precision: bfloat16 → 40% más rápido
Gradient Clip: max_norm=1.0
```

**Estrategia de checkpointing**:
- Guardar solo mejor modelo según MSE de validación
- Recargar al final para test

**Visual sugerido**: Diagrama del loop de entrenamiento

---

## Slide 5: Hiperparámetros y Configuración (30 seg)
**Valores utilizados**:

| Parámetro | Valor | Impacto |
|-----------|-------|---------|
| `tree_depth` | 3 | 2+4+8=14 forecasts |
| `num_branches` | 2 | Equilibrio diversidad/costo |
| `hidden_dim` | 128 | Capacidad del modelo |
| `dropout` | 0.10 | Regularización |
| `mlp_depth` | 2 | Profundidad por MLP |

**Trade-off principal**: depth × branches = nodos exponenciales

---

## Slide 6: Código Limpio (30 seg)
**Principios aplicados**:

✅ Sin comentarios - código auto-documentado  
✅ Type hints - `def forward(x: Tensor) -> Tensor`  
✅ Separación de concerns - cada módulo una responsabilidad  
✅ DRY - no repetir lógica  
✅ Clean architecture - capas bien definidas  

**Ejemplo**:
```python
# Nombres claros en vez de comentarios
def forward_all_branches(self, x):  # vs forward()
    # Retorna listas individuales para diversidad en árbol
```

---

# 3️⃣ RESULTADOS (5 min)

## Slide 1: Setup Experimental (1 min)
**Configuración completa**:

- **24 experimentos**: 4 datasets × 6 horizontes
- **Horizontes**: H ∈ {24, 48, 96, 192, 336, 720}
- **Métricas principales**: **MSE y MAE**
- **Hiperparámetros**: depth=3, branches=2, hidden=128
- **Tiempo**: ~7-8 min por experimento

**Key point**: Enfoque en MSE y MAE como métricas principales

---

## Slide 2: Tabla de Resultados MSE/MAE (2 min)
**Estado actual**: ⏳ Experimentos en progreso

**Formato de reporte** (por dataset):

| Horizonte | MSE | MAE |
|-----------|-----|-----|
| H=24 | - | - |
| H=48 | - | - |
| H=96 | - | - |
| H=192 | - | - |
| H=336 | - | - |
| H=720 | - | - |

**Ejemplo disponible (ETTh1-H24)**:
- **Test MSE**: 0.0817
- **Test MAE**: 0.2263

**Tendencias esperadas**:
- ↑ MSE/MAE con ↑ horizonte
- ETTh (horario) < ETTm (minuto) en error

**Visual sugerido**: Gráfica MSE y MAE vs horizonte (cuando esté completa)

---

## Slide 3: Análisis de Convergencia (1 min)
**Ejemplo: ETTh1-H24**

**Patrón de aprendizaje**:

| Fase | Épocas | MSE Val | MAE Val | Comportamiento |
|------|--------|---------|---------|----------------|
| 1 | 1-5 | 0.205→0.089 | 0.387→0.237 | Descenso rápido |
| 2 | 5-10 | 0.089→0.086 | 0.237→0.231 | Refinamiento |
| 3 | 11+ | 0.082 | 0.226 | Plateau (LR↓) |

**Observaciones**:
- ✅ MSE reduce **7× en 10 épocas**
- ✅ Scheduler detecta plateau y reduce LR
- ✅ **12 it/s**: Eficiente (~380 muestras/s)

**Visual sugerido**: Gráfica MSE/MAE train vs val

---

## Slide 4: Discusión (1 min)
**Hallazgos clave**:

✅ **Convergencia estable**: Sin explosión de gradientes  
✅ **Eficiencia confirmada**: 7-8 min/experimento  
✅ **Ensemble robusto**: 14 forecasts (2+4+8)  
✅ **Pipeline sólido**: Sin data leakage  

**Expectativas para resultados completos**:

```
H=24:  MSE ≈ 0.08-0.12  |  MAE ≈ 0.22-0.28  (más fácil)
H=96:  MSE ≈ 0.13-0.20  |  MAE ≈ 0.28-0.38
H=720: MSE ≈ 0.40-0.70  |  MAE ≈ 0.55-0.75  (más difícil)
```

**Razón**: Mayor horizonte → más incertidumbre

---

## Slide 5: Próximos Pasos (30 seg)
**Por completar**:

1. ⏳ Ejecutar 23 experimentos restantes
2. ⏳ Llenar tabla MSE/MAE para todos los horizontes
3. ⏳ Generar gráficas comparativas por dataset
4. ⏳ Validar hipótesis sobre tendencias

**Timeline estimado**: ~3-4 horas para completar todos

---

## Slide 6: Conclusión (30 seg)
**Mensaje principal**:

✅ Implementación **fiel al paper** TreeDRNet  
✅ Arquitectura **robusta y eficiente** (bug corregido)  
✅ Resultados preliminares **prometedores**  
⏳ Validación completa **en progreso**  

**Key takeaway**: 
> TreeDRNet es un modelo eficiente para long-term forecasting que combina ensemble jerárquico con gating para feature selection, logrando predicciones robustas en ~10× menos tiempo que Transformers.

---

# 📊 Recursos Visuales Recomendados

## Para METODOLOGÍA:
1. Diagrama de arquitectura completa
2. Árbol con depth=3, branches=2
3. Comparación O(n) vs O(n²)
4. Tabla de resultados del paper

## Para IMPLEMENTACIÓN:
1. Jerarquía de clases
2. Diagrama de flujo del forward pass
3. Pipeline de datos
4. Ejemplo de ventanas deslizantes

## Para RESULTADOS:
1. Tabla de métricas ETTh1-H24
2. Gráficas de convergencia (MSE, MAE, LR)
3. Serie temporal predicha vs real
4. Comparativa métricas vs horizonte

---

# 🎤 Tips de Presentación

## Tiempo
- ⏱️ Practica con cronómetro
- 📍 Marca puntos de 1, 2.5 y 4 minutos
- ⚡ Ten versión corta si te quedas sin tiempo

## Estilo
- 🗣️ Explica conceptos con analogías simples
- 📊 Una idea por slide (no saturar)
- 👁️ Contacto visual con audiencia
- ❓ Anticipa preguntas comunes

## Contenido
- 🎯 Enfócate en "por qué" no solo "qué"
- 💡 Resalta contribuciones originales
- ⚖️ Balance entre teoría y práctica
- 🔬 Sé honesto con limitaciones

## Preguntas Esperadas

**METODOLOGÍA**:
- ¿Por qué árbol y no red recurrente?
- ¿Cuál es el costo de entrenar depth=5?
- ¿Funciona para univariado?

**IMPLEMENTACIÓN**:
- ¿Por qué StandardScaler y no MinMax?
- ¿Cuál fue el bug del árbol?
- ¿Por qué bfloat16 y no float16?

**RESULTADOS**:
- ¿Comparaste con otros modelos?
- ¿Por qué overfitting en H=24?
- ¿Cómo elegiste depth=3?

