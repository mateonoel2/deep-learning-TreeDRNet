# Metodología TreeDRNet

## Paper Original

**TreeDRNet: A Robust Deep Model for Long Term Time Series Forecasting**  
Zhou, T., Zhu, J., Wang, X., Ma, Z., Wen, Q., Sun, L., & Jin, R. (2022)  
[arXiv:2206.12106](https://arxiv.org/abs/2206.12106)

## Motivación

### Problema a Resolver

**Long-term Time Series Forecasting**: Predecir múltiples pasos futuros (ej. 720 horizones) a partir de secuencias largas de entrada.

### Limitaciones de Métodos Existentes

1. **Transformers**:
   - Alta complejidad computacional: O(n²)
   - Deterioro de performance con secuencias largas
   - Tendencia al overfitting con modelos muy complejos

2. **Modelos tradicionales**:
   - Incapaces de capturar patrones complejos no lineales
   - Sensibles a ruido y outliers
   - Pobre generalización

---

## Arquitectura TreeDRNet

### 1. Doubly Residual (DRes) Structure

Inspirado en **robust regression**, cada bloque DRes produce:

```
Input x → MLP → {backcast, forecast}
```

- **Backcast (bc)**: Reconstrucción del input para capturar patrones históricos
- **Forecast (fc)**: Predicción del horizonte futuro

**Ventaja**: La separación backcast/forecast mejora la estabilidad y robustez de las predicciones.

```python
residual = x - backcast  # Lo que no se capturó
prediction = Σ forecasts  # Agregación de predicciones
```

### 2. Gating Mechanism (Feature Selection)

Cada rama tiene un **gate network** que aprende a seleccionar features relevantes:

```
gate = σ(MLP(x))           # σ = sigmoid
x_selected = x ⊙ gate      # ⊙ = element-wise multiplication
```

**Ventaja**: 
- Selección automática de features importantes
- Diferentes ramas seleccionan diferentes aspectos
- Reduce ruido y mejora robustez

### 3. Tree Structure (Hierarchical Ensemble)

Basado en el **Teorema de Kolmogorov-Arnold**: funciones multivariadas pueden descomponerse jerárquicamente.

```
Nivel 0:           [x₀]
                   ↙  ↘
Nivel 1:      [x₀-bc₁] [x₀-bc₂]
              ↙ ↘      ↙ ↘
Nivel 2:   [...]  [...] [...] [...]
```

Cada nivel:
- **Procesa** el nodo actual → genera forecast
- **Divide** el nodo en K ramas (hijos) usando diferentes backcasts
- **Acumula** forecasts para predicción final

**Ventaja**:
- Ensemble de múltiples "expertos" (cada rama)
- Representación jerárquica de patrones
- Diversidad entre ramas mejora robustez

### 4. Multi-Branch per Node

En cada nodo, K ramas paralelas procesan el mismo input:

```
       x_node
     /   |   \
   br₁  br₂  br₃
    |    |    |
  bc₁  bc₂  bc₃  → K diferentes backcasts
  fc₁  fc₂  fc₃  → K diferentes forecasts
```

**Ventaja**: Ensemble interno en cada nodo aumenta diversidad.

---

## Fundamentos Teóricos

### 1. Robust Regression

TreeDRNet adapta ideas de regresión robusta:
- **Residuos múltiples**: Cada nivel refina el residuo anterior
- **M-estimators**: Múltiples ramas actúan como estimadores robustos
- **Resistencia a outliers**: Ensemble reduce impacto de predicciones anómalas

### 2. Kolmogorov-Arnold Representation Theorem

> Toda función continua multivariada f(x₁, ..., xₙ) puede representarse como:
> 
> f(x) = Σᵢ φᵢ(Σⱼ ψᵢⱼ(xⱼ))

TreeDRNet implementa esto mediante:
- **Outer sum** (Σᵢ): Suma de forecasts a través de niveles
- **Inner sum** (Σⱼ): Combinación de features vía gates y MLPs
- **Hierarchy**: Estructura de árbol = descomposición funcional

### 3. Ensemble Learning

- **Bagging effect**: Múltiples ramas → reducción de varianza
- **Boosting effect**: Múltiples niveles → refinamiento secuencial
- **Diversidad**: Gates diferentes → especializaciones distintas

---

## Algoritmo de Forward Pass

```
Input: x ∈ ℝ^(L×D)  # L=longitud ventana, D=dimensiones

1. Preprocesar:
   x₀ = Conv1D(x) si usar_conv
   x₀ = flatten(x₀)  # Shape: (L×D,)

2. Inicializar:
   nodos = [x₀]
   total_forecast = 0

3. Para cada nivel l en árbol:
   nuevos_nodos = []
   level_forecasts = []
   
   Para cada nodo en nodos:
      Para cada rama k en [1..K]:
         gate_k = sigmoid(MLP_gate_k(nodo))
         x_sel = nodo ⊙ gate_k
         bc_k = MLP_backcast_k(x_sel)
         fc_k = MLP_forecast_k(x_sel)
         
         level_forecasts.append(fc_k)
         nuevos_nodos.append(nodo - bc_k)
   
   total_forecast += mean(level_forecasts)
   nodos = nuevos_nodos

4. Return: total_forecast
```

---

## Ventajas Clave del Método

### 1. Eficiencia Computacional
- **Solo MLPs**: O(n) vs O(n²) de attention
- **10× más rápido** que Transformers según el paper
- Paralelizable en GPU de forma natural

### 2. Robustez
- **Ensemble múltiple**: Reduce overfitting
- **Doubly residual**: Maneja mejor patrones complejos
- **Gating**: Filtra ruido automáticamente

### 3. Long-term Forecasting
- **Estructura jerárquica**: Captura patrones multi-escala
- **Refinamiento progresivo**: Cada nivel mejora la predicción
- **Representación rica**: Árbol aumenta capacidad expresiva

### 4. Interpretabilidad (parcial)
- **Gates**: Muestran qué features son importantes
- **Niveles**: Cada nivel captura diferentes aspectos
- **Ramas**: Especializaciones visualizables

---

## Resultados Reportados en el Paper

### Performance vs State-of-the-Art

En datasets de benchmark (ETT, Weather, Electricity):
- **20-40% reducción** en MSE/MAE comparado con Informer, Autoformer, FEDformer
- **Mejor robustez** en horizontes largos (H=720)
- **10× más rápido** en entrenamiento e inferencia

### Ablation Studies

| Componente Removido | Impacto en MSE |
|-------------------|----------------|
| Sin Gating | +15-25% |
| Sin Tree (depth=1) | +10-18% |
| Sin DRes | +12-20% |
| Sin Multi-branch | +8-15% |

**Conclusión**: Todos los componentes son importantes para el performance.

---

## Hiperparámetros Clave

| Parámetro | Descripción | Valor Típico |
|-----------|-------------|--------------|
| `tree_depth` | Profundidad del árbol | 2-4 |
| `num_branches` | Ramas por nodo | 2-4 |
| `hidden_dim` | Dimensión oculta MLPs | 128-256 |
| `mlp_depth` | Capas en cada MLP | 2-3 |
| `dropout` | Regularización | 0.1-0.2 |

**Trade-offs**:
- ↑ `tree_depth`: Más capacidad pero más parámetros
- ↑ `num_branches`: Más diversidad pero más costo computacional
- ↑ `hidden_dim`: Más expresividad pero mayor riesgo de overfitting

---

## Comparación con Otros Métodos

| Método | Complejidad | Long-term | Robustez | Eficiencia |
|--------|-------------|-----------|----------|------------|
| ARIMA | O(n) | ❌ Pobre | ⚠️ Media | ✅ Alta |
| LSTM | O(n) | ⚠️ Media | ⚠️ Media | ✅ Alta |
| Transformer | O(n²) | ✅ Buena | ⚠️ Media | ❌ Baja |
| **TreeDRNet** | **O(n)** | **✅ Muy Buena** | **✅ Alta** | **✅ Alta** |

---

## Conceptos Clave para Recordar

1. **Doubly Residual**: Backcast (reconstrucción) + Forecast (predicción)
2. **Gating**: Selección automática de features relevantes
3. **Tree Structure**: Ensemble jerárquico de modelos
4. **Multi-branch**: Múltiples "expertos" por nodo
5. **MLP-only**: Eficiencia computacional sin sacrificar performance

---

## Aplicabilidad

TreeDRNet es especialmente útil para:

✅ **Series largas** (L > 96)  
✅ **Horizontes largos** (H > 96)  
✅ **Datos ruidosos** (robustez importante)  
✅ **Recursos limitados** (eficiencia crítica)  
✅ **Multi-variable** (covariables disponibles)

❌ Menos ideal para:
- Series muy cortas (< 50 puntos)
- Horizontes muy cortos (H < 24)
- Datos univariados simples

