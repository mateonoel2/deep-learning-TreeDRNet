# Implementación TreeDRNet

> **Tiempo de presentación**: 5 minutos  
> **Objetivo**: Explicar detalles técnicos de la implementación

## 📑 Índice de Contenidos (Timing Sugerido)

1. **Arquitectura de Módulos** (1 min) - DResBlock → TreeDRNet
2. **Estructura del Árbol** (2 min) - Forward pass y bug corregido
3. **Pipeline de Datos** (1 min) - Carga, normalización, ventanas
4. **Optimización** (1 min) - Entrenamiento, scheduler, checkpointing

---

## 🏗️ Arquitectura Implementada

### Jerarquía de Módulos

```
TreeDRNet
├── Conv1D (opcional)
└── MultiBranchBlock
    └── GatedBranch (×K)
        ├── Gate Network
        └── DResBlock
            ├── MLP
            ├── Backcast Head
            └── Forecast Head
```

---

## 📦 Módulos Principales

### 1. DResBlock

**Responsabilidad**: Implementar el bloque doubly residual básico.

```python
class DResBlock(nn.Module):
    def __init__(self, dim_in, hidden, horizonte, mlp_depth=2, dropout=0.0):
        self.mlp = Sequential(
            Linear(dim_in, hidden), ReLU(),
            [Dropout(dropout) if dropout > 0],
            Linear(hidden, hidden), ReLU(),
            ...
        )
        self.backcast = Linear(hidden, dim_in)
        self.forecast = Linear(hidden, horizonte)
```

**Forward**:
```python
h = self.mlp(x_flat)
backcast = self.backcast(h)  # Reconstrucción del input
forecast = self.forecast(h)   # Predicción del horizonte
return backcast, forecast
```

**Dimensiones**:
- Input: `(batch, L×D)` donde L=96, D=7 para ETT
- Output backcast: `(batch, L×D)`
- Output forecast: `(batch, H)` donde H ∈ {24, 48, ..., 720}

---

### 2. GatedBranch

**Responsabilidad**: Implementar selección de features + procesamiento.

```python
class GatedBranch(nn.Module):
    def __init__(self, dim_in, hidden_gate, hidden_core, horizonte, ...):
        self.gate = Sequential(
            Linear(dim_in, hidden_gate), 
            ReLU(),
            Linear(hidden_gate, dim_in), 
            Sigmoid()  # ← Importante: valores en [0,1]
        )
        self.core = DResBlock(dim_in, hidden_core, horizonte, ...)
```

**Forward**:
```python
gate_weights = self.gate(x_flat)      # (batch, L×D)
x_selected = x_flat * gate_weights    # Element-wise
backcast, forecast = self.core(x_selected)
return backcast, forecast
```

**Intuición**: El gate aprende qué features son importantes para esta rama específica.

---

### 3. MultiBranchBlock

**Responsabilidad**: Ensemble de K ramas paralelas en cada nodo.

```python
class MultiBranchBlock(nn.Module):
    def __init__(self, dim_in, horizonte, num_ramas, ...):
        self.branches = ModuleList([
            GatedBranch(...) for _ in range(num_ramas)
        ])
```

**Dos modos de forward**:

#### a) Forward regular (ensemble promediado):
```python
def forward(self, x_flat):
    bcs, fcs = [], []
    for branch in self.branches:
        bc, fc = branch(x_flat)
        bcs.append(bc)
        fcs.append(fc)
    
    avg_bc = torch.stack(bcs).mean(dim=0)  # Promedio de backcasts
    avg_fc = torch.stack(fcs).mean(dim=0)  # Promedio de forecasts
    return avg_bc, avg_fc
```

#### b) Forward todas las ramas (para árbol):
```python
def forward_all_branches(self, x_flat):
    bcs, fcs = [], []
    for branch in self.branches:
        bc, fc = branch(x_flat)
        bcs.append(bc)
        fcs.append(fc)
    return bcs, fcs  # Listas de K elementos
```

**Diferencia clave**: 
- `forward()`: Para predicción directa (ensemble simple)
- `forward_all_branches()`: Para crear hijos diferentes en el árbol

---

### 4. TreeDRNet (Modelo Principal)

**Responsabilidad**: Coordinar la estructura de árbol completa.

```python
class TreeDRNet(nn.Module):
    def __init__(self, entrada_dim, salida_dim, horizonte, long_ventana,
                 profundidad_arbol=3, num_ramas=2, ...):
        self.conv1x1 = Conv1d(entrada_dim, entrada_dim, 1)  # Opcional
        self.block = MultiBranchBlock(...)
        self.profundidad = profundidad_arbol
```

**Preprocesamiento**:
```python
def _prep_x(self, x):
    # x shape: (batch, L, D)
    if self.usar_conv:
        x = x.transpose(1, 2)  # → (batch, D, L)
        x = self.conv1x1(x)    # Procesar covariables
        x = x.transpose(1, 2)  # → (batch, L, D)
    
    x = x.reshape(batch, -1)   # → (batch, L×D)
    return x
```

**Forward Pass (Estructura del Árbol)**:
```python
def forward(self, x):
    x0 = self._prep_x(x)           # Flatten y opcional conv
    total_forecast = 0.0
    nodos = [x0]                   # Nivel 0: 1 nodo
    
    for nivel in range(self.profundidad):
        fcs = []
        nuevos_nodos = []
        
        for nodo in nodos:
            # Procesar nodo con todas las ramas
            bcs, fcs_branch = self.block.forward_all_branches(nodo)
            fcs.extend(fcs_branch)  # Acumular forecasts
            
            # Crear hijos (uno por cada rama)
            for bc in bcs:
                nuevo_nodo = nodo - bc
                nuevos_nodos.append(nuevo_nodo)
        
        # Agregar forecasts de este nivel
        nivel_forecast = torch.stack(fcs).mean(dim=0)
        total_forecast += nivel_forecast
        
        # Pasar a siguiente nivel
        nodos = nuevos_nodos
    
    return total_forecast.unsqueeze(-1)  # (batch, H, 1)
```

**Ejemplo de expansión del árbol** (profundidad=3, num_ramas=2):
```
Nivel 0: 1 nodo  → genera 2 forecasts
Nivel 1: 2 nodos → generan 4 forecasts
Nivel 2: 4 nodos → generan 8 forecasts
Nivel 3: 8 nodos → generan 16 forecasts

Total forecasts = 2 + 4 + 8 + 16 = 30
Predicción final = promedio de los 30 forecasts
```

---

## 🔧 Detalles Técnicos Importantes

### 1. Por qué `forward_all_branches`?

**Problema original**:
```python
# ❌ INCORRECTO (bug original)
nuevos_nodos.extend([nodo - bc, nodo - bc])  # Duplicados idénticos!
```

**Solución**:
```python
# ✅ CORRECTO
bcs, fcs = self.block.forward_all_branches(nodo)
for bc in bcs:  # Cada rama genera un backcast diferente
    nuevos_nodos.append(nodo - bc)
```

**Por qué importa**:
- Cada rama tiene su propio gate → selecciona diferentes features
- Backcasts diferentes → nodos hijos diferentes
- Diversidad en el árbol → mejor ensemble

### 2. Manejo de Dimensiones

```python
Input: (batch=32, L=96, D=7)
↓ _prep_x + flatten
(32, 672)  # 96 × 7
↓ MultiBranchBlock
backcasts: (32, 672)
forecasts: (32, H)
↓ Acumulación por niveles
total_forecast: (32, H)
↓ unsqueeze
Output: (32, H, 1)
```

### 3. Conv1D para Covariables

**Motivación**: Procesar las D=7 covariables temporales antes de flatten.

```python
Conv1d(in_channels=7, out_channels=7, kernel_size=1)
```

- **kernel_size=1**: Solo mezcla entre covariables, no temporal
- **in=out**: Mantiene número de variables
- **Efecto**: Aprende combinaciones lineales de covariables útiles

---

## 🎛️ Configuración de Hiperparámetros

### Valores Utilizados

```python
EPOCAS = 60
BATCH = 32
LR = 1e-4
SEED = 42

# Modelo
TD_OCULTO = 128        # hidden_gate y hidden_core
TD_PROF = 3            # profundidad_arbol
TD_RAMAS = 2           # num_ramas
MLP_DEPTH = 2          # Capas por MLP
DROPOUT = 0.10

# Optimización
USAR_AMP = True        # Mixed precision (bfloat16)
USAR_SCHEDULER = True  # ReduceLROnPlateau
FACTOR_SCHED = 0.5
PACIENCIA_SCHED = 3

# Early Stopping
USAR_EARLYSTOP = True
PACIENCIA_ES = 8
MIN_DELTA_ES = 1e-5
```

### Trade-offs

**Profundidad del Árbol (TD_PROF)**:
- ↑ Profundidad → Más forecasts, más capacidad
- ↑ Profundidad → Más nodos, más cómputo
- Profundidad=3: 2 + 4 + 8 = 14 forecasts promediados
- Profundidad=4: 2 + 4 + 8 + 16 = 30 forecasts promediados

**Número de Ramas (TD_RAMAS)**:
- ↑ Ramas → Más diversidad en ensemble
- ↑ Ramas → Crecimiento exponencial de nodos (K^depth)
- num_ramas=2 con depth=3: 8 nodos finales
- num_ramas=3 con depth=3: 27 nodos finales

---

## 📊 Pipeline de Datos

### 1. Carga y Split

```python
# src/preprocesamiento.py
def cargar_ett(dataset, long_ventana, horizonte, col_objetivo="OT"):
    df = pd.read_csv(f"datos/ETT-small/{dataset}.csv")
    
    # Split temporal (sin shuffle)
    n = len(df)
    n_train = int(n * 0.70)  # 70% train
    n_val = int(n * 0.10)    # 10% val
    # Resto (20%) es test
    
    df_train = df[:n_train]
    df_val = df[n_train:n_train+n_val]
    df_test = df[n_train+n_val:]
    
    return ds_train, ds_val, ds_test, info
```

### 2. Normalización

```python
# Ajustar scalers SOLO en train
esc_x = StandardScaler().fit(df_train[features])
esc_y = StandardScaler().fit(df_train[["OT"]])

# Aplicar a todos los sets
X_train = esc_x.transform(df_train[features])
y_train = esc_y.transform(df_train[["OT"]])
# ... similar para val y test
```

**Importante**: Evita data leakage usando solo estadísticas de train.

### 3. Ventanas Deslizantes

```python
class DatasetVentanasETT(Dataset):
    def __getitem__(self, i):
        L, H = self.long_ventana, self.horizonte
        
        x = self.X[i : i+L]         # (L, D) - entrada
        y = self.y[i+L : i+L+H]     # (H, 1) - target
        
        return torch.from_numpy(x), torch.from_numpy(y)
```

**Ejemplo** (L=96, H=24):
```
Index 0:  X = datos[0:96],    Y = datos[96:120]
Index 1:  X = datos[1:97],    Y = datos[97:121]
...
```

### 4. DataLoader Optimizado

```python
dl_train = DataLoader(
    ds_train,
    batch_size=32,
    shuffle=True,           # Solo en train
    num_workers=6,          # Carga paralela
    pin_memory=True,        # Optimización GPU
    persistent_workers=True,# Mantener workers vivos
    prefetch_factor=2       # Pre-cargar batches
)
```

---

## 🏋️ Loop de Entrenamiento

### 1. Optimización

```python
optimizer = AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-2  # Regularización L2
)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,        # LR → LR/2
    patience=3,        # Esperar 3 épocas sin mejora
    min_lr=1e-6
)
```

### 2. Mixed Precision Training

```python
scaler = GradScaler(enabled=True)

for xb, yb in train_loader:
    with autocast(dtype=torch.bfloat16):  # Precisión reducida
        pred = model(xb)
        loss = mse_loss(pred, yb)
    
    scaler.scale(loss).backward()
    clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
    scaler.step(optimizer)
    scaler.update()
```

**Beneficio**: ~40% más rápido con pérdida mínima de precisión.

### 3. Early Stopping

```python
class EarlyStopping:
    def step(self, val_loss):
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
            return False  # Continuar
        else:
            self.counter += 1
            return self.counter >= self.patience
```

### 4. Checkpointing

```python
if val_mse < best_val_mse:
    best_val_mse = val_mse
    torch.save({
        'modelo': model.state_dict(),
        'epoch': epoch,
        'val_mse': val_mse
    }, checkpoint_path)
```

**Estrategia**: Guardar solo el mejor modelo según MSE de validación.

---

## 📈 Evaluación

### Métricas Implementadas

```python
def evaluar_test(modelo, ds_test, device, esc_y, mape_original=True):
    for xb, yb in test_loader:
        pred = modelo(xb)
        
        # Métricas en escala normalizada
        mse = MSELoss()(pred, yb)
        mae = L1Loss()(pred, yb)
        rmse = sqrt(mse)
        r2 = 1 - SS_res/SS_tot
        
        # MAPE en escala original (desnormalizado)
        if mape_original:
            y_orig = esc_y.inverse_transform(yb)
            p_orig = esc_y.inverse_transform(pred)
            mape = mean(|y_orig - p_orig| / |y_orig|) * 100
```

**Por qué MAPE original?**
- MSE/MAE en escala normalizada son comparables entre horizontes
- MAPE necesita escala original para interpretabilidad (%)

---

## 🎨 Visualizaciones

### Gráficas Generadas

1. **Métricas por época**: MSE, MAE, RMSE, MAPE, R² en validación
2. **Learning rate**: Evolución con scheduler
3. **Velocidad**: Iteraciones/s y muestras/s
4. **Serie temporal**: Última ventana de test (historial + predicción vs real)
5. **Comparativas**: Métricas vs horizonte de predicción

---

## 🔍 Debugging Tips

### Verificar Shapes

```python
print(f"Input: {x.shape}")           # (32, 96, 7)
print(f"After prep: {x0.shape}")     # (32, 672)
print(f"Backcast: {bc.shape}")       # (32, 672)
print(f"Forecast: {fc.shape}")       # (32, H)
print(f"Output: {y.shape}")          # (32, H, 1)
```

### Verificar Gradientes

```python
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm().item():.4f}")
```

### Verificar Diversidad de Ramas

```python
bcs, fcs = model.block.forward_all_branches(x0)
print(f"Backcast 0: {bcs[0][:5]}")
print(f"Backcast 1: {bcs[1][:5]}")
# Deben ser diferentes!
```

---

## 📝 Clean Code Practices

1. **Sin comentarios**: El código se auto-documenta con nombres claros
2. **Type hints**: `def func(x: torch.Tensor) -> torch.Tensor`
3. **Separación de concerns**: Cada módulo tiene una responsabilidad
4. **DRY**: No repetir lógica (ej. `forward_all_branches`)
5. **Configuración centralizada**: Todos los hiperparámetros en `experimentos.py`

