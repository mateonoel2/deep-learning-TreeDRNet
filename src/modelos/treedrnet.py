import torch
import torch.nn as nn

# Bloque residual: divide en dos cosas
# - backcast = lo que se puede explicar del pasado (como "lo que ya se vio")
# - forecast = lo que aporta al futuro (lo que quiero predecir)
class BloqueResidual(nn.Module):
    def __init__(self, dim_in, dim_hidden, horizonte, dropout=0.10):
        super().__init__()
        # red para sacar el backcast (mapea al mismo tamaño de entrada)
        self.back = nn.Sequential(
            nn.Linear(dim_in, dim_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_in)
        )
        # red para sacar el forecast (mapea al horizonte futuro)
        self.fore = nn.Sequential(
            nn.Linear(dim_in, dim_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim_hidden, horizonte)
        )
        # normalización del residuo
        self.ln = nn.LayerNorm(dim_in)

    def forward(self, x):
        back = self.back(x)          # parte explicada
        resid = self.ln(x - back)    # lo que sobra se vuelve nuevo residuo
        fore = self.fore(x)          # parte que empuja al futuro
        return resid, fore           # devuelvo ambos


class TreeDRNet(nn.Module):
    """
    idea general:
    (B,L,F) → saco vector de features (último paso)
    → lo mando por un árbol de bloques residuales (profundidad x ramas)
    → voy sumando los forecasts
    → al final paso por un MLP y obtengo (B,H,1)
    """
    def __init__(self, entrada_dim, salida_dim=1, horizonte=24,
                 tam_oculto=128, profundidad=2, num_ramas=2, dropout=0.10):
        super().__init__()
        self.horizonte = horizonte
        self.salida_dim = salida_dim
        self.dim_in = entrada_dim

        # árbol = lista de niveles, cada nivel tiene varias ramas (bloques en paralelo)
        self.niveles = nn.ModuleList([
            nn.ModuleList([
                BloqueResidual(self.dim_in, tam_oculto, horizonte, dropout=dropout)
                for _ in range(num_ramas)
            ])
            for _ in range(profundidad)
        ])

        # ajuste final después de sumar forecasts
        self.final = nn.Sequential(
            nn.Linear(horizonte, horizonte),
            nn.ReLU(),
            nn.Linear(horizonte, horizonte * salida_dim)
        )

    def forward(self, x):
        # si la entrada es (B,L,F) → me quedo solo con el último paso
        if x.dim() == 3:
            B, L, F = x.shape
            x_feat = x[:, -1, :]
        else:
            x_feat = x  # si ya está como vector, lo uso directo

        resid = x_feat
        forecast_total = 0.0  # acá voy acumulando todos los forecasts

        # recorro
