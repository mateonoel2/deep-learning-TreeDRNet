import torch
import torch.nn as nn
import torch.nn.functional as F

# promedio movil → basicamente para suavizar la serie
# esto me saca la tendencia (lo "lento" de la serie)
def promedio_movil(x: torch.Tensor, k: int = 25) -> torch.Tensor:
    B, T, Fdim = x.shape
    # pytorch pide (B,F,T) para conv1d, asi que cambio orden
    x_bt_f_t = x.transpose(1, 2).contiguous()
    # kernel = ventanita llena de 1/k → promedio
    kernel = torch.ones(Fdim, 1, k, device=x.device) / k
    # reflejo bordes para que no meta ceros feos
    pad = (k - 1) // 2
    x_pad = F.pad(x_bt_f_t, (pad, pad), mode="reflect")
    # conv1d aplica el promedio movil
    suav = F.conv1d(x_pad, kernel, groups=Fdim)
    # vuelvo a (B,T,F) normal
    return suav.transpose(1, 2).contiguous()


class Autoformer(nn.Module):
    """
    idea: separar en tendencia + estacionalidad
    dos caminos (mlp para cada cosa)
    al final los sumo y ya
    """
    def __init__(self, entrada_dim, salida_dim=1, horizonte=24,
                 tam_oculto=128, k_promedio=25, dropout=0.10):
        super().__init__()
        self.horizonte = horizonte
        self.salida_dim = salida_dim
        self.k_promedio = k_promedio

        # embeddings del horizonte → uno para cada paso futuro
        self.h_emb = nn.Embedding(horizonte, tam_oculto)

        # mlp para tendencia
        self.mlp_tendencia = nn.Sequential(
            nn.Linear(entrada_dim + tam_oculto, 2 * tam_oculto),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * tam_oculto, salida_dim)
        )
        # mlp para estacionalidad
        self.mlp_estacional = nn.Sequential(
            nn.Linear(entrada_dim + tam_oculto, 2 * tam_oculto),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * tam_oculto, salida_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # saco tendencia con promedio movil
        tendencia = promedio_movil(x, self.k_promedio)
        # y estacionalidad = lo que sobra
        estacional = x - tendencia

        # me quedo solo con el último paso de cada cosa
        res_t = tendencia[:, -1, :]   # la tendencia final
        res_s = estacional[:, -1, :] # la estacionalidad final

        # genero embeddings para horizonte (0..H-1)
        B = x.size(0)
        idx = torch.arange(self.horizonte, device=x.device)
        eH = self.h_emb(idx).unsqueeze(0).expand(B, -1, -1)

        # junto resumen + embedding para pasarlo al mlp
        feat_t = torch.cat([res_t.unsqueeze(1).expand(-1, self.horizonte, -1), eH], dim=-1)
        feat_s = torch.cat([res_s.unsqueeze(1).expand(-1, self.horizonte, -1), eH], dim=-1)

        # mando cada cosa a su mlp
        y_t = self.mlp_tendencia(feat_t)     # salida tendencia
        y_s = self.mlp_estacional(feat_s)    # salida estacionalidad

        # sumo → predicción final
        return (y_t + y_s).view(B, self.horizonte, self.salida_dim)
