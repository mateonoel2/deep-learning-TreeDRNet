import math
import torch
import torch.nn as nn


# Positional Encoding sinusoidal (el clásico de transformers pero robusto)
# guardo una tabla con senos/cosenos ya calculados (max_len, d)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # posiciones 0..max_len
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)  # senos en las pares
        pe[:, 1::2] = torch.cos(pos * div)  # cosenos en las impares
        self.register_buffer("pe_base", pe)  # lo guardo como buffer fijo

    def forward(self, x):
        # x: (B, L, d)  batch primero
        B, L, d = x.shape
        # chequeos de seguridad para no pasarme
        assert d == self.pe_base.size(1), f"d_model={d} != pe_dim={self.pe_base.size(1)}"
        assert L <= self.pe_base.size(0), f"L={L} > max_len={self.pe_base.size(0)}"
        pe = self.pe_base[:L, :]  # recorto al largo de la secuencia
        pe = pe.unsqueeze(0)  # (1, L, d)
        # a veces pasa que queda con ejes cruzados, me cubro
        if pe.size(1) != L and pe.size(2) == L:
            pe = pe.transpose(1, 2)  # acomodo (1,L,d)
        return x + pe  # sumo posiciones a los embeddings


class ModeloTransformer(nn.Module):
    def __init__(self, entrada_dim, salida_dim=1, horizonte=24,
                 d_modelo=128, num_cabezas=4, num_capas=2, dim_ff=256, dropout=0.10):
        super().__init__()
        self.horizonte = horizonte
        self.salida_dim = salida_dim

        # proyecto los features de entrada al tamaño del modelo
        self.proy_entrada = nn.Linear(entrada_dim, d_modelo)
        # positional encoding (la versión de arriba)
        self.pos = PositionalEncoding(d_modelo)

        # capa base del encoder transformer (usa multihead self-attn + ff)
        capa = nn.TransformerEncoderLayer(d_model=d_modelo, nhead=num_cabezas,
                                          dim_feedforward=dim_ff, dropout=dropout,
                                          batch_first=True)
        self.encoder = nn.TransformerEncoder(capa, num_layers=num_capas)

        # query especial (parametro entrenable) para hacer pooling de la secuencia
        self.query_pool = nn.Parameter(torch.randn(1, 1, d_modelo))
        self.attn_pool = nn.MultiheadAttention(d_modelo, num_heads=num_cabezas, batch_first=True)

        # embedding del horizonte (uno para cada paso futuro)
        self.h_emb = nn.Embedding(horizonte, d_modelo)

        # cabecera final: junta contexto + horizonte y predice
        self.head = nn.Sequential(
            nn.Linear(2 * d_modelo, d_modelo),
            nn.ReLU(),
            nn.Linear(d_modelo, salida_dim)
        )

    def forward(self, x):
        z = self.proy_entrada(x)  # paso de features a dimensión del modelo
        z = self.pos(z)  # le sumo posiciones
        z = self.encoder(z)  # paso por las capas transformer

        # pooling con query especial entrenable
        q = self.query_pool.expand(z.size(0), -1, -1)
        z_pool, _ = self.attn_pool(q, z, z)  # atención query sobre toda la secuencia
        z_pool = z_pool[:, 0, :]  # me quedo con el vector de salida

        # ahora preparo el horizonte con sus embeddings
        B = z_pool.size(0)
        idx = torch.arange(self.horizonte, device=z_pool.device)
        eH = self.h_emb(idx)  # (H,d)

        # repito el contexto para cada paso futuro y lo concateno con su embedding
        ctx = z_pool.unsqueeze(1).expand(B, self.horizonte, -1)
        feat = torch.cat([ctx, eH.unsqueeze(0).expand(B, -1, -1)], dim=-1)

        # paso por la cabeza y listo → salida (B,H,1)
        return self.head(feat)
