from __future__ import annotations
import torch
import torch.nn as nn
from typing import Tuple, List


class DResBlock(nn.Module):
    def __init__(self, dim_in: int, hidden: int, horizonte: int, mlp_depth: int = 2, dropout: float = 0.0):
        super().__init__()
        mlp: List[nn.Module] = []
        d = dim_in
        for _ in range(mlp_depth):
            mlp += [nn.Linear(d, hidden), nn.ReLU()]
            if dropout > 0:
                mlp += [nn.Dropout(dropout)]
            d = hidden
        self.mlp = nn.Sequential(*mlp)
        self.backcast = nn.Linear(hidden, dim_in)
        self.forecast = nn.Linear(hidden, horizonte)

    def forward(self, x_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.mlp(x_flat)
        bc = self.backcast(h)
        fc = self.forecast(h)
        return bc, fc


class GatedBranch(nn.Module):
    def __init__(
        self,
        dim_in: int,
        hidden_gate: int,
        hidden_core: int,
        horizonte: int,
        mlp_depth: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim_in, hidden_gate), nn.ReLU(), nn.Linear(hidden_gate, dim_in), nn.Sigmoid()
        )
        self.core = DResBlock(dim_in, hidden_core, horizonte, mlp_depth=mlp_depth, dropout=dropout)

    def forward(self, x_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        g = self.gate(x_flat)
        x_sel = x_flat * g
        bc, fc = self.core(x_sel)
        return bc, fc


class MultiBranchBlock(nn.Module):
    def __init__(
        self,
        dim_in: int,
        horizonte: int,
        num_ramas: int,
        hidden_gate: int,
        hidden_core: int,
        mlp_depth: int,
        dropout: float,
    ):
        super().__init__()
        self.num_ramas = num_ramas
        self.branches = nn.ModuleList(
            [
                GatedBranch(dim_in, hidden_gate, hidden_core, horizonte, mlp_depth=mlp_depth, dropout=dropout)
                for _ in range(num_ramas)
            ]
        )

    def forward(self, x_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bcs, fcs = [], []
        for br in self.branches:
            bc, fc = br(x_flat)
            bcs.append(bc)
            fcs.append(fc)
        bc = torch.stack(bcs, dim=0).mean(dim=0)
        fc = torch.stack(fcs, dim=0).mean(dim=0)
        return bc, fc

    def forward_all_branches(self, x_flat: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        bcs, fcs = [], []
        for br in self.branches:
            bc, fc = br(x_flat)
            bcs.append(bc)
            fcs.append(fc)
        return bcs, fcs


class TreeDRNet(nn.Module):
    def __init__(
        self,
        entrada_dim: int,
        salida_dim: int,
        horizonte: int,
        long_ventana: int,
        profundidad_arbol: int = 2,
        num_ramas: int = 2,
        hidden_gate: int = 128,
        hidden_core: int = 128,
        mlp_depth: int = 2,
        dropout: float = 0.0,
        usar_conv_covariate: bool = True,
    ):
        super().__init__()
        self.horizonte = horizonte
        self.long_ventana = long_ventana
        self.entrada_dim = entrada_dim
        self.flatten_dim = long_ventana * entrada_dim
        self.usar_conv = usar_conv_covariate
        if self.usar_conv:
            self.conv1x1 = nn.Conv1d(in_channels=entrada_dim, out_channels=entrada_dim, kernel_size=1)
        self.block = MultiBranchBlock(
            dim_in=self.flatten_dim,
            horizonte=horizonte,
            num_ramas=num_ramas,
            hidden_gate=hidden_gate,
            hidden_core=hidden_core,
            mlp_depth=mlp_depth,
            dropout=dropout,
        )
        self.profundidad = profundidad_arbol
        self.out_proj = nn.Identity() if salida_dim == 1 else nn.Linear(salida_dim, salida_dim)

    def _prep_x(self, x: torch.Tensor) -> torch.Tensor:
        if self.usar_conv:
            x = x.transpose(1, 2)
            x = self.conv1x1(x)
            x = x.transpose(1, 2)
        x = x.reshape(x.size(0), -1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self._prep_x(x)
        total_forecast = 0.0
        nodos = [x0]
        for _ in range(self.profundidad):
            fcs = []
            nuevos_nodos = []
            for nodo in nodos:
                bcs, fcs_branch = self.block.forward_all_branches(nodo)
                fcs.extend(fcs_branch)
                for bc in bcs:
                    nuevos_nodos.append(nodo - bc)
            fci = torch.stack(fcs, dim=0).mean(dim=0)
            total_forecast = total_forecast + fci
            nodos = nuevos_nodos
        y = total_forecast.unsqueeze(-1)
        return y
