"""
Cross-Sectional Return Prediction Network.

A stock-level MLP that predicts expected returns for each asset in the
universe.  Weights are shared across stocks (cross-sectional scoring
function), so the network generalises across the universe rather than
memorising stock-specific behaviour.
"""

import torch
import torch.nn as nn


class ReturnPredictionNet(nn.Module):
    """
    Input  : (batch, n_stocks, n_features)
    Output : (batch, n_stocks)  — predicted return per stock

    The same MLP is applied identically to every stock (weight sharing).

    Parameters
    ----------
    n_features : int
        Number of input features per stock.
    hidden_dims : list[int]
        Width of each hidden layer.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        n_features: int,
        hidden_dims: list = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers = []
        in_dim = n_features
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))

        self.mlp = nn.Sequential(*layers)

        # Xavier init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (batch, n_stocks, n_features)

        Returns
        -------
        Tensor (batch, n_stocks) — predicted return per stock
        """
        B, N, F = x.shape
        x_flat = x.reshape(B * N, F)           # (B*N, F)
        out = self.mlp(x_flat)                  # (B*N, 1)
        return out.reshape(B, N)                # (B, N)
