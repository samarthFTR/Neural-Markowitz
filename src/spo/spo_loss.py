"""
SPO+ Loss — Smart Predict-then-Optimize Surrogate Loss.

Reference: Elmachtoub & Grigas (2021), "Smart Predict, then Optimize",
           Management Science 68(1):9–26.

The SPO+ loss is a convex, sub-differentiable upper bound on the true
decision regret.  It trains the prediction network to produce forecasts
whose downstream portfolio decisions have low regret, rather than simply
minimising prediction error.

Also provides a standard MSE baseline and a hybrid blended loss.
"""

import torch
import torch.nn as nn


class SPOPlusLoss(nn.Module):
    """
    L_SPO+(ĉ, c) = obj(w*(2ĉ − c), c) − obj(w*(ĉ), c)

    Where:
        obj(w, c) = −c^T w + γ w^T Σ w   (Markowitz objective)
        w*(v)     = argmin obj(w, v)       (optimal weights under v)

    Parameters
    ----------
    portfolio_layer : DifferentiableMarkowitz
        The differentiable optimisation layer.
    """

    def __init__(self, portfolio_layer):
        super().__init__()
        self.portfolio_layer = portfolio_layer

    def forward(self, predicted_returns, true_returns, covariance):
        """
        Parameters
        ----------
        predicted_returns : Tensor (batch, n_assets)
        true_returns      : Tensor (batch, n_assets)
        covariance        : Tensor (batch, n_assets, n_assets)

        Returns
        -------
        loss : scalar Tensor — mean SPO+ loss over the batch
        """
        # 1. Optimal weights under the prediction
        w_pred = self.portfolio_layer(predicted_returns, covariance)

        # 2. Optimal weights under the SPO+ surrogate cost vector
        surrogate_cost = 2.0 * predicted_returns - true_returns
        w_surrogate = self.portfolio_layer(surrogate_cost, covariance)

        # 3. Evaluate both decisions under the TRUE cost
        obj_surrogate = self.portfolio_layer.portfolio_objective(
            w_surrogate, true_returns, covariance
        )
        obj_pred = self.portfolio_layer.portfolio_objective(
            w_pred, true_returns, covariance
        )

        # SPO+ loss = E[ obj(w_surrogate) − obj(w_pred) ]
        loss = (obj_surrogate - obj_pred).mean()
        return loss


class DecisionRegretLoss(nn.Module):
    """
    True decision regret (non-differentiable — for evaluation only).

    L(ĉ, c) = obj(w*(ĉ), c) − obj(w*(c), c)
    """

    def __init__(self, portfolio_layer):
        super().__init__()
        self.portfolio_layer = portfolio_layer

    @torch.no_grad()
    def forward(self, predicted_returns, true_returns, covariance):
        w_pred = self.portfolio_layer(predicted_returns, covariance)
        w_oracle = self.portfolio_layer(true_returns, covariance)

        obj_pred = self.portfolio_layer.portfolio_objective(
            w_pred, true_returns, covariance
        )
        obj_oracle = self.portfolio_layer.portfolio_objective(
            w_oracle, true_returns, covariance
        )
        return (obj_pred - obj_oracle).mean()


class HybridLoss(nn.Module):
    """
    λ · SPO+  +  (1 − λ) · MSE

    Blends decision quality with prediction accuracy.  Setting λ=0
    recovers a pure MSE baseline; λ=1 is pure SPO+.
    """

    def __init__(self, portfolio_layer, lam: float = 0.5):
        super().__init__()
        self.spo_loss = SPOPlusLoss(portfolio_layer)
        self.mse_loss = nn.MSELoss()
        self.lam = lam

    def forward(self, predicted_returns, true_returns, covariance):
        l_spo = self.spo_loss(predicted_returns, true_returns, covariance)
        l_mse = self.mse_loss(predicted_returns, true_returns)
        return self.lam * l_spo + (1.0 - self.lam) * l_mse
