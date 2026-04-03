"""
Differentiable Markowitz Mean-Variance Optimizer.

Since cvxpylayers does not build on Python 3.13, this module implements
the differentiable optimization layer from scratch using:

  Forward pass  — CVXPY solves the QP
  Backward pass — Implicit differentiation through KKT conditions
                  (torch.autograd.Function)

This is equivalent to what cvxpylayers does internally, but avoids the
diffcp C++ dependency.
"""

import numpy as np
import cvxpy as cp
import torch
import torch.nn as nn
from utils.logger import logging


# =====================================================================
#  Low-level: CVXPY solver wrapped as a torch autograd function
# =====================================================================

class _MarkowitzSolve(torch.autograd.Function):
    """
    Custom autograd function that solves a Markowitz QP in the forward
    pass and differentiates through its KKT conditions in the backward
    pass.

    The QP:
        min  −μ^T w  +  γ w^T Σ w
        s.t. 1^T w = 1,   w ≥ 0,   w ≤ w_max

    KKT system (simplified for the backward pass):
        ∂L/∂w = −μ + 2γΣw − λ_eq·1 − λ_lb + λ_ub = 0
        complementarity on bounds

    We use implicit differentiation:  dw/dμ from the active-set KKT.
    """

    @staticmethod
    def forward(ctx, mu_tensor, sigma_tensor, gamma, max_weight):
        """
        Parameters
        ----------
        mu_tensor    : (n,)  predicted returns
        sigma_tensor : (n, n) covariance matrix
        gamma        : float, risk aversion
        max_weight   : float, max allocation per asset

        Returns
        -------
        w_tensor : (n,) optimal weights
        """
        n = mu_tensor.shape[0]
        mu_np = mu_tensor.detach().cpu().numpy().astype(np.float64)
        sigma_np = sigma_tensor.detach().cpu().numpy().astype(np.float64)

        # ── Solve with CVXPY ──
        w = cp.Variable(n)
        mu_param = mu_np
        sigma_param = sigma_np

        objective = cp.Minimize(
            -mu_param @ w + gamma * cp.quad_form(w, sigma_param)
        )
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            w <= max_weight,
        ]
        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.SCS, max_iters=10000, eps=1e-7)
            if prob.status not in ("optimal", "optimal_inaccurate"):
                raise cp.SolverError(f"Solver status: {prob.status}")
            w_np = w.value.astype(np.float64)
        except Exception:
            # Fallback: equal weight
            w_np = np.full(n, 1.0 / n, dtype=np.float64)

        w_tensor = torch.tensor(w_np, dtype=mu_tensor.dtype,
                                device=mu_tensor.device)

        # Save for backward
        ctx.save_for_backward(mu_tensor, sigma_tensor, w_tensor)
        ctx.gamma = gamma
        ctx.max_weight = max_weight

        return w_tensor

    @staticmethod
    def backward(ctx, grad_output):
        """
        Implicit differentiation through the KKT conditions.

        We identify the active set and solve the linear system:
            M · dw = d_mu
        where M = 2γΣ restricted to the free (active) variables,
        augmented with the equality constraint row.
        """
        mu, sigma, w_opt = ctx.saved_tensors
        gamma = ctx.gamma
        max_weight = ctx.max_weight
        n = w_opt.shape[0]

        EPS = 1e-6
        w_np = w_opt.detach().cpu().numpy()

        # Identify free variables (not on a bound)
        free_mask = (w_np > EPS) & (w_np < max_weight - EPS)
        free_idx = np.where(free_mask)[0]
        n_free = len(free_idx)

        if n_free == 0:
            # All variables on bounds → gradient is zero
            return torch.zeros_like(mu), None, None, None

        # Build the KKT submatrix for the free variables:
        #   [ 2γ Σ_ff   1 ] [ dw_f  ]   [ dμ_f ]
        #   [ 1^T       0 ] [ dλ    ] = [ 0    ]
        sigma_np = sigma.detach().cpu().numpy().astype(np.float64)
        S_ff = 2.0 * gamma * sigma_np[np.ix_(free_idx, free_idx)]

        # Augmented system (n_free+1) x (n_free+1)
        M = np.zeros((n_free + 1, n_free + 1), dtype=np.float64)
        M[:n_free, :n_free] = S_ff
        M[:n_free, n_free] = 1.0    # equality constraint column
        M[n_free, :n_free] = 1.0    # equality constraint row
        # M[n_free, n_free] = 0 (already zero)

        # We want  dw/dμ.  For each perturbation dμ_f:
        #   M [dw_f; dλ] = [dμ_f; 0]  →  dw_f = M_inv[:n_free, :n_free] · dμ_f
        # But we need grad_mu = grad_output · (dw/dμ),  i.e. the transpose:
        #   grad_mu_f = (dw/dμ)^T · grad_w_f

        # Solve M^T · z = [grad_w_f; 0] to get grad_mu_f = z[:n_free]
        grad_w_np = grad_output.detach().cpu().numpy().astype(np.float64)

        rhs = np.zeros(n_free + 1, dtype=np.float64)
        rhs[:n_free] = grad_w_np[free_idx]

        try:
            z = np.linalg.solve(M.T, rhs)
        except np.linalg.LinAlgError:
            # Singular — fall back to pseudo-inverse
            z = np.linalg.lstsq(M.T, rhs, rcond=None)[0]

        # Assemble full gradient w.r.t. μ
        grad_mu = np.zeros(n, dtype=np.float64)
        grad_mu[free_idx] = z[:n_free]

        grad_mu_tensor = torch.tensor(grad_mu, dtype=mu.dtype, device=mu.device)

        # No gradient w.r.t. sigma, gamma, max_weight
        return grad_mu_tensor, None, None, None


# =====================================================================
#  High-level nn.Module
# =====================================================================

class DifferentiableMarkowitz(nn.Module):
    """
    Differentiable Markowitz mean-variance optimizer.

    Forward:  (predicted_returns, covariance) → optimal portfolio weights
    Backward: Implicit differentiation through KKT conditions

    Parameters
    ----------
    n_assets : int
        Number of assets in the universe.
    gamma : float
        Risk-aversion parameter.  Higher → more conservative.
    max_weight : float
        Maximum weight per asset (concentration limit).
    """

    def __init__(
        self,
        n_assets: int,
        gamma: float = 0.5,
        max_weight: float = 0.10,
    ):
        super().__init__()
        self.n_assets = n_assets
        self.gamma = gamma
        self.max_weight = max_weight

        logging.info(
            f"DifferentiableMarkowitz: n_assets={n_assets}, γ={gamma}, "
            f"max_weight={max_weight} [custom autograd — no cvxpylayers]"
        )

    def forward(self, predicted_returns, covariance):
        """
        Parameters
        ----------
        predicted_returns : Tensor (batch, n_assets) or (n_assets,)
        covariance        : Tensor (batch, n_assets, n_assets) or (n_assets, n_assets)

        Returns
        -------
        weights : Tensor, same leading dims as input
        """
        squeeze = False
        if predicted_returns.dim() == 1:
            predicted_returns = predicted_returns.unsqueeze(0)
            covariance = covariance.unsqueeze(0)
            squeeze = True

        batch_size = predicted_returns.shape[0]
        all_weights = []

        for i in range(batch_size):
            w_i = _MarkowitzSolve.apply(
                predicted_returns[i],
                covariance[i],
                self.gamma,
                self.max_weight,
            )
            all_weights.append(w_i)

        weights = torch.stack(all_weights, dim=0)
        if squeeze:
            weights = weights.squeeze(0)

        return weights

    def portfolio_objective(self, weights, true_returns, covariance):
        """
        Evaluate the Markowitz objective under TRUE returns.

        Returns scalar (or batch of scalars):
            -μ^T w  +  γ w^T Σ w
        """
        port_return = (weights * true_returns).sum(dim=-1)
        risk = torch.einsum("...i,...ij,...j->...", weights, covariance, weights)
        return -port_return + self.gamma * risk
