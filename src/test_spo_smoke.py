"""
Smoke test: Verify the full SPO forward + backward pass works.
Run from src/:  python test_spo_smoke.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

from spo.prediction_net import ReturnPredictionNet
from spo.portfolio_layer import DifferentiableMarkowitz
from spo.spo_loss import SPOPlusLoss, DecisionRegretLoss

# ── Config ──
N_ASSETS   = 10
N_FEATURES = 8
BATCH      = 1
GAMMA      = 0.5
MAX_WEIGHT = 0.20

print("="*50)
print("  SPO SMOKE TEST")
print("="*50)

# ── 1. Build components ──
pred_net   = ReturnPredictionNet(n_features=N_FEATURES, hidden_dims=[16, 8])
port_layer = DifferentiableMarkowitz(n_assets=N_ASSETS, gamma=GAMMA, max_weight=MAX_WEIGHT)
spo_loss   = SPOPlusLoss(port_layer)
regret     = DecisionRegretLoss(port_layer)

# ── 2. Synthetic data ──
torch.manual_seed(42)
features     = torch.randn(BATCH, N_ASSETS, N_FEATURES)
true_returns = torch.randn(BATCH, N_ASSETS) * 0.01

# Make a valid covariance matrix (positive definite)
A = torch.randn(N_ASSETS, N_ASSETS) * 0.01
cov = (A @ A.T + 0.001 * torch.eye(N_ASSETS)).unsqueeze(0)  # (1, N, N)

# ── 3. Forward pass ──
print("\n[1] Prediction network forward pass...")
predicted = pred_net(features)
print(f"    Input shape:  {features.shape}")
print(f"    Output shape: {predicted.shape}")
print(f"    Predicted returns (sample): {predicted[0, :5].detach().numpy()}")

print("\n[2] Markowitz layer forward pass...")
weights = port_layer(predicted.detach(), cov)
print(f"    Weights shape: {weights.shape}")
print(f"    Weights sum:   {weights.sum(dim=-1).item():.6f}")
print(f"    Weights min:   {weights.min().item():.6f}")
print(f"    Weights max:   {weights.max().item():.6f}")
print(f"    Weights: {weights[0].detach().numpy()}")

# ── 4. SPO+ Loss with backward pass ──
print("\n[3] SPO+ loss forward + backward...")
loss = spo_loss(predicted, true_returns, cov)
print(f"    SPO+ loss: {loss.item():.8f}")

loss.backward()

# Check gradients
has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
               for p in pred_net.parameters())
print(f"    Gradients flow to pred_net: {has_grad}")

if has_grad:
    grad_norms = [p.grad.norm().item() for p in pred_net.parameters() if p.grad is not None]
    print(f"    Gradient norms: {[f'{g:.6f}' for g in grad_norms[:4]]}...")

# ── 5. Decision regret (eval metric) ──
print("\n[4] Decision regret (eval only)...")
with torch.no_grad():
    dr = regret(predicted, true_returns, cov)
print(f"    Decision regret: {dr.item():.8f}")

# ── 6. Optimizer step ──
print("\n[5] Optimizer step test...")
optimizer = torch.optim.Adam(pred_net.parameters(), lr=1e-3)
for step in range(3):
    optimizer.zero_grad()
    pred = pred_net(features)
    l = spo_loss(pred, true_returns, cov)
    l.backward()
    optimizer.step()
    print(f"    Step {step+1}: loss = {l.item():.8f}")

print("\n" + "="*50)
print("  ALL TESTS PASSED ✓")
print("="*50)
