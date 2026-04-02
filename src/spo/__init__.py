# SPO — Smart Predict-then-Optimize Layer
# End-to-end decision-focused learning for portfolio optimization.
#
# Modules:
#   prediction_net   — Neural network for cross-sectional return prediction
#   portfolio_layer  — Differentiable Markowitz optimizer (cvxpylayers)
#   spo_loss         — SPO+ surrogate loss (Elmachtoub & Grigas, 2021)
#   covariance       — Rolling covariance estimation with shrinkage
#   trainer          — End-to-end training loop
#   evaluation       — Portfolio-level backtesting & metrics
