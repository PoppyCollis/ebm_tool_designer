# Project Memory

## What This Project Is
Energy-Based Model (EBM) for robotic tool design. Given a task (target end-effector location), use Langevin Dynamics to sample optimal tool geometries (l1, l2, θ) from a learned energy landscape.

## Architecture
- **Reward model** (`reward_model.py`): MLP → f(l1, l2, sin θ, cos θ, x_target, y_target) → reward
- **EBM sampler** (`ebm.py`): Langevin Dynamics in reparameterized (phi) space with log-det Jacobian correction
- **Prior** (`tool_design_prior.py`): Uniform prior with logit/sigmoid reparameterization (tau ↔ phi)
- **Dataset** (`tool_dataset.py`): 50k synthetic tool designs, reward = saturated Euclidean distance to target

## Current State
- Reward model: trained, saved to `weights/reward_model_best.pt`, normalization stats saved with checkpoint
- Langevin sampler: implemented with noise + log-det correction; **not yet converging reliably**
- MH correction: TODO (placeholder in ebm.py line 77)
- Prior energy: hardcoded to 0 (placeholder, ebm.py line 63)

## Key Decisions
- sin/cos encoding for θ to avoid angle discontinuity
- Reparameterization trick (logit/sigmoid) required because uniform prior has zero gradient
- E_SCALING_FACTOR=1e-5, ETA=1e-4 — gradient magnitudes between energy and Jacobian terms are imbalanced (printed at sampling start)
- MSELoss used for reward model (L1Loss commented out as alternative)
