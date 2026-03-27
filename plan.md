# Plan

## Immediate Priorities

- [ ] **Fix EBM sampling convergence** — energy should decrease over Langevin steps
  - Tune ETA and E_SCALING_FACTOR
  - Investigate gradient magnitude imbalance (energy vs Jacobian terms)
  - Consider switching to Unadjusted Langevin (simpler) before adding MH correction

- [ ] **Implement prior energy term** — currently hardcoded to 0 in `ebm.py:63`
  - Uniform prior energy = -log(1/(b-a)) per dimension (constant, but should be included for correctness)

- [ ] **MALA acceptance/rejection step** — TODO in `ebm.py:77`
  - Optional, but needed for asymptotically correct sampling

## Medium-term

- [ ] **Refactor**: Move reward model training into `train_reward_model.py` (separate from `reward_model.py`)
- [ ] **Validate sampled tools**: Verify predicted reward of sampled designs matches target reward; visualize tool + target + end-effector

## Longer-term

- [ ] **Integrate with Matteo's dynamic-planning codebase** (offline dataset collection pathway)
