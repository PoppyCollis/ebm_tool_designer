# EBM Tool Designer

Task-conditioned robotic tool design using Energy-Based Models (EBMs). Given a target end-effector location, this package uses Langevin dynamics to sample optimal tool geometries from a learned energy landscape.

## Overview

A two-link planar tool is parameterised by three design variables:

| Parameter | Description | Default bounds |
|-----------|-------------|----------------|
| `l1` | Length of the first (vertical) link | [100, 300] |
| `l2` | Length of the second link | [100, 300] |
| `theta` | Angle of the second link relative to vertical | [0, 360] degrees |

The end-effector position is computed as:

```
x = l2 * sin(theta)
y = l1 + l2 * cos(theta)
```

The pipeline has three stages:

1. **Dataset generation** -- Synthetic (static) dataset of (tool parameters, target location, reward) tuples, where reward is based on the Euclidean distance between the tool's end-effector and the target.
2. **Reward model training** -- An MLP learns a mapping `f(l1, l2, sin(theta), cos(theta), x_target, y_target) -> reward`.
3. **EBM sampling** -- Langevin dynamics in a reparameterised (logit) space samples tool designs that minimise the energy (i.e. maximise the predicted reward for a given target location).

## Installation

Requires Python 3.8+ and PyTorch.

```bash
# Clone the repository
git clone <repo-url>
cd ebm_tool_designer

# Install in editable mode
pip install -e .
```

Additional dependencies not listed in `pyproject.toml` but required at runtime:

```bash
pip install torch pandas scikit-learn scipy tqdm pyarrow fastparquet
```

## Project Structure

```
src/ebm_tool_designer/
    config.py               # All hyperparameters and paths (ToolDatasetConfig, RewardModelConfig, EBMConfig)
    tool_dataset.py         # Dataset generation with rejection sampling for uniform end-effector coverage
    reward_model.py         # MLP architecture and energy function
    train_offline.py        # Reward model training loop
    tool_design_prior.py    # Uniform prior with logit/sigmoid reparameterisation (tau <-> phi)
    ebm.py                  # EnergyBasedModel class and Langevin dynamics sampler
    helpers/
        plots.py            # Visualisation utilities (tool diagrams, energy curves, heatmaps)
        utils.py            # General utilities
    data/                   # Generated datasets (parquet file format)
    weights/                # Saved model checkpoints

tests/
    sample_tools_EBM.py     # End-to-end example: initialise EBM, set target, sample tools, visualise
```

## Usage

### 1. Generate the dataset

```bash
python -m ebm_tool_designer.tool_dataset
```

This creates a parquet file at the path specified in `ToolDatasetConfig.SAVE_PATH` containing tool designs paired with target locations and rewards. By default it generates 10,000 samples with rejection sampling to ensure uniform end-effector coverage across the workspace.

Configuration in `config.py` -> `ToolDatasetConfig`:
- `NUM_DESIGNS` -- number of tool-target pairs
- `L1_BOUNDS`, `L2_BOUNDS`, `THETA_BOUNDS` -- parameter ranges
- `REWARD_TYPE` -- `"euclidean_distance"`, `"mse"`, `"gaussian_kernel"`, or `"saturated_euclidean_distance"`

### 2. Train the reward model

```bash
python -m ebm_tool_designer.train_offline
```

Trains the MLP reward predictor and saves the best checkpoint (including normalisation statistics) to `RewardModelConfig.WEIGHTS_SAVE_PATH`.

Configuration in `config.py` -> `RewardModelConfig`:
- `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE` -- training hyperparameters
- `SIGMA` -- standard deviation used in the energy function `E = (r_target - r_pred)^2 / (2 * sigma^2)`
- `HIDDEN_FEATURES`, `OUT_FEATURES` -- network architecture

### 3. Sample tool designs with the EBM

```bash
python -m tests.sample_tools_EBM
```

Or use the API directly:

```python
import torch
from ebm_tool_designer.tool_design_prior import ToolDesignPrior
from ebm_tool_designer.config import ToolDatasetConfig, EBMConfig, RewardModelConfig
from ebm_tool_designer.ebm import EnergyBasedModel

# Set up prior and EBM
device = EBMConfig.DEVICE
prior = ToolDesignPrior(
    ToolDatasetConfig.L1_BOUNDS,
    ToolDatasetConfig.L2_BOUNDS,
    ToolDatasetConfig.THETA_BOUNDS,
    device
)
ebm = EnergyBasedModel(prior, weights_path=RewardModelConfig.WEIGHTS_SAVE_PATH)

# Define target location and reward
n_samples = 9
x_target = torch.tensor([-150.0], device=device)
y_target = torch.tensor([450.0], device=device)
c_target = torch.stack([x_target, y_target], dim=-1).expand(n_samples, 2)
r_target = torch.tensor([0.0], device=device).expand(n_samples)  # 0 = perfect reach

# Sample tools via Langevin dynamics
tool_samples, energy_history = ebm.langevin_dynamics(c_target, r_target, batch_size=n_samples)
# tool_samples is a (n_samples, 3) tensor of [l1, l2, theta]
```

Configuration in `config.py` -> `EBMConfig`:
- `N_SAMPLES` -- number of tool designs to sample in parallel
- `N_SAMPLING_STEPS` -- Langevin dynamics iterations
- `ETA` -- step size
- `E_SCALING_FACTOR` -- scales the energy gradient relative to the log-det Jacobian correction

## How It Works

**Reparameterisation**: The uniform prior over `[a, b]` has zero gradient in the interior and infinite gradient at the boundaries, making direct gradient-based sampling impossible. Instead, tool parameters `tau` are mapped to an unconstrained space `phi` via:

```
phi = logit((tau - a) / (b - a))
```

Langevin dynamics runs in `phi`-space, with a log-determinant Jacobian correction to account for the change of variables. Samples are mapped back to `tau`-space via the sigmoid inverse.

**Energy function**: The joint energy combines a conditional term (how well the tool reaches the target) and a prior term:

```
E(tau, c, r) = E_scaling * E_cond(tau, c, r) - log|det J(phi -> tau)|
```

where `E_cond = (r_target - r_predicted)^2 / (2 * sigma^2)`.

## Visualisation

The package includes plotting utilities in `helpers/plots.py`:

- `visualise_tools` -- renders each sampled tool as a two-link arm with the target location
- `plot_energy_hist` -- energy vs. Langevin iteration
- `plot_density_heatmap` -- 2D density of end-effector positions
- `plot_prediction_error_heatmap` -- spatial distribution of reward model prediction error
- `plot_sample_predictions` -- paired end-effector / target positions with true vs. predicted reward
