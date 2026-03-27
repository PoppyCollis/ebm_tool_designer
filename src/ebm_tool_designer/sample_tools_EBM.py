"""
Initialises EBM with a (pre-trained) reward prediction network and a tool design prior.
Outputs a batch of tools that are task-conditioned by langevin sampling from the EBM 
given a target location (c_target) and reward target (r_target is always 0 as we want 0 distance
between the end-effector and the target).
"""

import torch
from ebm_tool_designer.tool_design_prior import ToolDesignPrior
from ebm_tool_designer.config import ToolDatasetConfig, EBMConfig, RewardModelConfig
from ebm_tool_designer.helpers.plots import plot_energy_hist, visualise_tools 
from ebm_tool_designer.ebm import EnergyBasedModel


def main():
    
    # initailise EBM
    device = EBMConfig.DEVICE
    print(f"Device: {device}")
    prior = ToolDesignPrior(ToolDatasetConfig.L1_BOUNDS, ToolDatasetConfig.L2_BOUNDS, ToolDatasetConfig.THETA_BOUNDS, ToolDatasetConfig.DEVICE)
    ebm = EnergyBasedModel(prior, weights_path=RewardModelConfig.WEIGHTS_SAVE_PATH)
    
    # define a target location 
    n_samples = EBMConfig.N_SAMPLES # For example
    x = torch.tensor([-150.0], device=device)
    y = torch.tensor([450.0], device=device)
    single_c_target = torch.stack([x, y], dim=-1)
    c_target = single_c_target.expand(n_samples, 2) # expand it for batch of tools [n_samples, 2]
    
    # set a reward target of 0 
    single_r_target = torch.tensor([0.0], device=device) 
    print(f"Target location: {single_c_target.cpu().detach().numpy()}, Reward target: {single_r_target.item()}")
    r_target = single_r_target.expand(n_samples) # expand it for batch of tools [n_samples]

    # sample from EBM
    tool_sample, energy_hist = ebm.langevin_dynamics(c_target, r_target, batch_size=n_samples)
    plot_energy_hist(energy_hist)

    # visualise tools
    final_tool_energies = ebm.joint_energy(tool_sample, c_target, r_target).view(n_samples, -1).sum(dim=1)
    energy_per_tool = final_tool_energies.cpu().detach().numpy().flatten()
    designs = {}
    designs['l1'] = tool_sample[:, 0].cpu().detach().numpy()
    designs['l2'] = tool_sample[:, 1].cpu().detach().numpy()
    designs['theta'] = tool_sample[:, 2].cpu().detach().numpy()

    if n_samples < 26:
        visualise_tools(designs, target_location=single_c_target.cpu().detach().numpy(), energies = energy_per_tool)

if __name__ == "__main__":
    main()