import numpy as np
from ebm_tool_designer.reward_model import MLP
import torch
from ebm_tool_designer.tool_design_prior import ToolDesignPrior
from ebm_tool_designer.config import ToolDatasetConfig, EBMConfig, RewardModelConfig
from ebm_tool_designer.helpers.plots import plot_energy_hist, visualise_tools 
    
class EnergyBasedModel:
    def __init__(self, prior, weights_path):
        """
        Initializes the EBM with a prior and a reward model.
        
        Args:
            prior (ToolDesignPrior): The unconditioned prior over tool designs.
            weights_path (str): Path to the pre-trained reward model weights.
        """
        self.device = RewardModelConfig.DEVICE
        self.prior = prior
        self.reward_model, self.feature_stats, self.label_stats = self.load_reward_model(weights_path) # load in pretrained weights
        self.n_sampling_steps = EBMConfig.N_SAMPLING_STEPS
        self.eta = EBMConfig.ETA
        
    def load_reward_model(self, weights_path):
        # load in architecture (must match one used for pre-trained model weights)
        reward_model = MLP(
            in_features=RewardModelConfig.IN_FEATURES, 
            hidden_features=RewardModelConfig.HIDDEN_FEATURES, 
            out_features=RewardModelConfig.OUT_FEATURES) 
        
        # load the pre-trained weights into the model
        checkpoint = torch.load(weights_path, map_location=device)
        reward_model.load_state_dict(checkpoint['model_state_dict'])
        reward_model.eval()
        reward_model.to(device)
            
        # Keep stats on CPU for dataset normalization; move to device only for inference
        feature_stats = {k: v.to(device) for k, v in checkpoint['feature_stats'].items()}
        label_stats = {k: v.to(device) for k, v in checkpoint['label_stats'].items()}
        
        # also need to load in normalisation stats for features
        return reward_model, feature_stats, label_stats
    
    def joint_energy(self, tau, c_target, r_target):
        
        # slice tau to get component tool parameters
        # tau from uniform prior only gives us 3 parameters
        l1 = tau[:, 0:1]
        l2 = tau[:, 1:2]
        theta = tau[:, 2:3]
        
        # feature engineering
        sin = torch.sin(theta)
        cos = torch.cos(theta)
        l2sintheta = l2 * sin
        l2costheta = l2 * cos

        # concatenate into a single input vector for reward prediction network
        x = torch.cat([l1, l2, sin, cos, l2sintheta, l2costheta, c_target], dim=-1)
        
        # apply the same standardisation used during training
        x = (x - self.feature_stats['mean']) / self.feature_stats['std']
        r_target = (r_target - self.label_stats['mean']) / self.label_stats['std']

        cond_energy = self.reward_model.energy(x, r_target)
        
        prior_energy = 0 # placeholder
        
        return cond_energy + prior_energy
    
    def langevin_dynamics(self, c_target, r_target, batch_size):
            
        # 1. start with a sample in tau space
        tau = self.prior.sample(batch_size)
        #print(" initial tau:", tau.cpu().detach().numpy())

        # 2. translate to phi space
        # 2. translate to phi space
        phi = self.prior.transform_to_phi(tau).detach().requires_grad_(True) # now we are tracking grads, phi is the leaf node
        
        c_target = c_target.detach()
        r_target = r_target.detach()

        # 3. optimise in phi space 
        #optimizer = torch.optim.Adam([phi], lr=self.eta)
        noise_std = np.sqrt(2.0 * self.eta)
        
        energy_hist = []
        for i in range(self.n_sampling_steps):
            
            if phi.grad is not None:
                phi.grad.zero_()
                
            tau_current = prior.transform_to_tau(phi)
            
            sigmoid_phi = torch.sigmoid(phi)
            log_det_jacobian = torch.log((self.prior.bounds_high - self.prior.bounds_low) * sigmoid_phi * (1 - sigmoid_phi) + 1e-8).sum()
            energy = EBMConfig.E_SCALING_FACTOR * self.joint_energy(tau_current, c_target, r_target) - log_det_jacobian
            
            # Check the magnitude of the two forces
            if i ==0:
                grad_energy = torch.autograd.grad(self.joint_energy(tau_current, c_target, r_target).sum(), phi, retain_graph=True)[0]
                grad_jacobian = torch.autograd.grad(log_det_jacobian.sum(), phi, retain_graph=True)[0]

                print(f"Energy Grad Mean Abs: {grad_energy.abs().mean().item()}")
                print(f"Jacobian Grad Mean Abs: {grad_jacobian.abs().mean().item()}")
                            
                
            energy.sum().backward()
            
            with torch.no_grad():
                energy_hist.append(energy.sum().item())
                
                noise = torch.randn_like(phi) * noise_std
                phi -= (self.eta/2) * phi.grad + noise
            
        tau_final = tau_current
        
        #print("final tau:", tau_final.cpu().detach().numpy())
        
        return tau_final, energy_hist
        
device = EBMConfig.DEVICE

prior = ToolDesignPrior(ToolDatasetConfig.L1_BOUNDS, ToolDatasetConfig.L2_BOUNDS, ToolDatasetConfig.THETA_BOUNDS, ToolDatasetConfig.DEVICE)
        
ebm = EnergyBasedModel(prior, weights_path=RewardModelConfig.WEIGHTS_SAVE_PATH)


# sample a random target location and set a reward target
n_samples = EBMConfig.N_SAMPLES # For example

max_radius = prior.bounds_high[0] + prior.bounds_high[1]
theta = torch.rand(1, device=device) * np.pi # always positive y coordinate # torch.rand(1, device=device) * 2 * np.pi
r = max_radius * torch.sqrt(torch.rand(1, device=device))
x = r * torch.cos(theta)
y = r * torch.sin(theta)

# Combine into a single tensor of shape (n_samples, 2)

# short cut
#single_c_target = torch.stack([x, y], dim=-1)
x = torch.tensor([-150.0], device=device)
y = torch.tensor([450.0], device=device)
single_c_target = torch.stack([x, y], dim=-1)

single_r_target = torch.tensor([0.0], device=device) # whats an appropriate reward?

print(f"Target location: {single_c_target.cpu().detach().numpy()}, Reward target: {single_r_target.item()}")

# Expand it to [n_samples, 2]
c_target = single_c_target.expand(n_samples, 2)
r_target = single_r_target.expand(n_samples)

tool_sample, energy_hist = ebm.langevin_dynamics(c_target, r_target, batch_size=n_samples)
# tool_sample = tool_sample.cpu().detach().numpy()

# print("tools:", tool_sample)


# get final energies (dont sum)
final_tool_energies = ebm.joint_energy(tool_sample, c_target, r_target).view(n_samples, -1).sum(dim=1)

designs = {}
designs['l1'] = tool_sample[:, 0].cpu().detach().numpy()
designs['l2'] = tool_sample[:, 1].cpu().detach().numpy()
designs['theta'] = tool_sample[:, 2].cpu().detach().numpy()


plot_energy_hist(energy_hist)

energy_per_tool = final_tool_energies.cpu().detach().numpy().flatten()

#print("energies", energy_per_tool)
#print("tools", tool_sample.cpu().detach().numpy())

if n_samples < 26:
    visualise_tools(designs, target_location=single_c_target.cpu().detach().numpy(), energies = energy_per_tool)

# I want to plot the energies of each tool alongside 

# todo:
# make it work for batch
# add log det of jacobian to overleaf
# add MH correction?
# optimise lr and iter size, check energy is decreasing and converging
# check predicted reward of final tool is close to reward 

# plot the tool sample and visualise w.r.t. target location and show reward!
