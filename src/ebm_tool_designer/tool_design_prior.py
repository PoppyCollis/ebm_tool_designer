"""
    We can extend towards more expressive and complex priors in the future (e.g. diffusion prior).
"""
import torch

class ToolDesignPrior:
    """
    Unconditioned uniform prior distributions over tool parameters. Sampling from this prior will give you random tools.
    Includes logic to map into an optimisable space phi.
    """
    def __init__(self, l1_bounds, l2_bounds, theta_bounds, device):
        self.device = device
        self.bounds_low = torch.tensor([l1_bounds[0], l2_bounds[0], theta_bounds[0]], requires_grad=True).to(device)
        self.bounds_high = torch.tensor([l1_bounds[1], l2_bounds[1], theta_bounds[1]], requires_grad=True).to(device)
        
        
    def sample(self, batch_size=1):
        # sample in a way that will allow pytorch to track gradients
        n_params = 3 # l1,l2, theta
        
        epsilon = torch.rand(batch_size, n_params, device=self.device) # batch sample
        tau = self.bounds_low + (self.bounds_high - self.bounds_low) * epsilon
        
        tau.requires_grad_(True)
        return tau
    
    def transform_to_phi(self, tau):
        """
        Reparametrisation trick for tau into an unconstrained space phi by normalising and applying the logit func.
        This maps into a space between -infty, +infty such that we can do gradient update without the constraint 
        of hard bounds [a,b] of a uniform distribution, which causes problems with infinite gradients at the bounds.
        """
       # tau must be a tensor
        # Step 1: Normalize to [0, 1]
        u = (tau - self.bounds_low) / (self.bounds_high - self.bounds_low)
        # add a tiny epsilon to prevent log(0)
        eps = 1e-8
        u = torch.clamp(u, eps, 1.0 - eps)
        # transform to between (-\infty, + \infty)
        phi = torch.logit(u).detach().requires_grad_(True)
        return phi
        
    def transform_to_tau(self, phi):
        # phi must be a tensor, with  requires_grad=True
        tau = self.bounds_low + (self.bounds_high - self.bounds_low) * torch.sigmoid(phi)
        # do I need to return the log determinant here for MALA later?
        return tau

    
    