"""
Neural network reward prediction model f(τ, c) → R; takes in tool design parameters τ 
and task description c and outputs a scalar reward
"""

import torch.nn as nn
from ebm_tool_designer.config import RewardModelConfig

class MLP(nn.Module):
    def __init__(self, in_features=8, hidden_features=128, out_features=64):
        
        # Use super __init__ to inherit from parent nn.Module class
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LeakyReLU(0.1), # Allows gradients even if the neuron is "off"
            nn.Linear(hidden_features, hidden_features),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_features, hidden_features),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_features, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1) # NO activation here!
        )
        # Apply the weight initialization
        self.apply(self._init_weights)
        self.sigma = RewardModelConfig.SIGMA
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Kaiming Uniform is ideal for ReLU activations
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        
    def forward(self, x):
        return self.net(x)
    
    def energy(self,x, r_target):
        """
        Computes the conditional energy for a specific target reward value and input x
        input,x, includes tool design params tau and task description c.
        """
        r_pred = self.forward(x)
        E = 1/(2* (self.sigma**2)) * (r_target - r_pred)**2
        return E
    
