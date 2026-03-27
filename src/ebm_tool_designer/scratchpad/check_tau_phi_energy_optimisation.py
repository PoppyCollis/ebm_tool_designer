"""
    Here, I test that the reparameterisation trick works by intrucing a dummy energy function E = tau **2.
    When I optimise in phi space, the final tau parameteres are the lower bounds "a" of all of the uniform distrubitions.
    This is exactly what we would expect.
"""


from ebm_tool_designer.tool_design_prior import ToolDesignPrior
from ebm_tool_designer.config import ToolDatasetConfig
import torch

    
def main():
    prior = ToolDesignPrior(l1_bounds=ToolDatasetConfig.L1_BOUNDS, l2_bounds=ToolDatasetConfig.L2_BOUNDS, theta_bounds=ToolDatasetConfig.THETA_BOUNDS, device=ToolDatasetConfig.DEVICE)

    # 1. start with a sample in tau space
    tau = prior.sample(batch_size=2)
    print(tau)

    # 2. translate to phi space
    phi = prior.transform_to_phi(tau) # now we are tracking grads, phi is the leaf node
    print(phi)

    # 3. optimise in phi space
    optimizer = torch.optim.SGD([phi], lr=1e-2) # will likely need Adam for a more complex energy function than this dummy one

    for i in range(10):
        optimizer.zero_grad() 
        
        tau_current = prior.transform_to_tau(phi) # re-derive tau from phi inside the loop.

        energy = torch.sum(tau_current ** 2) # dummy energy function
        
        # backprop
        energy.backward()
        
        optimizer.step()


    print(tau_current)
    
if __name__ == "__main__":
    main()  

