import torch

l1_bounds = (150.0,300.0)
l2_bounds = (150.0,300.0)
theta_bounds = (0.0,360.0)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

bounds_low = torch.tensor([l1_bounds[0], l2_bounds[0], theta_bounds[0]], requires_grad=True).to(device)
bounds_high = torch.tensor([l1_bounds[1], l2_bounds[1], theta_bounds[1]], requires_grad=True).to(device)

def transform_to_phi(tau):
    # tau must be a tensor, with  requires_grad=True
    # Step 1: Normalize to [0, 1]
    u = (tau - bounds_low) / (bounds_high - bounds_low)
    # add a tiny epsilon to prevent log(0)
    eps = 1e-6
    u = torch.clamp(u, eps, 1.0 - eps)
    # transform to between (-\infty, + \infty)
    phi = torch.logit(u)
    return phi
    
def transform_to_tau(phi):
    # phi must be a tensor, with  requires_grad=True
    tau = bounds_low + (bounds_high - bounds_low) * torch.sigmoid(phi)
    return tau



epsilon = torch.rand(3, device=device)

tau = bounds_low + (bounds_high - bounds_low) * epsilon

print(f"Sampled Vector Tau: {tau}")

phi = transform_to_phi(tau)

print(f"Sampled Vector Phi: {phi}")

tau_2 = transform_to_tau(phi)

print(f"Reconstructed Vector Tau: {tau_2}")


# so my sampling process in my tool design prior should match this

# epsilon = torch.rand(3, device=device)

# tau = bounds_low + (bounds_high - bounds_low) * epsilon


# # 1. Define your bounds as tensors with gradients enabled
# a = torch.tensor([1.0], requires_grad=True)
# b = torch.tensor([10.0], requires_grad=True)

# # 2. Use the distributions module
# # Note: We use the rsample() method if available, 
# # but for Uniform, we often manually reparameterize.
# dist = torch.distributions.Uniform(a, b)

# # 3. Sample using the reparameterization trick
# # In PyTorch, Uniform does not support rsample() directly because 
# # the boundaries are not usually differentiable in the standard sense.
# # So, we do it manually:
# epsilon = torch.rand(1) # Sample from U(0,1)
# tau = a + (b - a) * epsilon

# # 4. Define a dummy loss (e.g., we want to minimize tau)
# loss = tau.pow(2)

# # 5. Backpropagate
# loss.backward()

# print(f"Sampled tau: {tau.item()}")
# print(f"Gradient w.r.t. a: {a.grad.item()}")
# print(f"Gradient w.r.t. b: {b.grad.item()}")



# a, b = 150.0, 300.0
# tau = torch.tensor([215]) # We want to start at 7.5

# # Step 1: Normalize to [0, 1]
# u = (tau - a) / (b - a)

# # Step 2: Transform to unconstrained space
# phi = torch.logit(u) 

# print("tau:", tau.item())
# print(f"The unconstrained parameter phi is: {phi.item():.4f}")
# # Output will be 1.0986, because sigmoid(1.0986) is 0.75


# # 1. Setup our bounds
# a = torch.tensor(1.0)
# b = torch.tensor(10.0)

# # 2. Start with an unconstrained phi (e.g., from an optimizer)
# phi = torch.tensor([2.5], requires_grad=True)

# # 3. Convert back to tau space
# # This is the "Forward" pass
# tau = a + (b - a) * torch.sigmoid(phi)

# print(f"Phi (unconstrained): {phi.item():.4f}")
# print(f"Tau (bounded [0, 10]): {tau.item():.4f}")

# # 4. Check the gradient
# # Even if phi is huge, the gradient of tau w.r.t phi 
# # will smoothly approach zero, never infinity.
# tau.backward()
# print(f"Gradient d_tau/d_phi: {phi.grad.item():.4f}")