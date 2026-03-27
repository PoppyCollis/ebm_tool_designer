import numpy as np
import torch
import matplotlib.pyplot as plt
from helpers.plots import plot_end_effector_and_target_distribution

l_min, l_max = 100,300

center = (0.0, (l_max - l_min))
r_inner = l_min
r_outer = l_min+l_max 

c_inner = 2 * np.pi * r_inner
c_outer = 2 * np.pi * r_outer

n_samples = 10000

def sample_points_in_annulus(n_samples):
    # Sample random angles uniformly between 0 and 2*pi
    angles = np.random.uniform(0, 2 * np.pi, n_samples)
    
    # Sample random radii with uniform distribution in the annulus area
    radii = np.sqrt(np.random.uniform(r_inner**2, r_outer**2, n_samples))
    
    # Convert polar coordinates to Cartesian coordinates
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)
    
    return np.stack([x, y], axis=-1)    


def sample_target_in_bounded_box(n_samples):
    min_y = center[1] - (l_min+l_max)
    max_y = center[1] + (l_min+l_max)
    min_x = center[0] - (l_min+l_max)
    max_x = center[0] + (l_min+l_max)
    x = np.random.uniform(min_x, max_x, n_samples)
    y = np.random.uniform(min_y, max_y, n_samples)
    return np.stack([x, y], axis=-1)

    

points = sample_points_in_annulus(n_samples)
targets = sample_target_in_bounded_box(n_samples)

plot_end_effector_and_target_distribution(points, targets)
