"""
Constructs an offline static dataset of tool parameters (where end-effector positions are sampled uniformly
rather than uniform sampling of each tool parameter independently). Target locations sampled uniformly and 
dataset label (r) is the euclidean distance between tool end-effector and target.
"""
import numpy as np
import pandas as pd
from ebm_tool_designer.helpers.plots import *
import torch
from torch.utils.data import Dataset
from ebm_tool_designer.config import ToolDatasetConfig

    
    
class ToolDataset:
    def __init__(self, l1_range, l2_range, theta_range, reward_type = "mse"):
        """
        l1_range and l2_range: tuples (min_length, max_length)
        theta_range: tuple (min_deg, max_deg)
        """
        self.l1_bounds = l1_range
        self.l2_bounds = l2_range
        self.theta_bounds = (np.radians(theta_range[0]), np.radians(theta_range[1]))
        self.reward_type = reward_type
        self.density_2d, self.x_edges, self.y_edges, self.max_inv_density = self.find_2D_density(n_samples=10000, bins=50) # precompute the 2D density for rejection sampling


    def sample_design_uniform_prior(self, n_samples=1):
        """
        Samples tools using independent Uniform priors over tool parameters.
        """
        l1 = np.random.uniform(self.l1_bounds[0], self.l1_bounds[1], n_samples)
        l2 = np.random.uniform(self.l2_bounds[0], self.l2_bounds[1], n_samples)
        theta = np.random.uniform(self.theta_bounds[0], self.theta_bounds[1], n_samples)
        
        designs =  {
            'l1': l1,
            'l2': l2,
            'theta': theta,
            'sin_theta': np.sin(theta),
            'cos_theta': np.cos(theta),
            'l2sintheta': l2 * np.sin(theta), # include interacting features for reward prediction network
            'l2costheta': l2 * np.cos(theta) # include interacting features for reward prediction network
        }
        return designs
    

    def find_2D_density(self, n_samples=1000, bins=50):
        """
        Computes a 2D density histogram of end-effector positions sampled from the uniform prior.
        Used for 2D rejection sampling to achieve a uniform (x, y) distribution.
        """
        # sample from the uniform prior
        designs = self.sample_design_uniform_prior(n_samples)
        l1, l2, theta = designs['l1'], designs['l2'], designs['theta']
        # calculate end effector positions
        ee_x = l2 * np.sin(theta)
        ee_y = l1 + l2 * np.cos(theta)

        # Compute 2D histogram and probability densities of each bin
        counts, x_edges, y_edges = np.histogram2d(ee_x, ee_y, bins=bins, density=True)
        points = np.stack([ee_x, ee_y], axis=-1)
        plot_density_heatmap(points)

        # replace empty bins with the minimum nonzero density
        min_nonzero = np.min(counts[counts > 0])
        counts = np.where(counts == 0, min_nonzero, counts)
        
        # calculate the maximum inverse density for normalisation in acceptance probability
        # (i.e. goes from density back to counts)
        max_inv_density = np.max(1.0 / counts)

        return counts, x_edges, y_edges, max_inv_density

    def acceptance_probability(self, ee_x, ee_y):
        """
        Calculates the 2D acceptance probability for each end-effector position.
        Accepts with probability inversely proportional to the local 2D density,
        yielding a uniform distribution over (x, y).
        """
        # gets the relevant histogram bin for each end-effector position
        xi = np.digitize(ee_x, self.x_edges) - 1
        yi = np.digitize(ee_y, self.y_edges) - 1

        
        n_x, n_y = self.density_2d.shape
        # makes sure to only calculate acceptance probability for points that fall within the histogram bounds
        valid_mask = (xi >= 0) & (xi < n_x) & (yi >= 0) & (yi < n_y)

        p_accept = np.zeros(len(ee_x))
        # gets the density for the valid points and calculates acceptance probability of density (normalized by max_inv_density)
        f_xy = self.density_2d[xi[valid_mask], yi[valid_mask]] # gets the density for the valid points for end-effector positions
        p_accept[valid_mask] = (1.0 / f_xy) / self.max_inv_density

        return np.random.rand(len(ee_x)) < p_accept
    
    
    def sample_designs_uniform_end_effector(self, n_samples=1):
        """
        Samples tools such that end-effector position distributed uniformly in the annulus defined by reach of the tool.
        Implements rejection sampling: we sample from the uniform prior and reject samples according to acceptance probability.
        """
        # sample from prior
        designs = self.sample_design_uniform_prior(n_samples)
        
        # calculate end-effector positions
        l1 = designs['l1']
        l2 = designs['l2']
        theta = designs['theta']
        ee_x = l2 * np.sin(theta)
        ee_y = l1 + l2 * np.cos(theta)
        points = np.stack([ee_x, ee_y], axis=-1)
        
        # calculate acceptance probability based on the 2D density
        accept_mask = self.acceptance_probability(ee_x, ee_y)
        
        valid_end_points = points[accept_mask]
        
        # Note that we construct the dataset to include sin(theta) and cos(theta) so that the network understands that 0 is near 360
        # We have also input interacting features l2sintheta, l2costheta
        valid_designs = {
            'l1': l1[accept_mask],
            'l2': l2[accept_mask],
            'theta': theta[accept_mask],
            'sin_theta': np.sin(theta[accept_mask]),
            'cos_theta': np.cos(theta[accept_mask]),
            'l2sintheta': l2[accept_mask] * np.sin(theta[accept_mask]), # a bit of feature engineering here
            'l2costheta': l2[accept_mask] * np.cos(theta[accept_mask])
        }
        
        return valid_end_points, valid_designs
    
    
    def sample_target_locations(self, n_samples):
        """
        Samples random target locations within a (square) bounding box defined by the maximum reach of the tool.
        """
        center_x = 0.0
        center_y = (self.l1_bounds[1] + self.l1_bounds[0]) / 2  # mean l1
        center = np.array([center_x, center_y])

        min_y = center[1] - (self.l1_bounds[0]+self.l1_bounds[1])
        max_y = center[1] + (self.l1_bounds[0]+self.l1_bounds[1])
        min_x = center[0] -  (self.l1_bounds[0]+self.l1_bounds[1])
        max_x = center[0] + (self.l1_bounds[0]+self.l1_bounds[1])
        x = np.random.uniform(min_x, max_x, n_samples)
        y = np.random.uniform(min_y, max_y, n_samples)
        return np.stack([x, y], axis=-1)
    
    def sample_dataset(self, n_samples=1):
        """
        Samples tools and random target location and calculates reward as distance between the two to create a dataset.
        """
        
        # sample tool parameters with uniform end-effector distribution
        end_effectors, designs = self.sample_designs_uniform_end_effector(n_samples)
        # sample target locations uniformly in the (square) bounding box defined by the maximum reach of the tool
        targets = self.sample_target_locations(len(end_effectors)) # needs to match the number of valid designs after rejection sampling

        # Compute rewards based on the reward type
        if self.reward_type == "euclidean_distance":
            
            rewards = -np.linalg.norm(targets - end_effectors, axis=1)
            # invert 
            rewards *= -1
            # scale between 0 and 1
            rewards = (rewards-np.min(rewards))/(np.max(rewards)-np.min(rewards))
            
        elif self.reward_type == "mse":
            rewards = np.mean((targets - end_effectors)**2, axis=1)
        elif self.reward_type == "gaussian_kernel":
            sigma = 225
            rewards = np.exp(-0.5 * (np.linalg.norm(targets - end_effectors, axis=1) / sigma) ** 2)
        elif self.reward_type == "saturated_euclidean_distance":
            sigma = 300
            rewards = -np.tanh(np.linalg.norm(targets - end_effectors, axis=1) / sigma)

        else:
            raise ValueError(f"Invalid reward type: {self.reward_type}")

        data = {
            'l1': designs['l1'],
            'l2': designs['l2'],
            'theta': designs['theta'],
            'sin_theta': designs['sin_theta'],
            'cos_theta': designs['cos_theta'],
            'l2sintheta': designs["l2sintheta"], # include interacting features for reward prediction network
            'l2costheta': designs["l2costheta"], # include interacting features for reward prediction network
            'x_target': targets[:, 0],
            'y_target': targets[:, 1],
            'reward': rewards
        }
        
        return data

class CustomDataset(Dataset):
    def __init__(self, dataframe, dataset_stats=None, label_stats=None):
        # Load the data into memory once
        self.data = dataframe
        
        # Pre-convert columns to tensors to save time during training
        self.features = torch.tensor(self.data[['l1', 'l2', 'sin_theta', 'cos_theta', 'l2sintheta', 'l2costheta',"x_target", "y_target"]].values, dtype=torch.float32)
        self.labels = torch.tensor(self.data['reward'].values, dtype=torch.float32).unsqueeze(1)
        
        # 2. Apply normalization if stats are provided
        if dataset_stats is not None:
            self.features = (self.features - dataset_stats['mean']) / dataset_stats['std']
            
        if label_stats is not None:
            self.labels = (self.labels - label_stats['mean']) / label_stats['std']
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
def main():
    
    save_file = ToolDatasetConfig.SAVE_PATH
    l1_bounds = ToolDatasetConfig.L1_BOUNDS 
    l2_bounds = ToolDatasetConfig.L2_BOUNDS 
    theta_bounds = ToolDatasetConfig.THETA_BOUNDS   
    reward_type = ToolDatasetConfig.REWARD_TYPE
    
    tool_dataset = ToolDataset(l1_bounds, l2_bounds, theta_bounds, reward_type)

    num_designs = ToolDatasetConfig.NUM_DESIGNS

    end_effectors, designs = tool_dataset.sample_designs_uniform_end_effector(num_designs)
        
    plot_end_effector_and_target_distribution(end_effectors)
    plot_density_heatmap(end_effectors)
        
    targets = tool_dataset.sample_target_locations(len(end_effectors)) # needs to match the number of valid designs after rejection sampling    
    plot_end_effector_and_target_distribution(end_effectors, targets)


    data = tool_dataset.sample_dataset(num_designs)
    
    print(len(data['l1']), len(data['l2']), len(data['theta']), len(data['x_target']), len(data['y_target']), len(data['reward']))
    
    df = pd.DataFrame(data)
    
    # UNCOMMENT TO SAVE DATASET
    #df.to_parquet(save_file)
        
    # visualise_tools(data)
    
    visualise_dataset_distribution(df)

    visualise_target_loc_dist(df)

if __name__ == "__main__":
    main()