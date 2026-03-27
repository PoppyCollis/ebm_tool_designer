import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
import numpy as np
    
def visualise_tools(designs, target_location=None, energies=None):
    n_samples = len(designs["l1"])
    
    # We need to store the actual (x, y) points to determine the plot bounds
    all_joint_positions = [] 
    
    for i in range(n_samples):
        l1 = designs['l1'][i]
        l2 = designs['l2'][i]
        theta = designs['theta'][i]
        
        # Calculate vertical orientation coordinates
        p1 = (0, 0)           # Base
        p2 = (0, l1)          # Joint (Vertical)
        p3 = (l2 * np.sin(theta), l1 + l2 * np.cos(theta)) # Tip
        
        all_joint_positions.append([p1, p2, p3])

    # Convert to a flat numpy array of (x, y) pairs to find the global max/min
    # Shape will be (n_samples * 3, 2)
    all_pts = np.array(all_joint_positions).reshape(-1, 2)
    
    # Calculate bounds based on the maximum excursion in any direction
    max_val = np.max(np.abs(all_pts)) * 1.1 
    
    # Set up subplots
    cols = 3
    rows = (n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4), squeeze=False, constrained_layout=True)
    axes = axes.flatten()

    for i in range(n_samples):
        
        
        p1, p2, p3 = all_joint_positions[i]
        ax = axes[i]
                
        # Segment 1 (l1): base to joint
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-o', lw=4, markersize=2, color='#2c3e50', label='l1' if i == 0 else None)
        # Segment 2 (l2): joint to tip
        ax.plot([p2[0], p3[0]], [p2[1], p3[1]], '-o', lw=4, markersize=2, color='#e74c3c', label='l2' if i == 0 else None)
        
        if target_location is not None:
            x = target_location.flatten()[0]
            y = target_location.flatten()[1]
            ax.scatter(x, y, color='red', s=100, label='Target Location', zorder=5)
                
        # Consistent scaling across all subplots
        if energies is not None:
            ax.set_title(f"θ: {np.degrees(designs['theta'][i]):.1f}°\nl1: {designs['l1'][i]:.1f}\nl2: {designs['l2'][i]:.1f}\n Energy: {energies[i]:.4f}")
        else:   
            ax.set_title(f"θ: {np.degrees(designs['theta'][i]):.1f}°\nl1: {designs['l1'][i]:.1f}\nl2: {designs['l2'][i]:.1f}")
        ax.set_xlim(-max_val, max_val)
        ax.set_ylim(-max_val, max_val) 
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.6)

        
    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()
    
    
def plot_losses(epochs, epoch_train_losses, epoch_val_losses, title= "Loss vs Epochs"):  
    
    plt.plot(range(epochs), epoch_train_losses, label='Train')
    plt.plot(range(epochs), epoch_val_losses, label='Validation')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.title(title)
    
    plt.show()
    
    
def plot_mean_losses(epochs, mean_train_loss, std_train_loss, mean_val_loss, std_val_loss, title= "Mean Loss vs Epochs"):
    x_axis = range(len(mean_train_loss))
    
    plt.plot(x_axis, mean_train_loss, label='Mean train Loss')
    plt.fill_between(x_axis, 
                 mean_train_loss - std_train_loss, 
                 mean_train_loss + std_train_loss, 
                 alpha=0.2, label='$\pm$ 1 Std Dev')
    
    plt.plot(x_axis, mean_val_loss, label='Mean val Loss')
    plt.fill_between(x_axis, 
                 mean_val_loss - std_val_loss, 
                 mean_val_loss + std_val_loss, 
                 alpha=0.2, label='$\pm$ 1 Std Dev')
    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.title(title)
    plt.show()
    
    
def plot_energy_hist(energy_hist):
    e = np.array(energy_hist)
    std = np.std(e)
    x = np.arange(len(e))
    
    # Plot the main line
    plt.plot(x, e, color='blue', label='Energy')
    
    # Create the shaded error region
    plt.fill_between(x, e - std, e + std, color='blue', alpha=0.2, lw=0)
    
    plt.xlabel('Iteration')
    plt.ylabel('Energy')
    plt.legend()
    plt.show()

def visualise_dataset_distribution(df):
    
    # Set the number of columns for the grid
    n_cols = 3
    n_rows = (len(df.columns) + n_cols - 1) // n_cols

    # Create the figure
    plt.figure(figsize=(15, 4 * n_rows))

    for i, col in enumerate(df.columns):
        plt.subplot(n_rows, n_cols, i + 1)
        # Plotting the histogram
        plt.hist(df[col], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        
        # Formatting titles and labels with LaTeX-style names
        title_label = col.replace('_', ' ')
        plt.title(f'Distribution of {title_label}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()
    
def visualise_end_effector_dist(df):
    
    fig, ax = plt.subplots(figsize=(10, 7))

    # Now ax.hexbin will work directly
    hb1 = ax.hexbin(df['end_effector_x'], df['end_effector_y'], 
                    gridsize=40, cmap='viridis', mincnt=1)
    
    # Add the colorbar to the single ax
    fig.colorbar(hb1, ax=ax, label='Sample Count')
    
    ax.set_title("End Effector Position Density\n(The 'Potential' Workspace)")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.show() 
    
def visualise_target_loc_dist(df):
    
    fig, ax = plt.subplots(figsize=(10, 7))

    # 2. Target Location Heatmap (The "Goals")
    hb2 = ax.hexbin(df['x_target'], df['y_target'], 
                    gridsize=40, cmap='magma', mincnt=1)
    # Add the colorbar to the single ax
    fig.colorbar(hb2, ax=ax, label='Sample Count')
    
    ax.set_title("Target Location Density")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()
    

def plot_end_effector_and_target_distribution(points, targets=None):
    eps = 25
    plt.figure(figsize=(6,6))
    if targets is not None:
        plt.scatter(targets[:,0], targets[:,1], alpha=0.5, s=10, color='red', label='Target Samples')
    plt.scatter(points[:,0], points[:,1], alpha=0.5, s=10, label='End Effector Samples')
    plt.xlim(np.min(points[:,0])-eps, np.max(points[:,0])+eps)
    plt.ylim(np.min(points[:,1])-eps, np.max(points[:,1])+eps)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Uniformly Sampled Points in Annulus')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid()
    plt.legend()
    plt.show()

def plot_density_heatmap(points, targets=None, bins=25):
    # Setup the pivot for visual reference
    # Assuming the anchor point we discussed
    anchor_x = 0.0
    anchor_y = 0.0 # Update this to your specific (l1_max - l1_min) if needed
    
    plt.figure(figsize=(10, 8))
    
    # 1. Plot the Heatmap of End Effector Points
    # 'bins' controls the resolution, 'cmap' controls the colors
    plt.hist2d(points[:, 0], points[:, 1], bins=bins, cmap='magma', cmin=1)
    cb = plt.colorbar()
    cb.set_label(label='Point Density (Count per Bin)', fontsize=15)

    cb.ax.tick_params(labelsize=15)

    
    # 2. Overlay Target Samples as a subtle wireframe or outline if they exist
    # (Or just plot them as small dots to see coverage)
    if targets is not None:
        plt.scatter(targets[:, 0], targets[:, 1], alpha=0.1, s=1, color='cyan', label='Target Coverage')
    
    # 3. Mark the Pivot Point - The "Center" of the Donut
    plt.scatter(anchor_x, anchor_y, color='red', marker='x', s=100, label='Arm Pivot')

    # Formatting
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('2D Density Heatmap of End-Effector Positions', fontsize=20)
    plt.xlabel('X Coordinate', fontsize=20)
    plt.ylabel('Y Coordinate', fontsize=20)
    plt.grid(alpha=0.3)
    plt.legend(loc='upper right', fontsize=15)
    plt.show()
    
    
def plot_sample_predictions(ee_positions, target_positions, rewards_true, rewards_pred, n_samples=5):
    """
    Plots n tool end-effector positions, their target locations, and a dotted line between each pair.
    Annotates each pair with the true and predicted reward.
    """
    indices = np.random.choice(len(ee_positions), size=n_samples, replace=False)

    fig, ax = plt.subplots(figsize=(8, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, n_samples))

    for i, idx in enumerate(indices):
        ee = ee_positions[idx]
        tgt = target_positions[idx]
        c = colors[i]

        # Dotted line connecting ee to target
        ax.plot([ee[0], tgt[0]], [ee[1], tgt[1]], '--', color=c, alpha=0.6, lw=1.5)
        # End effector
        ax.scatter(ee[0], ee[1], color=c, marker='o', s=80, zorder=5)
        # Target
        ax.scatter(tgt[0], tgt[1], color=c, marker='x', s=80, zorder=5)
        # Label with rewards
        mid_x = (ee[0] + tgt[0]) / 2
        mid_y = (ee[1] + tgt[1]) / 2
        ax.annotate(f"true={rewards_true[idx]:.3f}\npred={rewards_pred[idx]:.3f}",
                    (mid_x, mid_y), fontsize=7, ha='center',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

    # Legend entries
    ax.scatter([], [], color='gray', marker='o', s=80, label='End Effector')
    ax.scatter([], [], color='gray', marker='x', s=80, label='Target')

    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f'{n_samples} Sample Predictions')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_prediction_error_heatmap(ee_positions, errors, bins=50):
    """
    Plots a 2D heatmap of average prediction error binned by end-effector (x, y) position.

    Args:
        ee_positions: (N, 2) array of end-effector [x, y] positions
        errors: (N,) array of absolute prediction errors
        bins: number of bins per axis
    """

    x = ee_positions[:, 0]
    y = ee_positions[:, 1]

    # Bin the errors by (x, y) position, taking the mean error in each bin
    stat, x_edges, y_edges, binnumber = binned_statistic_2d(
        x, y, errors, statistic='mean', bins=bins
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    # stat is (bins_x, bins_y), pcolormesh expects (bins_y, bins_x) via transpose
    mesh = ax.pcolormesh(x_edges, y_edges, stat.T, cmap='hot', shading='auto')
    fig.colorbar(mesh, ax=ax, label='Mean Absolute Error')

    ax.scatter(0, 0, color='cyan', marker='x', s=100, label='Arm Base')
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Mean Prediction Error by End-Effector Position')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


def plot_radial_density(radii, bins = 100):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(radii, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    ax.set_title('1D Radial Density')
    ax.set_xlabel('Radius $r$ (Distance from Base)')
    ax.set_ylabel('Frequency')
    plt.tight_layout()
    plt.show()