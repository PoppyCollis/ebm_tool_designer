"""
Tests the performance of the trained reward prediction network and visualises the results.
"""


import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from ebm_tool_designer.tool_dataset import CustomDataset
from ebm_tool_designer.helpers.plots import plot_prediction_error_heatmap, plot_sample_predictions
from ebm_tool_designer.config import RewardModelConfig
from ebm_tool_designer.reward_model import MLP

    
    
def main():
    
    # --- CHECK PREDICTIONS OF TRAINED MODEL ---

    device = RewardModelConfig.DEVICE
    save_path = RewardModelConfig.WEIGHTS_SAVE_PATH
    train_test_split_seed = RewardModelConfig.TEST_SPLIT_SEED

    checkpoint = torch.load(save_path, map_location=device)

    reward_model = MLP(
        in_features=RewardModelConfig.IN_FEATURES,
        hidden_features=RewardModelConfig.HIDDEN_FEATURES,
        out_features=RewardModelConfig.OUT_FEATURES
    )
    reward_model.load_state_dict(checkpoint['model_state_dict'])
    reward_model.eval()
    reward_model.to(device)

    # Keep stats on CPU for dataset normalization; move to device only for inference
    feature_stats = {k: v.cpu() for k, v in checkpoint['feature_stats'].items()}
    label_stats = {k: v.cpu() for k, v in checkpoint['label_stats'].items()}

    # Load test data
    full_df = pd.read_parquet(RewardModelConfig.DATA_PATH)
    _, test_df = train_test_split(full_df, test_size=0.2, random_state=train_test_split_seed)

    # Build test dataset (normalized)
    cols = ['l1', 'l2', 'sin_theta', 'cos_theta', 'l2sintheta', 'l2costheta', "x_target", "y_target"]
    test_dataset = CustomDataset(test_df, feature_stats, label_stats)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    # Collect predictions and ground truth
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            preds = reward_model(features)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).squeeze().numpy()
    all_labels = torch.cat(all_labels).squeeze().numpy()

    # Unnormalize predictions and labels back to original scale
    label_mean = label_stats['mean'].cpu().numpy()
    label_std = label_stats['std'].cpu().numpy()
    all_preds = all_preds * label_std + label_mean
    all_labels = all_labels * label_std + label_mean

    # Compute absolute errors
    errors = np.abs(all_preds - all_labels)

    # Compute end-effector positions from raw test data
    l1 = test_df['l1'].values
    l2 = test_df['l2'].values
    theta = test_df['theta'].values
    ee_x = l2 * np.sin(theta)
    ee_y = l1 + l2 * np.cos(theta)
    ee_positions = np.stack([ee_x, ee_y], axis=-1)

    print(f"Test MAE: {np.mean(errors):.4f}")
    print(f"Test Median AE: {np.median(errors):.4f}")

    plot_prediction_error_heatmap(ee_positions, errors)

    # Plot 5 sample predictions with ee, target, and dotted distance lines
    target_positions = test_df[['x_target', 'y_target']].values
    plot_sample_predictions(ee_positions, target_positions, all_labels, all_preds, n_samples=10)
    
if __name__ == "__main__":
    main()

