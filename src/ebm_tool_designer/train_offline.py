"""
Trains the reward prediction network on an offline dataset.
"""
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from ebm_tool_designer.tool_dataset import CustomDataset
from ebm_tool_designer.helpers.plots import plot_mean_losses, plot_losses, plot_prediction_error_heatmap
from ebm_tool_designer.config import RewardModelConfig, ToolDatasetConfig
from ebm_tool_designer.reward_model import MLP

def train_model(device, reward_model, train_loader, test_loader, optimiser, criterion, epochs):
    
    reward_model.eval()
    reward_model.to(device)
    
    # run a quick pre-test of the network performance before training

    val_loss = 0.0
    train_loss = 0.0
        
    with torch.no_grad():
        for f, l in train_loader:
            fs = f.to(device)
            ls = l.to(device)
            preds = reward_model(fs)
            loss = criterion(preds, ls)
            train_loss += loss.item()
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            preds = reward_model(features)
            loss2 = criterion(preds, labels)
            val_loss += loss2.item()
            
    print(f"Pre-test | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # training loop 
    
    reward_model.train()
    
    epoch_train_losses = []
    epoch_val_losses = []

    for epoch in range(epochs):

        train_loss = 0.0
        val_loss = 0.0
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        for batch_idx, (features, labels) in enumerate(pbar):
            
            features = features.to(device)
            labels = labels.to(device)

            optimiser.zero_grad()
            r_pred = reward_model.forward(features)
            loss = criterion(r_pred, labels)
            loss.backward()
            optimiser.step()
            
            running_loss += loss.item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # store train and test loss for each epoch
        reward_model.eval() 
        
        with torch.no_grad():
            for f, l in train_loader:
                fs = f.to(device)
                ls = l.to(device)
                preds = reward_model(fs)
                loss = criterion(preds, ls)
                train_loss += loss.item()
        
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(device)
                labels = labels.to(device)
                preds = reward_model(features)
                loss2 = criterion(preds, labels)
                val_loss += loss2.item()
                
        avg_train_loss = train_loss / len(train_loader)
        epoch_train_losses.append(avg_train_loss)
        
        avg_val_loss = val_loss / len(test_loader)
        epoch_val_losses.append(avg_val_loss)
        
        reward_model.train()
       
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
    # plot_losses(epochs, epoch_train_losses, epoch_val_losses)
        
    return epoch_train_losses, epoch_val_losses

def run_n_trials(num_trials, device, train_loader, test_loader, criterion, epochs, lr):

    train_losses_list = []
    val_losses_list = []
    for i in range(num_trials):
        print(f"Starting Seed {i+1}/5")
        reward_model = MLP(in_features=RewardModelConfig.IN_FEATURES, hidden_features=RewardModelConfig.HIDDEN_FEATURES, out_features=RewardModelConfig.OUT_FEATURES).to(device)
        optimiser = torch.optim.Adam(reward_model.parameters(), lr=lr)
        train_losses, val_losses = train_model(device, reward_model, train_loader, test_loader, optimiser, criterion, epochs=epochs)
        train_losses_list.append(train_losses)
        val_losses_list.append(val_losses)
        last_model = reward_model

    mean_train_loss = np.mean(np.array(train_losses_list), axis=0)
    mean_val_loss = np.mean(np.array(val_losses_list), axis=0)
    std_train_loss = np.std(np.array(train_losses_list), axis=0)
    std_val_loss = np.std(np.array(val_losses_list), axis=0)

    return last_model, mean_train_loss, mean_val_loss, std_train_loss, std_val_loss


def main():
        
    # load in config params
    batch_size = RewardModelConfig.BATCH_SIZE
    epochs = RewardModelConfig.EPOCHS
    lr = RewardModelConfig.LEARNING_RATE
    num_trials = RewardModelConfig.NUM_SEEDS
    device = RewardModelConfig.DEVICE
    save_path = RewardModelConfig.WEIGHTS_SAVE_PATH
    train_test_split_seed = RewardModelConfig.TEST_SPLIT_SEED

    
    print('device:', device)
    
    # --- LOAD IN DATASET, TRAIN-TEST SPLIT AND STANDARDISE ---
    
    # Read data
    data_path =  RewardModelConfig.DATA_PATH
    full_df = pd.read_parquet(data_path)
    
    # train test split 80%
    train_df, test_df = train_test_split(full_df, test_size=0.2, random_state=train_test_split_seed)
        
    cols = ['l1', 'l2', 'sin_theta', 'cos_theta', 'l2sintheta', 'l2costheta', "x_target", "y_target"]
    label_col = ['reward']
    
    # Calculate mean and std of features: should only calculate them based on the train portion of data
    mean_vals = torch.tensor(train_df[cols].mean().values, dtype=torch.float32)
    std_vals = torch.tensor(train_df[cols].std().values, dtype=torch.float32)
    
    # sin and cos theta are already normalised between 1,-1
    # set mean to 0 and std to 1
    mean_vals[2:4] = 0.0
    std_vals[2:4] = 1.0
    
    feature_stats = {'mean': mean_vals, 'std': std_vals}

    # Calculate mean and std of rewards
    label_stats = {
    'mean': torch.tensor(train_df[label_col].mean().values, dtype=torch.float32),
    'std': torch.tensor(train_df[label_col].std().values, dtype=torch.float32)
    }

    train_dataset = CustomDataset(train_df, feature_stats, label_stats)
    test_dataset = CustomDataset(test_df, feature_stats, label_stats)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(train_dataset.data.shape)

    print(test_dataset.data.shape)
    
    
    # --- TRAIN REWARD MODEL OFFLINE ---

    # Use Mean Squared Error loss for regression
    criterion = nn.MSELoss()

    reward_model, mean_train_loss, mean_val_loss, std_train_loss, std_val_loss = run_n_trials(num_trials, device, train_loader, test_loader, criterion, epochs, lr)
    
    plot_mean_losses(epochs, mean_train_loss, std_train_loss, mean_val_loss, std_val_loss)
    
    
    # --- SAVING ---
    
    # Save the model state, but also the optimizer state and metadata

    torch.save({
        'epoch': epochs,
        'model_state_dict': reward_model.state_dict(),
        'feature_stats': feature_stats, # Save normalization stats
        'label_stats': label_stats,
    }, save_path)

    
if __name__ == "__main__":
    main()

