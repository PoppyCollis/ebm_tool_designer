import torch

class ToolDatasetConfig:
    
    L1_BOUNDS = (100.0,300.0) # make sure these are floats
    L2_BOUNDS = (100.0,300.0)
    THETA_BOUNDS = (0.0,360.0)
    
    NUM_DESIGNS = 10000
    
    REWARD_TYPE = "euclidean_distance"
    
    SAVE_PATH = 'src/ebm_tool_designer/data/dummy_dataset.parquet'
    
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class RewardModelConfig:
    
    NUM_SEEDS = 1
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    TEST_SPLIT_SEED = 42
    
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    DATA_PATH = 'src/ebm_tool_designer/data/dummy_dataset.parquet'
    WEIGHTS_SAVE_PATH = 'src/ebm_tool_designer/weights/reward_model_best.pt'
    
    IN_FEATURES = 8 
    HIDDEN_FEATURES = 128
    OUT_FEATURES = 64
    
    SIGMA = 0.1 # 0.01 after normalizing all the continuous attributes to [0, 1]
    
    
class EBMConfig:
    
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    SAMPLING_METHOD = 'langevin'
    N_SAMPLES = 9
    N_SAMPLING_STEPS = 10000
    ETA = 1e-4
    
    E_SCALING_FACTOR = 1e-1 # 1e-4