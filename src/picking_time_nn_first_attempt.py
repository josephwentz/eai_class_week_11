import numpy as np  
from sklearn.model_selection import train_test_split, KFold 
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt 

data = np.load("picking_time_data.npz")
X, y = data["X"], data["y"]
feature_names = list(data["feature_names"])

### Task 1: Load and prepare the data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 80 held out for testing, 320 for 5-fold CV

# Indices for each split
kf = KFold()
train_indices = {} # Dictionaries containing indices
validation_indices = {}

for i, (train_index, test_index) in enumerate(kf.split(X_train)):
    train_indices[f'split_{i+1}'] = train_index 
    validation_indices[f'split_{i+1}'] = test_index
    # These can be used for X and y

# I'm gonna do normalization by hand because we need to save everything 
# and I think it will be easier this way. Otherwise, StandardScaler 
# or a predefined function may be easier. I also don't want to do this 
# all in a loop because I am afraid of losing things or making 
# debugging needlessly difficult. I know it's ugly...

# Normalizing split 1
X_mean_split_1 = np.mean(X_train[train_indices["split_1"]], axis=0) # Saving all means and std devs
X_std_split_1 = np.std(X_train[train_indices["split_1"]], axis=0)
X_norm_split_1 = (X_train - X_mean_split_1) / X_std_split_1 # Normalizing as one, will need to index later for CV
y_mean_split_1 = np.mean(y_train[train_indices["split_1"]], axis=0)
y_std_split_1 = np.std(y_train[train_indices["split_1"]], axis=0)
y_norm_split_1 = (y_train - y_mean_split_1) / y_std_split_1

# Into TensorDataset to put into DataLoader, taken from solution. 
# I could probably just do a tuple of (X, y).
ds1 = TensorDataset(
    torch.tensor(X_norm_split_1, dtype=torch.float32), 
    torch.tensor(y_norm_split_1, dtype=torch.float32).unsqueeze(1)
)
dataloader_split_1 = DataLoader(ds1, batch_size=32) # DataLoader for split 1

# Normalizing split 2
X_mean_split_2 = np.mean(X_train[train_indices["split_2"]], axis=0)
X_std_split_2 = np.std(X_train[train_indices["split_2"]], axis=0)
X_norm_split_2 = (X_train - X_mean_split_2) / X_std_split_2
y_mean_split_2 = np.mean(y_train[train_indices["split_2"]], axis=0)
y_std_split_2 = np.std(y_train[train_indices["split_2"]], axis=0)
y_norm_split_2 = (y_train - y_mean_split_2) / y_std_split_2

ds2 = TensorDataset(
    torch.tensor(X_norm_split_2, dtype=torch.float32), 
    torch.tensor(y_norm_split_2, dtype=torch.float32).unsqueeze(1)
)
dataloader_split_2 = DataLoader(ds2, batch_size=32)

# Normalizing split 3
X_mean_split_3 = np.mean(X_train[train_indices["split_3"]], axis=0)
X_std_split_3 = np.std(X_train[train_indices["split_3"]], axis=0)
X_norm_split_3 = (X_train - X_mean_split_3) / X_std_split_3
y_mean_split_3 = np.mean(y_train[train_indices["split_3"]], axis=0)
y_std_split_3 = np.std(y_train[train_indices["split_3"]], axis=0)
y_norm_split_3 = (y_train - y_mean_split_3) / y_std_split_3

ds3 = TensorDataset(
    torch.tensor(X_norm_split_3, dtype=torch.float32), 
    torch.tensor(y_norm_split_3, dtype=torch.float32).unsqueeze(1)
)
dataloader_split_3 = DataLoader(ds3, batch_size=32)

# Normalizing split 4
X_mean_split_4 = np.mean(X_train[train_indices["split_4"]], axis=0)
X_std_split_4 = np.std(X_train[train_indices["split_4"]], axis=0)
X_norm_split_4 = (X_train - X_mean_split_4) / X_std_split_4
y_mean_split_4 = np.mean(y_train[train_indices["split_4"]], axis=0)
y_std_split_4 = np.std(y_train[train_indices["split_4"]], axis=0)
y_norm_split_4 = (y_train - y_mean_split_4) / y_std_split_4

ds4 = TensorDataset(
    torch.tensor(X_norm_split_4, dtype=torch.float32), 
    torch.tensor(y_norm_split_4, dtype=torch.float32).unsqueeze(1)
)
dataloader_split_4 = DataLoader(ds4, batch_size=32)

# Normalizing split 5
X_mean_split_5 = np.mean(X_train[train_indices["split_5"]], axis=0)
X_std_split_5 = np.std(X_train[train_indices["split_5"]], axis=0)
X_norm_split_5 = (X_train - X_mean_split_5) / X_std_split_5
y_mean_split_5 = np.mean(y_train[train_indices["split_5"]], axis=0)
y_std_split_5 = np.std(y_train[train_indices["split_5"]], axis=0)
y_norm_split_5 = (y_train - y_mean_split_5) / y_std_split_5

ds5 = TensorDataset(
    torch.tensor(X_norm_split_5, dtype=torch.float32), 
    torch.tensor(y_norm_split_5, dtype=torch.float32).unsqueeze(1)
)
dataloader_split_5 = DataLoader(ds5, batch_size=32)

### Task 2: 3-Hidden-Layer nn

model = nn.Sequential(
    nn.Linear(5, 32), nn.ReLU(),
    nn.Linear(32, 16), nn.ReLU(),
    nn.Linear(16, 8), nn.ReLU(),
    nn.Linear(8, 1)
)
