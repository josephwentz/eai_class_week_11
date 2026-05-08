import numpy as np  
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split, KFold

data = np.load("picking_time_data.npz")
X, y = data["X"], data["y"]
feature_names = list(data["feature_names"])

# Use X as features and y as targets
features = X
targets = y

# Hold out 20% as test set
features_train, features_test, targets_train, targets_test = train_test_split(
    features, targets, test_size=0.2, random_state=42
)

# Split the remaining 80% into 5 folds for cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
folds = []
fold_stats = []
for train_index, val_index in kf.split(features_train):
    X_fold_train = features_train[train_index]
    y_fold_train = targets_train[train_index]
    X_fold_val = features_train[val_index]
    y_fold_val = targets_train[val_index]
    
    # Normalize features using training fold statistics
    mean_X = np.mean(X_fold_train, axis=0)
    std_X = np.std(X_fold_train, axis=0)
    X_fold_train_norm = (X_fold_train - mean_X) / std_X
    X_fold_val_norm = (X_fold_val - mean_X) / std_X
    
    # Normalize targets using training fold statistics
    mean_y = np.mean(y_fold_train)
    std_y = np.std(y_fold_train)
    y_fold_train_norm = (y_fold_train - mean_y) / std_y
    y_fold_val_norm = (y_fold_val - mean_y) / std_y
    
    # Save stats for later use
    fold_stats.append({
        'mean_X': mean_X,
        'std_X': std_X,
        'mean_y': mean_y,
        'std_y': std_y
    })
    
    folds.append((X_fold_train_norm, y_fold_train_norm, X_fold_val_norm, y_fold_val_norm))

# Define the neural network model with variable hidden layers
class Net(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size=1):
        super(Net, self).__init__()
        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            prev_size = h
        layers.append(nn.Linear(prev_size, output_size))
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x

# Training parameters
input_size = features.shape[1]
output_size = 1
num_epochs = 200
batch_size = 32

# Function to evaluate a network depth using cross-validation
def evaluate_depth(hidden_sizes, folds, fold_stats, input_size, output_size, num_epochs, batch_size):
    fold_train_losses = []
    fold_val_losses = []
    fold_val_rmse_orig = []
    for fold_idx, ((X_train, y_train, X_val, y_val), stats) in enumerate(zip(folds, fold_stats)):
        # Convert to torch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
        
        # Create DataLoaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model, optimizer, loss
        model = Net(input_size, hidden_sizes, output_size)
        optimizer = optim.Adam(model.parameters())
        criterion = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        for epoch in range(num_epochs):
            # Training
            model.train()
            epoch_train_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
            epoch_train_loss /= len(train_loader)
            train_losses.append(epoch_train_loss)
            
            # Validation
            model.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    epoch_val_loss += loss.item()
            epoch_val_loss /= len(val_loader)
            val_losses.append(epoch_val_loss)
        
        # Compute fold RMSE in original units from normalized validation error
        std_y = stats['std_y']
        sum_sq_val_norm = 0.0
        n_val = 0
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                sum_sq_val_norm += torch.sum((outputs - y_batch) ** 2).item()
                n_val += y_batch.size(0)
        val_rmse_norm = np.sqrt(sum_sq_val_norm / n_val)
        fold_val_rmse_orig.append(val_rmse_norm * std_y)
        
        fold_train_losses.append(train_losses)
        fold_val_losses.append(val_losses)
    
    return fold_train_losses, fold_val_losses, fold_val_rmse_orig

# Evaluate the 3-layer network with 5-fold cross-validation
hidden_sizes = [32, 16, 8]
fold_train_losses, fold_val_losses, fold_rmse_orig = evaluate_depth(
    hidden_sizes, folds, fold_stats, input_size, output_size, num_epochs, batch_size
)

# Compute mean curves
mean_train_losses = np.mean(fold_train_losses, axis=0)
mean_val_losses = np.mean(fold_val_losses, axis=0)

# Compute cross-validation RMSE in original units
cv_rmse_mean = np.mean(fold_rmse_orig)
cv_rmse_std = np.std(fold_rmse_orig)
print(f'3-layer CV RMSE: {cv_rmse_mean:.4f} ± {cv_rmse_std:.4f} seconds')

# Plot per-fold validation curves and mean training/validation curves
plt.figure(figsize=(10, 6))
epochs = range(1, num_epochs + 1)
for fold_idx in range(len(folds)):
    plt.semilogy(epochs, fold_val_losses[fold_idx], alpha=0.5, color='gray', label='Fold Val' if fold_idx == 0 else None)

plt.semilogy(epochs, mean_train_losses, 'b-', linewidth=2, label='Mean Train')
plt.semilogy(epochs, mean_val_losses, 'r-', linewidth=2, label='Mean Val')

plt.xlabel('Epoch')
plt.ylabel('MSE Loss (log scale)')
plt.title('3-Layer Network: Per-Fold Val Curves and Mean Train/Val Curves')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# After cross-validation, retrain on the full training set
# Compute normalization stats for full training set
mean_X_full = np.mean(features_train, axis=0)
std_X_full = np.std(features_train, axis=0)
mean_y_full = np.mean(targets_train)
std_y_full = np.std(targets_train)

# Normalize full training set
features_train_norm = (features_train - mean_X_full) / std_X_full
targets_train_norm = (targets_train - mean_y_full) / std_y_full

# Normalize test set with same stats
features_test_norm = (features_test - mean_X_full) / std_X_full
targets_test_norm = (targets_test - mean_y_full) / std_y_full

# Convert to torch tensors
X_train_full = torch.tensor(features_train_norm, dtype=torch.float32)
y_train_full = torch.tensor(targets_train_norm, dtype=torch.float32).unsqueeze(1)
X_test_full = torch.tensor(features_test_norm, dtype=torch.float32)
y_test_full = torch.tensor(targets_test_norm, dtype=torch.float32).unsqueeze(1)

# DataLoader for full training set
train_dataset_full = TensorDataset(X_train_full, y_train_full)
train_loader_full = DataLoader(train_dataset_full, batch_size=batch_size, shuffle=True)

# Retrain model with same hyperparameters
hidden_sizes = [32, 16, 8]
model_final = Net(input_size, hidden_sizes, output_size)
optimizer_final = optim.Adam(model_final.parameters())
criterion_final = nn.MSELoss()

for epoch in range(num_epochs):
    model_final.train()
    for X_batch, y_batch in train_loader_full:
        optimizer_final.zero_grad()
        outputs = model_final(X_batch)
        loss = criterion_final(outputs, y_batch)
        loss.backward()
        optimizer_final.step()

# Evaluate on test set
model_final.eval()
with torch.no_grad():
    predictions_norm = model_final(X_test_full)

# Denormalize predictions
predictions = predictions_norm.squeeze().numpy() * std_y_full + mean_y_full

### Compute RMSE on original scale
rmse = np.sqrt(np.mean((predictions - targets_test)**2))
print(f'Final Test RMSE: {rmse:.4f}')

### Fit linear regression via normal equation on normalized data
# Add bias term (intercept)
X_train_bias = np.c_[np.ones(features_train_norm.shape[0]), features_train_norm]

# Normal equation: theta = (X^T X)^{-1} X^T y
theta = np.linalg.inv(X_train_bias.T @ X_train_bias) @ X_train_bias.T @ targets_train_norm

# Predict on normalized test set
X_test_bias = np.c_[np.ones(features_test_norm.shape[0]), features_test_norm]
predictions_lr_norm = X_test_bias @ theta

# Denormalize predictions
predictions_lr = predictions_lr_norm * std_y_full + mean_y_full

# Compute RMSE on original scale
rmse_lr = np.sqrt(np.mean((predictions_lr - targets_test)**2))
print(f'Linear Regression Test RMSE: {rmse_lr:.4f}')

# Scatter plots: predicted vs actual on held-out test set
min_val = min(targets_test.min(), predictions.min(), predictions_lr.min())
max_val = max(targets_test.max(), predictions.max(), predictions_lr.max())
line_vals = np.linspace(min_val, max_val, 100)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(targets_test, predictions, alpha=0.7, label='NN predictions')
plt.plot(line_vals, line_vals, 'k--', label='Perfect prediction')
plt.xlabel('Actual Picking Time')
plt.ylabel('Predicted Picking Time')
plt.title('Neural Network: Predicted vs Actual')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(targets_test, predictions_lr, alpha=0.7, label='Linear regression predictions')
plt.plot(line_vals, line_vals, 'k--', label='Perfect prediction')
plt.xlabel('Actual Picking Time')
plt.ylabel('Predicted Picking Time')
plt.title('Linear Regression: Predicted vs Actual')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

