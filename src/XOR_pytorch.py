import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Dataset
X = torch.tensor(
    [[0, 0], [0, 1], [1, 0], [1, 1]],
    dtype=torch.float32,
)
Y = torch.tensor(
    [[0], [1], [1], [0]],
    dtype=torch.float32,
)

# Model Definition
class XORNet(nn.Module):
    """Input(2) -> Hidden(4, ReLU) -> Output(1, sigmoid)."""
    def __init__(self):
        super().__init__()
        # Layer 1: 2 inputs -> 4 hidden neurons
        self.hidden = nn.Linear(2, 4)
        # Layer 2: 4 hidden -> 1 output
        self.output = nn.Linear(4, 1)
    def forward(self, x):
        # Hidden layer: linear transform + ReLU activation
        a1 = torch.relu(self.hidden(x))
        # Output layer: linear transform + sigmoid activation
        y_hat = torch.sigmoid(self.output(a1))
        return y_hat
    
# Training set up
torch.manual_seed(0)
model = XORNet()
criterion = nn.BCELoss()                          # binary cross-entropy
optimizer = optim.SGD(model.parameters(), lr=0.1)  # plain gradient descent
epochs = 5000

# Training loop
losses = []
for epoch in range(epochs):
    # Forward pass: all 4 examples in one matrix multiply
    y_hat = model(X)
    loss = criterion(y_hat, Y)
    # Backward pass + update (replaces our manual backward() + -= lr * grad)
    optimizer.zero_grad()   # reset gradients to zero
    loss.backward()         # compute dL/d(all parameters) via chain rule
    optimizer.step()        # W <- W - lr * dW  for every parameter
    losses.append(loss.item())

# Results
print("Final predictions:")
print(f"{'Input':>10s}  {'Target':>6s}  {'Predicted':>9s}")
with torch.no_grad():  # disable gradient tracking for inference
    preds = model(X)
    for i in range(len(X)):
        print(
            f"  {X[i].tolist()}     {Y[i].item():.0f}      "
            f"{preds[i].item():.4f}"
        )

