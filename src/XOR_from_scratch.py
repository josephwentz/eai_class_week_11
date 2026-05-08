import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    """Sigmoid activation (Section 6.1): maps any real z to (0, 1)."""
    return 1 / (1 + np.exp(-z))
def relu(z):
    """ReLU activation (Section 6.1): max(0, z), applied element-wise."""
    return np.maximum(0, z)
def relu_grad(z):
    """Derivative of ReLU w.r.t. the pre-activation z.
    Returns 1 where z > 0 (the linear region) and 0 where z <= 0
    (the dead region discussed in Section 6.1).
    """
    return (z > 0).astype(float)

# Forward pass
def forward(x, W1, b1, W2, b2):
    """Forward pass through both layers. Returns prediction and cache."""
    # --- Hidden layer (layer 1) ---
    z1 = W1 @ x + b1       # linear: weight matrix times input plus bias
    a1 = relu(z1)           # activation: element-wise ReLU
    # --- Output layer (layer 2) ---
    z2 = W2 @ a1 + b2       # linear: weight vector times hidden activations
    a2 = sigmoid(z2)        # activation: sigmoid produces probability in (0,1)
    # Store intermediates for the backward pass
    cache = {"x": x, "z1": z1, "a1": a1, "z2": z2, "a2": a2}
    return a2.item(), cache

# Loss function (BCE loss)
def bce_loss(y_hat, y):
    """Binary cross-entropy for a single example (Section 5.3).
    L = -[y log(y_hat) + (1-y) log(1-y_hat)]
    """
    eps = 1e-12  # prevent log(0)
    return -(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))

# Backward pass
def backward(y, cache, W2):
    """Backpropagation: compute gradients of BCE loss w.r.t. all parameters."""
    x  = cache["x"]
    z1 = cache["z1"]
    a1 = cache["a1"]
    a2 = cache["a2"]
    # --- Output layer gradients ---
    # d(BCE)/d(z2) = a2 - y  (sigmoid + BCE cancellation from Section 5.3)
    dz2 = a2 - y                        # (1,) gradient at output pre-activation
    dW2 = dz2[:, None] @ a1[None, :]   # (1, n_h) outer product: dL/dW2
    db2 = dz2                           # (1,) dL/db2 = dL/dz2
    # --- Hidden layer gradients ---
    # Propagate gradient back through output weights
    da1 = W2.T @ dz2                    # (n_h,) gradient at hidden activations
    # Propagate through ReLU: zero out gradients where z1 <= 0
    dz1 = da1 * relu_grad(z1)          # (n_h,) element-wise mask
    # Gradient w.r.t. hidden weights and biases
    dW1 = dz1[:, None] @ x[None, :]    # (n_h, 2) outer product: dL/dW1
    db1 = dz1                           # (n_h,) dL/db1 = dL/dz1
    return dW1, db1, dW2, db2

# Dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y = np.array([0, 1, 1, 0], dtype=float)

# Weight initialization
n_hidden = 4   # width of the hidden layer
lr = 0.1       # learning rate (alpha)
epochs = 5000  # number of full passes through the dataset
seed = 0       # for reproducibility
rng = np.random.default_rng(seed) # With a seed value, it will always choose the SAME random number value!!
# He initialization: variance = 2 / fan_in
W1 = rng.normal(0, np.sqrt(2 / 2), size=(n_hidden, 2))        # (4, 2)
b1 = np.zeros(n_hidden)                                        # (4,)
W2 = rng.normal(0, np.sqrt(2 / n_hidden), size=(1, n_hidden))  # (1, 4)
b2 = np.zeros(1)                                                # (1,)

# Training loop
losses = []
for epoch in range(epochs):
    epoch_loss = 0.0
    for i in range(len(X)):
        # 1. Forward pass
        y_hat, cache = forward(X[i], W1, b1, W2, b2)
        # 2. Accumulate loss for monitoring
        epoch_loss += bce_loss(y_hat, Y[i])
        # 3. Backward pass: compute all gradients
        dW1, db1_grad, dW2, db2_grad = backward(
            np.array([Y[i]]), cache, W2
        )
        # 4. Gradient descent update: theta <- theta - alpha * grad
        W1 -= lr * dW1
        b1 -= lr * db1_grad
        W2 -= lr * dW2
        b2 -= lr * db2_grad
    # Average loss over the 4 examples
    losses.append(epoch_loss / len(X))

# Results
print("Final predictions:")
print(f"{'Input':>10s}  {'Target':>6s}  {'Predicted':>9s}")
for i in range(len(X)):
    y_hat, _ = forward(X[i], W1, b1, W2, b2)
    print(f"  {X[i]}     {Y[i]:.0f}      {y_hat:.4f}")