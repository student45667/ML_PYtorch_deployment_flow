# PyTorch Linear Regression: Complete Learning Guide

**Level:** Beginner  
**Topic:** Neural network training, loss functions, optimization  
**Duration:** 10–15 minutes to read + understand

---

## Overview: What Your Code Does

Your code trains a simple neural network to learn a **linear relationship** from training data.

```
INPUT: 3 features (3 numbers)
       ↓
    [LINEAR LAYER: multiply by weights + bias]
       ↓
OUTPUT: 1 prediction (1 number)

Training: Learn weights so predictions match actual values
Testing: Use learned weights to predict on new data
```

---

## Step-by-Step Breakdown

### Section 1: Import Libraries

```python
import torch
import torch.nn as nn
import torch.optim as optim
```

| Import | Purpose |
|--------|---------|
| `torch` | Core PyTorch library (tensors, operations) |
| `torch.nn` | Neural network layers (Linear, Conv2D, etc.) |
| `torch.optim` | Optimization algorithms (SGD, Adam, etc.) |

---

### Section 2: Define the Model (Your Neural Network)

```python
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer = nn.Linear(3, 1)
    
    def forward(self, x):
        return self.layer(x)
```

#### Line-by-Line:

```python
class SimpleModel(nn.Module):
```
- **What:** Create a custom model class that inherits from `nn.Module`
- **Why:** PyTorch models must inherit from `nn.Module` to work with optimizers & training loops
- **Alternative:** Use `nn.Sequential()` for simple stacked models

```python
def __init__(self):
    super(SimpleModel, self).__init__()
```
- **What:** Constructor method called when you create `SimpleModel()`
- **super():** Call parent class (`nn.Module`) initialization
- **Why needed:** Registers the model properly with PyTorch

```python
self.layer = nn.Linear(3, 1)
```
- **What:** Create a linear layer (fully connected layer)
- **Parameters:**
  - `3` = input features (3 numbers per sample)
  - `1` = output features (1 prediction per sample)
- **What it does:** Creates weights `W` (3×1 matrix) and bias `b` (scalar)
  - Formula: `output = input × W + b`
- **Learnable?** Yes — weights & bias are updated during training

**Alternative Linear Layers:**

| Layer | Code | Purpose |
|-------|------|---------|
| Linear | `nn.Linear(3, 1)` | Fully connected layer (weights + bias) |
| Conv1D | `nn.Conv1D(in_channels, out_channels)` | 1D convolution (for sequences) |
| Conv2D | `nn.Conv2D(in_channels, out_channels)` | 2D convolution (for images) |

```python
def forward(self, x):
    return self.layer(x)
```
- **What:** Define what happens when you call `model(x)`
- **Input:** `x` = input tensor (your data)
- **Output:** Result of passing `x` through the linear layer
- **Formula:** `return x @ W + b` (matrix multiplication + bias)

**Alternative forward implementations:**

```python
# With activation function (adds nonlinearity)
def forward(self, x):
    x = self.layer(x)
    x = torch.relu(x)  # ReLU activation
    return x

# Multiple layers (deeper network)
def forward(self, x):
    x = self.layer1(x)
    x = torch.relu(x)
    x = self.layer2(x)
    return x

# With dropout (regularization)
def forward(self, x):
    x = self.layer(x)
    x = torch.dropout(x, p=0.5, training=self.training)
    return x
```

---

### Section 3: Initialize Model, Loss, Optimizer

```python
model = SimpleModel()
```
- **What:** Create an instance of your model
- **Result:** Model with random initial weights & bias
- **Weights:** Randomly initialized (will be tuned during training)

---

```python
loss_fn = nn.MSELoss()
```
- **What:** Define the loss function (how to measure prediction error)
- **Formula:** `loss = mean((y_pred - y_true)²)`
- **Purpose:** Tells optimizer how badly predictions are performing
- **Why squared?** Penalizes large errors more than small ones

**Alternative Loss Functions:**

| Loss | Code | Use Case |
|------|------|----------|
| **MSELoss** | `nn.MSELoss()` | Regression (predicting continuous values like price) |
| **MAELoss** | `nn.L1Loss()` | Regression (less sensitive to outliers) |
| **CrossEntropyLoss** | `nn.CrossEntropyLoss()` | Classification (predicting class labels) |
| **BCELoss** | `nn.BCELoss()` | Binary classification (0 or 1) |
| **HuberLoss** | `nn.HuberLoss()` | Robust regression |

**Comparison:**

```python
# Your data
y_true = [3, 5, 7, 8]
y_pred = [2.9, 5.1, 6.8, 8.2]

# MSELoss
mse = MSELoss()
loss = mse(y_pred, y_true)
# Computes: mean((2.9-3)² + (5.1-5)² + (6.8-7)² + (8.2-8)²)
# = mean([0.01, 0.01, 0.04, 0.04])
# = 0.025

# MAELoss (L1Loss)
mae = MAELoss()
loss = mae(y_pred, y_true)
# Computes: mean(|2.9-3| + |5.1-5| + |6.8-7| + |8.2-8|)
# = mean([0.1, 0.1, 0.2, 0.2])
# = 0.15
```

---

```python
optimizer = optim.SGD(model.parameters(), lr=0.01)
```
- **What:** Choose optimization algorithm (how to update weights)
- **`model.parameters()`:** All learnable weights & biases in the model
- **`lr=0.01`:** Learning rate (step size for weight updates)
  - Large LR: Fast training but may overshoot
  - Small LR: Slow training but more precise

**How it works:**
```
For each parameter:
  new_weight = old_weight - lr * gradient
                                ↑
                           Direction to improve
```

**Alternative Optimizers:**

| Optimizer | Code | Characteristics |
|-----------|------|---|
| **SGD** | `optim.SGD(lr=0.01)` | Simple, fast, may get stuck |
| **Adam** | `optim.Adam(lr=0.001)` | Adaptive learning rate, faster convergence |
| **RMSprop** | `optim.RMSprop(lr=0.01)` | Handles non-stationary problems |
| **Adagrad** | `optim.Adagrad(lr=0.01)` | Good for sparse data |

**Comparison on your data:**

```python
# SGD: Simple but slow
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Adam: Faster, automatically adjusts learning rate
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Often converges in 50 epochs instead of 100

# RMSprop: Middle ground
optimizer = optim.RMSprop(model.parameters(), lr=0.01)
```

---

### Section 4: Prepare Training Data

```python
x_train = torch.tensor(
    [[1,1,1], [2,2,2], [3,3,3], [4,4,4], [4,4,7]], 
    dtype=torch.float32
)
```

#### Shape Analysis:

```
[[1,1,1],
 [2,2,2],
 [3,3,3],
 [4,4,4],
 [4,4,7]]

Shape: (5, 3)
  ↑     ↑   ↑
  |     |   └─ 3 features per sample
  |     └───── 5 samples (5 training examples)
  └─────────── 2D tensor (batch + features)
```

#### Data Interpretation:

| Sample | Feature 1 | Feature 2 | Feature 3 | Purpose |
|--------|-----------|-----------|-----------|---------|
| 1 | 1 | 1 | 1 | Training example 1 |
| 2 | 2 | 2 | 2 | Training example 2 |
| 3 | 3 | 3 | 3 | Training example 3 |
| 4 | 4 | 4 | 4 | Training example 4 |
| 5 | 4 | 4 | 7 | Training example 5 |

```python
y_train = torch.tensor(
    [[3], [5], [7], [8], [8]], 
    dtype=torch.float32
)
```

#### Shape Analysis:

```
[[3],
 [5],
 [7],
 [8],
 [8]]

Shape: (5, 1)
  ↑     ↑   ↑
  |     |   └─ 1 output value per sample
  |     └───── 5 samples (matches x_train)
  └─────────── 2D tensor (required by PyTorch)
```

#### Data Relationship:

```
Sample | x_train         | y_train | Relationship
-------|-----------------|---------|---------------
1      | [1, 1, 1]       | [3]     | ???
2      | [2, 2, 2]       | [5]     | Pattern: y = 2*x + 1?
3      | [3, 3, 3]       | [7]     | (where x = avg of features)
4      | [4, 4, 4]       | [8]     | ???
5      | [4, 4, 7]       | [8]     | Outlier (breaks pattern)
```

**Alternative ways to create tensors:**

```python
# From Python list (what you're doing)
x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

# Random tensor
x = torch.randn(5, 3)  # 5 samples, 3 features, random values

# Zeros or ones
x = torch.zeros(5, 3)
x = torch.ones(5, 3)

# From NumPy array
import numpy as np
arr = np.array([[1, 2, 3], [4, 5, 6]])
x = torch.from_numpy(arr)

# Linspace (evenly spaced)
x = torch.linspace(0, 10, 5)  # [0, 2.5, 5, 7.5, 10]
```

---

### Section 5: Training Loop

```python
num_epochs = 100
for epoch in range(num_epochs):
```
- **What:** Repeat the training process 100 times
- **Epoch:** One complete pass through all training data
- **Why multiple?** Model improves gradually, needs multiple iterations

---

```python
    y_pred = model(x_train)
```
- **What:** Forward pass — predict outputs for all training inputs
- **Input:** `x_train` (5, 3) — 5 samples with 3 features each
- **Process:** Each sample goes through the linear layer
  - `sample[0] @ weights + bias → prediction[0]`
  - `sample[1] @ weights + bias → prediction[1]`
  - ... (5 total)
- **Output:** `y_pred` (5, 1) — 5 predictions

**What's happening inside:**

```
Model weights after random init:
  W = [[0.5],     (3×1 matrix)
       [0.3],
       [-0.2]]
  b = 0.1         (scalar)

For sample x = [1, 1, 1]:
  y_pred = [1, 1, 1] @ [[0.5], [0.3], [-0.2]] + 0.1
         = 1*0.5 + 1*0.3 + 1*(-0.2) + 0.1
         = 0.5 + 0.3 - 0.2 + 0.1
         = 0.7  (not correct, will improve during training)
```

---

```python
    loss = loss_fn(y_pred, y_train)
```
- **What:** Compare predictions to actual values
- **Formula:** `loss = mean((y_pred - y_train)²)`
- **Example:**
  ```
  y_pred = [0.7, 1.8, 3.2, 4.1, 4.0]
  y_train = [3, 5, 7, 8, 8]
  
  differences = [0.7-3, 1.8-5, 3.2-7, 4.1-8, 4.0-8]
              = [-2.3, -3.2, -3.8, -3.9, -4.0]
  
  squared = [5.29, 10.24, 14.44, 15.21, 16.0]
  loss = mean = 12.236
  ```
- **Interpretation:** Lower loss = better predictions

---

```python
    optimizer.zero_grad()
```
- **What:** Clear old gradients (start fresh)
- **Why?** PyTorch accumulates gradients by default
- **Without this:** Gradients from epoch 1, 2, 3... would add up and be wrong
- **Analogy:** Clear whiteboard before writing new calculations

---

```python
    loss.backward()
```
- **What:** Backpropagation — compute how much to change each weight
- **How:** Calculate gradient of loss with respect to each weight
  - "If I increase weight by 0.01, how does loss change?"
  - "If I decrease weight by 0.01, how does loss change?"
- **Automatically computes:**
  - Gradient of W[0] (first weight)
  - Gradient of W[1] (second weight)
  - Gradient of W[2] (third weight)
  - Gradient of b (bias)

**Simplified example:**

```
Current loss = 12.236

If we change W[0]:
  loss with W[0] + 0.001 = 12.234  (slightly better)
  Gradient direction: negative (decrease W[0])

If we change W[1]:
  loss with W[1] + 0.001 = 12.200  (much better)
  Gradient direction: even more negative (decrease W[1])

backward() computes all these gradients
```

---

```python
    optimizer.step()
```
- **What:** Update all weights using the gradients
- **Formula:** `weight = weight - lr * gradient`
- **Example (for one weight):**
  ```
  Gradient of W[0] = -5.2  (negative means decrease helps)
  W[0] = 0.5 - 0.01 * (-5.2)  = 0.5 + 0.052 = 0.552
  ```
- **Effect:** All weights move slightly in the "good direction"

**What happens over 100 epochs:**

```
Epoch 1:  W = [0.5, 0.3, -0.2],  loss = 12.236
Epoch 2:  W = [0.52, 0.35, -0.15],  loss = 10.8  (improved!)
Epoch 3:  W = [0.54, 0.40, -0.10],  loss = 9.5
...
Epoch 100: W = [1.0, 1.0, 0.33],  loss = 0.001  (very good!)
```

---

### Section 6: Testing (Inference)

```python
with torch.no_grad():
```
- **What:** Disable gradient computation for testing
- **Why?** Testing doesn't need gradients (we're not training)
- **Benefit:** Faster, uses less memory

---

```python
    test_input = torch.tensor([[3, 1, 2]], dtype=torch.float32)
```
- **What:** Create a new input (data the model hasn't seen)
- **Shape:** (1, 3) — 1 sample with 3 features
- **Values:** [3, 1, 2] — some arbitrary features

---

```python
    prediction = model(test_input)
```
- **What:** Forward pass on new data
- **Input:** [3, 1, 2]
- **Process:**
  ```
  prediction = [3, 1, 2] @ learned_weights + learned_bias
             = 3*w0 + 1*w1 + 2*w2 + b
             = (some number based on learned pattern)
  ```
- **Output:** A single number prediction

---

```python
    print(f"Prediction Final: {prediction.detach().numpy().round(1)}")
```

#### Breaking down the chain:

```python
prediction  # PyTorch tensor: tensor([[5.23]])

.detach()   # Detach from computation graph
            # Result: tensor([[5.23]]) (still a tensor)

.numpy()    # Convert to NumPy array
            # Result: array([[5.23]]) (NumPy format)

.round(1)   # Round to 1 decimal place
            # Result: array([[5.2]])

f"..."      # Format in f-string
            # Result: "Prediction Final: [[5.2]]"
```

**Alternative ways to print:**

```python
# Option 1: Just the tensor (shows PyTorch format)
print(prediction)
# Output: tensor([[5.2345]])

# Option 2: Convert to Python float
print(prediction.item())
# Output: 5.234500408172607

# Option 3: Round and convert
print(f"{prediction.item():.1f}")
# Output: 5.2

# Option 4: Convert to list
print(prediction.tolist())
# Output: [[5.234500408172607]]

# Option 5: NumPy array (your choice)
print(prediction.detach().numpy().round(1))
# Output: [[5.2]]
```

---

## Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ 1. SETUP                                                    │
│    ├─ Define model (nn.Linear)                             │
│    ├─ Choose loss function (MSELoss)                       │
│    └─ Choose optimizer (SGD)                               │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. TRAINING LOOP (100 epochs)                              │
│    ┌─────────────────────────────────────────────────────┐ │
│    │ For each epoch:                                      │ │
│    │  ① Forward pass: y_pred = model(x_train)            │ │
│    │  ② Calculate loss: loss = MSELoss(y_pred, y_train)  │ │
│    │  ③ Clear gradients: optimizer.zero_grad()           │ │
│    │  ④ Backprop: loss.backward() ← compute gradients    │ │
│    │  ⑤ Update: optimizer.step() ← apply gradients       │ │
│    │                                                       │ │
│    │  Result: Weights improve slightly each epoch         │ │
│    └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. TESTING (after training)                                │
│    ├─ Use learned weights (no gradient updates)            │
│    ├─ Make prediction on new data                          │
│    └─ Print result                                         │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Concepts Summary

| Concept | What It Is | Analogy |
|---------|-----------|---------|
| **Model** | Neural network with learnable weights | Student's brain learning |
| **Loss Function** | Measures prediction error | Report card grading |
| **Optimizer** | Updates weights to reduce loss | Teacher giving feedback |
| **Forward Pass** | Input → Model → Prediction | Student answering test |
| **Backward Pass** | Loss → Gradients → How to improve | Student reviewing mistakes |
| **Gradient** | Direction to move weights | Arrow pointing toward improvement |
| **Learning Rate** | Step size for weight updates | How big to adjust answer |
| **Epoch** | One complete pass through data | One study session |

---

## Common Issues & Solutions

### Issue 1: Loss Not Decreasing

```python
# Symptom: loss = 10.5, 10.4, 10.4, 10.5, 10.6 (bouncing)

# Solution 1: Lower learning rate
optimizer = optim.SGD(model.parameters(), lr=0.001)  # was 0.01

# Solution 2: Use Adam (auto-adjusts learning rate)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Solution 3: Check data (may be noisy/inconsistent)
print(x_train, y_train)  # Do they have a clear relationship?
```

### Issue 2: Shape Mismatch Error

```python
# Symptom: RuntimeError: expected scalar type Float but found Int

# Solution: Ensure dtype=torch.float32
x_train = torch.tensor(..., dtype=torch.float32)
y_train = torch.tensor(..., dtype=torch.float32)
```

### Issue 3: Model Not Learning Anything

```python
# Symptom: Weights never change

# Solution 1: Check if backward() is called
loss.backward()  # Must have this

# Solution 2: Check if optimizer.step() is called
optimizer.step()  # Must have this

# Solution 3: Verify loss is computed from trainable weights
y_pred = model(x_train)  # Use model, not hardcoded values
loss = loss_fn(y_pred, y_train)  # Must be from y_pred
```

---

## Extensions & Next Steps

### 1. Add Activation Functions (Nonlinearity)

```python
class BetterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(3, 16)      # 3 inputs → 16 hidden units
        self.layer2 = nn.Linear(16, 1)      # 16 → 1 output
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))      # Add nonlinearity
        x = self.layer2(x)
        return x
```

### 2. Use DataLoader (for larger datasets)

```python
from torch.utils.data import DataLoader, TensorDataset

# Combine x and y
dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    for x_batch, y_batch in dataloader:  # Mini-batches
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 3. Add Validation Set

```python
# Split into train/val
val_x = x_train[-1:]
val_y = y_train[-1:]
train_x = x_train[:-1]
train_y = y_train[:-1]

# Training loop
for epoch in range(num_epochs):
    # Train
    y_pred = model(train_x)
    loss = loss_fn(y_pred, train_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Validate
    with torch.no_grad():
        val_pred = model(val_x)
        val_loss = loss_fn(val_pred, val_y)
    
    if epoch % 10 == 0:
        print(f"Train loss: {loss:.4f}, Val loss: {val_loss:.4f}")
```

### 4. Save & Load Model

```python
# Save after training
torch.save(model.state_dict(), 'model.pth')

# Load later
model_loaded = SimpleModel()
model_loaded.load_state_dict(torch.load('model.pth'))
model_loaded.eval()
```

---

## Summary

Your code trains a simple neural network in 6 steps:

1. **Define** the model structure (Linear layer: 3 inputs → 1 output)
2. **Choose** loss function (MSELoss) and optimizer (SGD)
3. **Prepare** training data (5 samples, 3 features each)
4. **Train** for 100 epochs (forward → loss → backprop → update)
5. **Test** on new data using learned weights
6. **Print** the prediction

The key insight: **Weights start random, then gradually improve through repeated error measurement + gradient-based adjustment.**

---

**Next: Try modifying learning rate, number of epochs, or model architecture to see how it affects training!**
