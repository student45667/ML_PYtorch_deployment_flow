import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import subprocess
import os

# Load the nonlinear data
df = pd.read_csv('simple_nonlinear.csv')
x_train = torch.tensor(df[['x1', 'x2']].values, dtype=torch.float32)
y_train = torch.tensor(df[['output']].values, dtype=torch.float32)

print(f"Data shape: x={x_train.shape}  y={y_train.shape}")

# Model WITH hidden layer (can learn nonlinear functions)
class NonlinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = NonlinearModel()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train
for epoch in range(1000):
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"epoch {epoch:4d}  loss={loss.item():.6f}")

print("Done training")

# Save
torch.save(model.state_dict(), 'nonlinear_model.pth')
print("Saved: nonlinear_model.pth")

# Test
with torch.no_grad():
    test = torch.tensor([[5.0, 5.0]])
    pred = model(test).item()
    print(f"Test [x1=5, x2=5]: {pred:.4f}")



# ============================================================
# EXPORT TO ONNX (for Pi inference)
# ============================================================

# 1. Save normalization stats
x_mean = x_train.mean(dim=0)
x_std = x_train.std(dim=0)

import numpy as np
np.savez('norm_stats.npz',
         mean=x_mean.numpy(),
         std=x_std.numpy())
print("\nSaved: norm_stats.npz")
print(f"  Mean: {x_mean.tolist()}")
print(f"  Std:  {x_std.tolist()}")

# 2. Export to ONNX
print("\nExporting to ONNX...")
model.eval()  # <-- SET TO EVAL MODE FIRST


# Export simple
torch.onnx.export(
    model,
    torch.randn(1, 2),
    'nonlinear_model.onnx',
    input_names=['input'],
    output_names=['output'],
    opset_version=18
)




print("Saved: nonlinear_model.onnx")

# 3. Verify ONNX works
print("\nVerifying ONNX model...")
import onnxruntime as ort

sess = ort.InferenceSession('nonlinear_model.onnx')

# Test ONNX against PyTorch
test_raw = np.array([[5.0, 5.0]], dtype=np.float32)
test_norm = (test_raw - x_mean.numpy()) / x_std.numpy()

# PyTorch prediction
with torch.no_grad():
    torch_pred = model(torch.tensor(test_norm)).item()

# ONNX prediction
onnx_pred = sess.run(None, {'input': test_norm})[0][0][0]

print(f"  PyTorch: {torch_pred:.6f}")
print(f"  ONNX:    {onnx_pred:.6f}")
print(f"  Match:   {abs(torch_pred - onnx_pred) < 1e-5}")

print("\n" + "="*60)
print("Ready for Pi deployment!")

"""
print("="*60)
print("\nCopy to Pi:")
print("  scp nonlinear_model.onnx pi@raspi.local:~/")
print("  scp norm_stats.npz pi@raspi.local:~/")
print("  scp simple_nonlinear.csv pi@raspi.local:~/")
"""













# ============================================================
# CREATE EXAMPLE FILE FOR PI
# ============================================================

pi_file_content="""
#!/usr/bin/env python3
# onnx_minimal.py
# Simplest possible ONNX inference on Pi

import onnxruntime as ort
import numpy as np
import pandas as pd

# Load model
sess = ort.InferenceSession('nonlinear_model.onnx')

# Load norm stats
stats = np.load('norm_stats.npz')
x_mean = stats['mean']
x_std = stats['std']

# Single prediction
x_raw = np.array([[5.0, 5.0]], dtype=np.float32)
x_norm = (x_raw - x_mean) / x_std

result = sess.run(None, {'input': x_norm})
print(f"Input: {x_raw[0]} -> Prediction: {result[0][0][0]:.4f}")


# ============================================================

print("All predictions run and save csv")
import onnxruntime as ort
import numpy as np
import pandas as pd

sess = ort.InferenceSession('nonlinear_model.onnx')
stats = np.load('norm_stats.npz')

df = pd.read_csv('simple_nonlinear.csv')
x = df[['x1', 'x2']].values.astype(np.float32)
x_norm = (x - stats['mean']) / stats['std']

preds = []
for i in range(len(x_norm)):
    pred = sess.run(None, {'input': x_norm[i:i+1]})[0][0][0]
    preds.append(pred)

results = pd.DataFrame({'x1': df['x1'], 'x2': df['x2'], 'prediction': preds})
results.to_csv('predictions.csv', index=False)
print(f"Saved predictions.csv ({len(results)} rows)")
print(results)


# ============================================================
# History context loop feeeding back the model
# ============================================================

#import numpy as np
#import onnxruntime as ort
from collections import deque

# Load model
sess = ort.InferenceSession('nonlinear_model.onnx')
stats = np.load('norm_stats.npz')
x_mean = stats['mean']
x_std = stats['std']

# History buffer
history = deque(maxlen=10)  # Keep last 10 predictions

# Initial input
x1, x2 = 5.0, 5.0
print(f"Starting input: [{x1}, {x2}]")
print("Loop (feeding output back as input):")

for step in range(20):
    # Normalize
    x_raw = np.array([[x1, x2]], dtype=np.float32)
    x_norm = (x_raw - x_mean) / x_std
    
    # Predict
    result = sess.run(None, {'input': x_norm})
    pred = result[0][0][0]
    
    # Store in history
    history.append(pred)
    
    # Use prediction as next input (feedback loop)
    # Map output back to input space
    x1 = pred
    x2 = pred + step # slight variation
    
    print(f"Step {step:2d}: pred={pred:.4f}  history={[f'{h:.3f}' for h in list(history)]}")

print(f"Final history: {list(history)}")
print(f"Trend: {np.mean(list(history)):.4f}")


# ============================================================




"""

with open('ml_onnx_minimal_for_pi.py', 'w') as f:
    f.write(pi_file_content)


# ============================================================
# COPY FILES TO PI
# ============================================================

print("Copying files to Pi...")
subprocess.run("sshpass -p pi scp   nonlinear_model.onnx        pi@raspi.local:~/", shell=True)
subprocess.run("sshpass -p pi scp   nonlinear_model.onnx.data   pi@raspi.local:~/", shell=True)
subprocess.run("sshpass -p pi scp   norm_stats.npz              pi@raspi.local:~/", shell=True)
subprocess.run("sshpass -p pi scp   simple_nonlinear.csv        pi@raspi.local:~/", shell=True)
subprocess.run("sshpass -p pi scp   ml_onnx_minimal_for_pi.py   pi@raspi.local:~/", shell=True)
print("Done!")


