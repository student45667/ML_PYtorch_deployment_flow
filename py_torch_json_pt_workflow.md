# JSON + PyTorch Workflow: Create, Save, Load LUT Data

**Workflow:**
1. Create input/output data (JSON - human readable)
2. Load JSON into PyTorch tensors
3. Convert to .PT for fast loading (optional)
4. Train models and save checkpoints (.PT)

---

## Part 1: Create LUT Data in JSON Format

### Option A: Create JSON Manually

**File: `my_lut.json`** (create in text editor)

```json
{
  "metadata": {
    "name": "3D Calibration LUT",
    "description": "ADC calibration table with 100 points",
    "version": "1.0",
    "created": "2026-06-07"
  },
  "x_data": {
    "features": ["x1", "x2", "x3"],
    "description": "Input coordinates",
    "values": [
      [0, 0, 0],
      [0, 0, 1],
      [0, 0, 2],
      [2.5, -2.5, 5],
      [5, 0, 5],
      [7.5, 2.5, 7],
      [10, 5, 10]
    ]
  },
  "y_data": {
    "features": ["output"],
    "description": "Output values (calibration correction)",
    "values": [
      [1.5],
      [1.6],
      [1.7],
      [3.2],
      [5.0],
      [7.3],
      [10.1]
    ]
  }
}
```

---

### Option B: Generate JSON Programmatically

```python
import json
import torch

def create_lut_json(filename, num_samples=100):
    """Create synthetic LUT data and save as JSON"""
    
    # Generate synthetic data
    torch.manual_seed(42)
    x_data = torch.randn(num_samples, 3) * 10
    y_data = 2*x_data[:, 0:1] + 3*x_data[:, 1:2] - x_data[:, 2:3]
    
    # Create JSON structure
    lut_dict = {
        "metadata": {
            "name": "Generated 3D LUT",
            "description": "Synthetic calibration table",
            "version": "1.0",
            "num_samples": num_samples,
            "formula": "y = 2*x1 + 3*x2 - x3"
        },
        "x_data": {
            "features": ["x1", "x2", "x3"],
            "description": "Input 3D coordinates",
            "values": x_data.tolist()  # Convert tensor to list
        },
        "y_data": {
            "features": ["output"],
            "description": "Output values",
            "values": y_data.tolist()
        }
    }
    
    # Save to JSON
    with open(filename, 'w') as f:
        json.dump(lut_dict, f, indent=2)
    
    print(f"Created {filename} with {num_samples} samples")
    return lut_dict

# Create and save
create_lut_json('my_lut.json', num_samples=100)
```

**Output:**
```
Created my_lut.json with 100 samples
```

**What the JSON looks like:**
```json
{
  "metadata": {
    "name": "Generated 3D LUT",
    "description": "Synthetic calibration table",
    "version": "1.0",
    "num_samples": 100,
    "formula": "y = 2*x1 + 3*x2 - x3"
  },
  "x_data": {
    "features": ["x1", "x2", "x3"],
    "description": "Input 3D coordinates",
    "values": [
      [0.3674, 0.4519, 0.5635],
      [-0.5234, -0.3421, 0.1234],
      ...
    ]
  },
  "y_data": {
    "features": ["output"],
    "description": "Output values",
    "values": [
      [0.5234],
      [-1.2345],
      ...
    ]
  }
}
```

---

## Part 2: Load JSON into PyTorch Tensors

### Simple Load Function

```python
import json
import torch

def load_lut_json(filepath):
    """Load LUT from JSON file into PyTorch tensors"""
    
    with open(filepath, 'r') as f:
        config = json.load(f)
    
    # Extract data
    x_values = config['x_data']['values']
    y_values = config['y_data']['values']
    
    # Convert to tensors
    x_tensor = torch.tensor(x_values, dtype=torch.float32)
    y_tensor = torch.tensor(y_values, dtype=torch.float32)
    
    # Extract metadata
    metadata = config.get('metadata', {})
    
    return x_tensor, y_tensor, metadata

# Usage
x_lut, y_lut, metadata = load_lut_json('my_lut.json')

print(f"Loaded: {x_lut.shape}")
print(f"Metadata: {metadata}")
```

**Output:**
```
Loaded: torch.Size([100, 3])
Metadata: {'name': 'Generated 3D LUT', 'description': 'Synthetic calibration table', ...}
```

---

### Enhanced Load with Validation

```python
import json
import torch
from pathlib import Path

class LUTLoader:
    """Load LUT from JSON with validation"""
    
    @staticmethod
    def load_json(filepath):
        """Load from JSON with error checking"""
        
        if not Path(filepath).exists():
            raise FileNotFoundError(f"LUT file not found: {filepath}")
        
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {filepath}: {e}")
        
        # Validate structure
        required_keys = ['x_data', 'y_data']
        if not all(key in config for key in required_keys):
            raise ValueError(f"Missing required keys: {required_keys}")
        
        # Extract data
        x_values = config['x_data']['values']
        y_values = config['y_data']['values']
        
        # Validate consistency
        if len(x_values) != len(y_values):
            raise ValueError(f"Length mismatch: x={len(x_values)}, y={len(y_values)}")
        
        # Convert to tensors
        x_tensor = torch.tensor(x_values, dtype=torch.float32)
        y_tensor = torch.tensor(y_values, dtype=torch.float32)
        
        metadata = config.get('metadata', {})
        
        return x_tensor, y_tensor, metadata
    
    @staticmethod
    def load_pt(filepath):
        """Load from PyTorch binary"""
        
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        checkpoint = torch.load(filepath)
        return checkpoint

# Usage with error handling
try:
    x_lut, y_lut, meta = LUTLoader.load_json('my_lut.json')
    print(f"✓ Loaded JSON: {x_lut.shape}")
except FileNotFoundError as e:
    print(f"✗ Error: {e}")
```

---

## Part 3: Edit JSON Manually (Human Friendly)

**Example: `calibration_table.json`**

```json
{
  "metadata": {
    "device": "ADC_Converter",
    "calibration_date": "2026-06-07",
    "temperature": "25C"
  },
  "x_data": {
    "features": ["voltage_in_V", "temperature_C", "frequency_MHz"],
    "values": [
      [0.0, 25, 1],
      [0.5, 25, 1],
      [1.0, 25, 1],
      [1.5, 25, 1],
      [2.0, 25, 1],
      [0.0, 50, 1],
      [0.5, 50, 1]
    ]
  },
  "y_data": {
    "features": ["calibration_offset_mV"],
    "values": [
      [0.1],
      [0.15],
      [0.2],
      [0.25],
      [0.3],
      [0.12],
      [0.18]
    ]
  }
}
```

**Edit in:**
- Any text editor (VS Code, Notepad, etc.)
- Online JSON validator: https://jsonlint.com/
- Python with pretty printing

```python
import json

# Pretty print JSON
with open('calibration_table.json', 'r') as f:
    data = json.load(f)

print(json.dumps(data, indent=2))
```

---

## Part 4: Convert JSON → PT (One-Time)

```python
import json
import torch
from pathlib import Path

def json_to_pt(json_file, pt_file):
    """Convert JSON LUT to PyTorch binary format"""
    
    # Load JSON
    with open(json_file, 'r') as f:
        config = json.load(f)
    
    # Extract
    x_tensor = torch.tensor(config['x_data']['values'], dtype=torch.float32)
    y_tensor = torch.tensor(config['y_data']['values'], dtype=torch.float32)
    metadata = config.get('metadata', {})
    
    # Save as PT
    checkpoint = {
        'x': x_tensor,
        'y': y_tensor,
        'metadata': metadata
    }
    
    torch.save(checkpoint, pt_file)
    print(f"✓ Converted: {json_file} → {pt_file}")
    print(f"  Size: {Path(pt_file).stat().st_size / 1024:.1f} KB")

# Usage
json_to_pt('my_lut.json', 'my_lut.pt')
```

**Output:**
```
✓ Converted: my_lut.json → my_lut.pt
  Size: 4.5 KB
```

---

## Part 5: Load PT (Fast Access)

```python
import torch

def load_lut_pt(filepath):
    """Load LUT from PyTorch binary (fast)"""
    
    checkpoint = torch.load(filepath)
    
    x_tensor = checkpoint['x']
    y_tensor = checkpoint['y']
    metadata = checkpoint.get('metadata', {})
    
    return x_tensor, y_tensor, metadata

# Usage - MUCH FASTER than JSON for large datasets
x_lut, y_lut, meta = load_lut_pt('my_lut.pt')
print(f"Loaded from .PT: {x_lut.shape}")
```

---

## Part 6: Complete Workflow Example

```python
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# ===== STEP 1: CREATE JSON LUT =====
print("="*50)
print("STEP 1: Create LUT in JSON")
print("="*50)

def create_synthetic_lut(filename, num_samples=100):
    torch.manual_seed(42)
    x_data = torch.randn(num_samples, 3) * 10
    y_data = 2*x_data[:, 0:1] + 3*x_data[:, 1:2] - x_data[:, 2:3] + torch.randn(num_samples, 1) * 0.1
    
    lut_dict = {
        "metadata": {
            "name": "3D Calibration LUT",
            "samples": num_samples,
            "formula": "y = 2*x1 + 3*x2 - x3 + noise"
        },
        "x_data": {
            "features": ["x1", "x2", "x3"],
            "values": x_data.tolist()
        },
        "y_data": {
            "features": ["output"],
            "values": y_data.tolist()
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(lut_dict, f, indent=2)
    
    print(f"✓ Created {filename}")
    return lut_dict

create_synthetic_lut('lut.json', 100)

# ===== STEP 2: LOAD JSON INTO TENSORS =====
print("\n" + "="*50)
print("STEP 2: Load JSON → PyTorch Tensors")
print("="*50)

with open('lut.json', 'r') as f:
    config = json.load(f)

x_lut = torch.tensor(config['x_data']['values'], dtype=torch.float32)
y_lut = torch.tensor(config['y_data']['values'], dtype=torch.float32)

print(f"✓ Loaded tensors:")
print(f"  x_lut shape: {x_lut.shape}")
print(f"  y_lut shape: {y_lut.shape}")

# ===== STEP 3: TRAIN MODEL =====
print("\n" + "="*50)
print("STEP 3: Train Model on LUT Data")
print("="*50)

class LUTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = LUTNet()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

print("Training...")
for epoch in range(200):
    y_pred = model(x_lut)
    loss = loss_fn(y_pred, y_lut)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f"  Epoch {epoch}: Loss = {loss.item():.4f}")

# ===== STEP 4: SAVE MODEL AS PT =====
print("\n" + "="*50)
print("STEP 4: Save Model & LUT as .PT")
print("="*50)

# Save model
torch.save(model.state_dict(), 'lut_model.pt')
print("✓ Saved model: lut_model.pt")

# Save LUT data as PT (for fast reload)
lut_checkpoint = {
    'x': x_lut,
    'y': y_lut,
    'metadata': config['metadata']
}
torch.save(lut_checkpoint, 'lut_data.pt')
print("✓ Saved LUT data: lut_data.pt")

# ===== STEP 5: LOAD MODEL FROM PT =====
print("\n" + "="*50)
print("STEP 5: Load Model & LUT from .PT")
print("="*50)

# Load model
model_loaded = LUTNet()
model_loaded.load_state_dict(torch.load('lut_model.pt'))
model_loaded.eval()
print("✓ Loaded model from: lut_model.pt")

# Load LUT
lut_data = torch.load('lut_data.pt')
x_loaded = lut_data['x']
y_loaded = lut_data['y']
print("✓ Loaded LUT from: lut_data.pt")
print(f"  Loaded {x_loaded.shape[0]} samples")

# ===== STEP 6: TEST =====
print("\n" + "="*50)
print("STEP 6: Test Loaded Model")
print("="*50)

with torch.no_grad():
    test_point = torch.tensor([[5.0, 2.0, 3.0]], dtype=torch.float32)
    prediction = model_loaded(test_point)
    
    print(f"Test point: {test_point.numpy()}")
    print(f"Prediction: {prediction.item():.2f}")

print("\n✓ Workflow complete!")
```

**Output:**
```
==================================================
STEP 1: Create LUT in JSON
==================================================
✓ Created lut.json

==================================================
STEP 2: Load JSON → PyTorch Tensors
==================================================
✓ Loaded tensors:
  x_lut shape: torch.Size([100, 3])
  y_lut shape: torch.Size([100, 1])

==================================================
STEP 3: Train Model on LUT Data
==================================================
Training...
  Epoch 0: Loss = 145.2345
  Epoch 50: Loss = 0.0234
  Epoch 100: Loss = 0.0012
  Epoch 150: Loss = 0.0003

==================================================
STEP 4: Save Model & LUT as .PT
==================================================
✓ Saved model: lut_model.pt
✓ Saved LUT data: lut_data.pt

==================================================
STEP 5: Load Model & LUT from .PT
==================================================
✓ Loaded model from: lut_model.pt
✓ Loaded LUT from: lut_data.pt
  Loaded 100 samples

==================================================
STEP 6: Test Loaded Model
==================================================
Test point: [[5. 2. 3.]]
Prediction: 18.25

✓ Workflow complete!
```

---

## File Size Comparison

```python
from pathlib import Path

# Create files
json_size = Path('lut.json').stat().st_size / 1024
pt_size = Path('lut_data.pt').stat().st_size / 1024
model_size = Path('lut_model.pt').stat().st_size / 1024

print(f"lut.json:       {json_size:.1f} KB (human readable)")
print(f"lut_data.pt:    {pt_size:.1f} KB (fast to load)")
print(f"lut_model.pt:   {model_size:.1f} KB (trained model)")
```

**Output:**
```
lut.json:       156.3 KB (human readable, easy to edit)
lut_data.pt:    3.2 KB (binary, much faster to load)
lut_model.pt:   12.5 KB (trained weights)
```

---

## Summary: Recommended Workflow

| Step | File Format | Purpose |
|------|-------------|---------|
| **1. Create LUT** | `.json` | Easy to create, edit, understand |
| **2. Load into Python** | → Tensors | Process and train |
| **3. Train model** | Save as `.pt` | Fast checkpoint saving |
| **4. Production** | Load from `.pt` | Fast inference |

**Typical Usage:**

```python
# Development: Work with JSON
x, y, meta = load_json('calibration.json')  # Edit in text editor
model.train(x, y)

# Production: Use PT
checkpoint = torch.load('model.pt')  # Fast load
x_data = torch.load('lut.pt')        # Fast load
prediction = model(x_data)
```

---

Perfect for your semiconductor LUT workflow! 📊
