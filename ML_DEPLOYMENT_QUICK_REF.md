# PyTorch ML Training & Deployment — Quick Reference

## Three-File Pipeline Overview

```
[ai.local: RTX 3070] 
        ↓
    [File 1] Train linear model (basics)
        ↓
    [File 2] Train nonlinear + export ONNX → Deploy Python script to Pi
        ↓
    [File 3] Train nonlinear + export C/C++ headers → Compile native exe on Pi
        ↓
    [Target: RPi, ESP32, ARM MCU]
```

---

## File 1: Basic PyTorch Training (`1pytorch.py`)

**Purpose:** Learn PyTorch fundamentals  
**Model:** Linear (1 layer, 2→1)  
**Data:** Synthetic CSV (`y = 2*x1 + 3*x2`)  
**Output:** Checkpoint file (`.pth`)  
**Deployment:** None  

### Key operations:
```python
# Create model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 1)

# Train loop
for epoch in range(3000):
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Save & load
torch.save(model.state_dict(), 'py_torch_model.pth')
model.load_state_dict(torch.load('py_torch_model.pth'))
```

**When to use:** Learning PyTorch, testing basic training loops

---

## File 2: Python ONNX Pipeline (`2pytorch_non_linear.py`)

**Purpose:** Complete workflow: train → export ONNX → deploy to Pi (Python)  
**Model:** Nonlinear (3 layers: 2→32→32→1 with ReLU)  
**Optimizer:** Adam  
**Training epochs:** 1000  
**Export format:** ONNX + normalization stats  
**Deployment target:** Pi with Python + ONNX Runtime  
**Auto-deploy:** Yes (sshpass)  

### Workflow:
```
1. Load CSV data
2. Train NonlinearModel (Adam, 1000 epochs)
3. Save model state dict
4. Compute & save normalization stats (mean, std)
5. Export to ONNX with torch.onnx.export()
6. Verify ONNX predictions match PyTorch
7. Generate minimal Pi inference script
8. Auto-copy files via sshpass to pi@raspi.local:~/
```

### Model architecture:
```python
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
```

### Output files:
- `nonlinear_model.onnx` (~50 KB) — Serialized model
- `norm_stats.npz` (~100 B) — Normalization constants
- `ml_onnx_minimal_for_pi.py` (~1 KB) — Inference script for Pi

### Pi inference (minimal):
```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession('nonlinear_model.onnx')
stats = np.load('norm_stats.npz')

x_raw = np.array([[5.0, 5.0]], dtype=np.float32)
x_norm = (x_raw - stats['mean']) / stats['std']

pred = sess.run(None, {'input': x_norm})[0][0][0]
print(f"Prediction: {pred:.4f}")
```

### Features:
- **History feedback loop** — feeds model outputs back as inputs (20 steps shown)
- **Batch prediction** — processes entire CSV and saves results
- **ONNX verification** — compares PyTorch vs ONNX output

**Latency per prediction:** ~0.1–1 ms  
**Binary footprint:** ~50 KB (ONNX file) + Python runtime  
**Dependencies on target:** Python, onnxruntime, numpy  

---

## File 3: C/C++ Native Pipeline (`3train_export_deploy_cpp.py`)

**Purpose:** Complete workflow: train → export C headers → compile → deploy (C/C++)  
**Model:** Nonlinear (3 layers: 2→32→32→1 with ReLU, same as File 2)  
**Optimizer:** Adam  
**Training epochs:** 1000  
**Export format:** C header file (model_weights.h) + C/C++ source (inference.c/cpp)  
**Deployment target:** Any device with C/C++ compiler (Pi, ESP32, ARM MCU)  
**Auto-deploy:** Yes (compile and test on Pi via SSH)  

### 9-step pipeline:
```
1. Load data & train model (PyTorch)
2. Save normalization stats
3. Export to ONNX (for verification)
4. Extract weights → model_weights.h (static float arrays)
5. Generate inference.cpp (C++ with matrix math + ReLU)
6. Generate inference.c (C equivalent)
7. Test locally (g++ -O2)
8. Deploy to Pi via sshpass
9. Compile & test on Pi (g++ and gcc)
```

### Generated C header (`model_weights.h`):
```c
static const float W1[2048] = { 0.1234f, -0.5678f, ... };
static const float B1[32] = { ... };
static const float W2[1024] = { ... };
static const float B2[32] = { ... };
static const float W3[32] = { ... };
static const float B3[1] = { ... };

static const float MEAN[2] = { 4.5f, 5.2f };
static const float STD[2] = { 2.3f, 1.8f };
```

### Core inference functions:
```c
void linear_relu(const float* W, const float* B, const float* in,
                 float* out, int M, int N) {
    for (int j = 0; j < M; j++) {
        float v = B[j];
        for (int i = 0; i < N; i++)
            v += W[j*N + i] * in[i];
        out[j] = (v > 0.0f) ? v : 0.0f;  // ReLU
    }
}

void linear(const float* W, const float* B, const float* in,
            float* out, int M, int N) {
    for (int j = 0; j < M; j++) {
        float v = B[j];
        for (int i = 0; i < N; i++)
            v += W[j*N + i] * in[i];
        out[j] = v;
    }
}

float predict(float x1, float x2) {
    float x[2] = {
        (x1 - MEAN[0]) / STD[0],
        (x2 - MEAN[1]) / STD[1]
    };
    
    float h1[32], h2[32], out[1];
    linear_relu(W1, B1, x, h1, 32, 2);
    linear_relu(W2, B2, h1, h2, 32, 32);
    linear(W3, B3, h2, out, 1, 32);
    
    return out[0];
}
```

### Output files:
- `model_weights.h` (~40 KB) — All weights as static arrays
- `inference.cpp` (~4 KB) — C++ inference code
- `inference.c` (~4 KB) — C inference code
- `inference_cpp` (~50 KB) — Compiled executable (C++)
- `inference_c` (~45 KB) — Compiled executable (C)

### Compilation on Pi:
```bash
# C++
g++ -O2 -o inference_cpp inference.cpp
./inference_cpp

# C (link math library)
gcc -O2 -o inference_c inference.c -lm
./inference_c

# With MCU optimization (ARM Cortex-A)
g++ -O3 -march=native -o inference_cpp inference.cpp
```

### Using in embedded project (ESP32):
```c
#include "model_weights.h"  // Static weights
#include "inference.c"       // inference() function

void setup() {
    Serial.begin(115200);
}

void loop() {
    float result = predict(5.0f, 5.0f);
    Serial.println(result);
    delay(1000);
}
```

### Features:
- **Zero framework overhead** — compiled-in weights, no dynamic allocation
- **Compile-time constants** — all weights baked into binary
- **Universal portability** — pure C, works on any platform with a C compiler
- **History feedback loop** — same 20-step demo as File 2

**Latency per prediction:** ~1–2 µs (no overhead, pure arithmetic)  
**Binary footprint:** ~45–50 KB (executable only, includes weights)  
**Dependencies on target:** None (just C stdlib)  

---

## Quick Comparison

| Aspect | File 1 | File 2 (Python) | File 3 (C/C++) |
|--------|--------|-----------------|----------------|
| **Model complexity** | Linear (basic) | Nonlinear (3 layers) | Nonlinear (3 layers) |
| **Export format** | None (.pth) | ONNX + stats | C headers + source |
| **Deployment** | None | Python script | Compiled executable |
| **Target** | Development only | Pi + Python | Any (C/C++ compiler) |
| **Latency** | N/A | ~0.1–1 ms | ~1–2 µs |
| **Binary size** | N/A | ~50 KB (ONNX) | ~45 KB (exe) |
| **Dependencies** | None | Python, onnxruntime, numpy | None |
| **Auto-deploy** | No | Yes (sshpass) | Yes (sshpass + compile) |
| **Best for** | Learning | Prototyping, debugging | Production, embedded |

---

## Deployment Checklist

### Path A: Python ONNX (File 2)

**Prerequisites:**
- [ ] Raspberry Pi with SSH access
- [ ] `sshpass` installed on dev machine
- [ ] Python 3.8+ on Pi
- [ ] `pip install onnxruntime numpy pandas` on Pi
- [ ] Hostname set to `raspi.local` or configure IP

**Deployment:**
- [ ] Run File 2 on ai.local
- [ ] Auto-deploy triggered by sshpass commands in File 2
- [ ] SSH to Pi and run `python ml_onnx_minimal_for_pi.py`

### Path B: C/C++ (File 3)

**Prerequisites:**
- [ ] Raspberry Pi with SSH access
- [ ] `sshpass` installed on dev machine
- [ ] Build tools on Pi: `sudo apt-get install build-essential`
- [ ] Hostname set to `raspi.local` or configure IP

**Deployment:**
- [ ] Run File 3 on ai.local
- [ ] Auto-deploy triggered by sshpass commands in File 3
- [ ] Auto-compile and test on Pi (via SSH bash script)

---

## Performance Notes

### Training (ai.local, RTX 3070):
- **File 2/3:** 1000 epochs ≈ 2–5 seconds (depending on data size)

### Inference (Raspberry Pi Zero 2W):
- **File 2 (ONNX):** ~0.1–1 ms per prediction (includes ONNX Runtime overhead)
- **File 3 (C/C++):** ~1–2 µs per prediction (no overhead, pure computation)
- **Speedup:** C/C++ is ~100× faster than ONNX for tiny models

### Memory (Pi Zero 2W, 512 MB RAM):
- **File 2:** ~150 MB after loading ONNX Runtime + model + data
- **File 3:** ~5 MB (just the compiled executable)

---

## Troubleshooting

### sshpass issues:
```bash
# Install sshpass
sudo apt-get install sshpass

# Test SSH access first
ssh pi@raspi.local "echo OK"
```

### ONNX Runtime installation slow on Pi:
- Pre-download ARM wheels from GitHub
- Use `pip install --no-build-isolation` to skip compilation

### C compilation fails on Pi:
- Verify build tools: `gcc --version`, `g++ --version`
- Install if missing: `sudo apt-get install build-essential`

### Binary mismatch (PyTorch vs ONNX/C):
- Ensure `model.eval()` is set before export
- Verify normalization stats are identical in all paths
- Check that input order matches (x1, x2) everywhere

---

## Next Steps

**To extend this pipeline:**

1. **Different model sizes:** Edit the layer dimensions (e.g., 2→64→64→1 for larger capacity)
2. **Real data:** Replace `simple_nonlinear.csv` with actual sensor or measurement data
3. **Multi-output:** Change final layer to `nn.Linear(32, N)` for N outputs
4. **Time-series:** Use history feedback loop (already demo'd in Files 2 & 3)
5. **Quantization:** Convert weights to int8 (reduces binary size, slightly slower)
6. **Batch prediction:** Process multiple inputs in parallel in C code

---

## References

- PyTorch docs: https://pytorch.org/docs
- ONNX Runtime: https://onnxruntime.ai
- ARM toolchain for MCU: https://developer.arm.com/Tools-and-Software/open-source-software/developer-tools/gnu-toolchain
- Raspberry Pi docs: https://www.raspberrypi.com/documentation
