import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from pathlib import Path


# Quick Check: Is PyTorch using VRAM?
if torch.cuda.is_available():
    print("✅ GPU Available — VRAM will be used!")
else:
    print("❌ CPU Only — No VRAM usage")








# ======Create sample CSV if it doesn't exist===
csv_path = 'lut_calibration.csv'
import pandas as pd
df = pd.DataFrame({
    'x1'      : [0, 1, 2, 3, 4,5,6,7,8,9] ,
    'x2'      : [0, 1, 2, 3, 4,5,6,7,8,9] ,
})
df['output'] = 2*df['x1'] + 3*df['x2'] 
df.to_csv(csv_path, index=False)
print(f"Created sample CSV with {len(df)} rows")
#==================================================



#=============================================================
csv_path = 'lut_calibration.csv'
print(f"Loading LUT from CSV...{csv_path}")

# Load
df = pd.read_csv(csv_path)
x_train = torch.tensor(df[['x1', 'x2']].values, dtype=torch.float32)
y_train = torch.tensor(df[['output']].values, dtype=torch.float32)
print(f"Loaded LUT: inputs  pts x dimentions {x_train.shape}")
print(f"Loaded LUT: outputs pts x dimentions {y_train.shape}")
# === old =====
# FIX: Add outer brackets
#x_ = torch.tensor([ [1,1,1], [2,2,2], [3,3,3], [4,4,4] , [4,4,7] ], dtype=torch.float32)
#y_train = torch.tensor([ [3],     [5],     [7],     [8] ,     [8] ], dtype=torch.float32)  # ← Fixed
#=============================================================







print(f"Creating a model...")
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer = nn.Linear(2, 1)
    def forward(self, x):
        return self.layer(x)




model = SimpleModel()
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)





#=================================================
print(f"Training a model...")    
num_epochs = 3000
for epoch in range(num_epochs):
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch % 30 == 0):
        print(f"epoch {epoch} loss {loss:.9f} ")
        
print(f"Done Training...") 


# ===== SAVE TRAINED MODEL =====
print("Saving trained model...")
torch.save(model.state_dict(), 'py_torch_model.pth')
#===================================================




# =====  LOAD TRAINED MODEL =====
print("Loading trained model...")
model_loaded = SimpleModel()
model_loaded.load_state_dict(torch.load('py_torch_model.pth'))
model_loaded.eval()

# =====  TEST LOADED TRAINED MODEL =====
test_loops = 10
for test_loop in range(test_loops):
    with torch.no_grad():
        test_input = torch.tensor([[3, test_loop]], dtype=torch.float32)
        prediction = model_loaded(test_input)
        print(f"Testing loaded model: {prediction.detach().numpy().round(1)}")