import torch
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt

try:
    if torch.backend.mps.is_built() and torch.backends.mps.is_available():
        device = torch.device('mps')    
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(torch.cuda.is_available())

print('MPS?', torch.backends.mps.is_built())
print('MPS available ?', torch.backends.mps.is_available())

class SimpleODEFunc(torch.nn.Module):
    def forward(self, t, y):
        return -y
# Initial condition
y0 = torch.tensor([1.0], device=device)
t = torch.linspace(0,5,steps = 50)

# Instantiate ODE function
odefunc = SimpleODEFunc().to(device)

# Solve the ODE
y_res =  odeint(odefunc, y0, t)
print(y_res)
t_arr = t.cpu()
y_arr = y_res.cpu()

plt.plot(t_arr, y_arr,)
plt.show()
