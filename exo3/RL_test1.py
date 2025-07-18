#!/usr/bin/env python3
"""
RL_test1.py  · Deep-Learning Final Exam – Exercise 3 · Task 1
Loads the model « report3.pth », reads RL_test1.csv, predicts the
optimal action at each step and writes RL_result1.csv preserving all
original columns + Pbess, SOC, Pgrid.
"""

import math, pandas as pd, torch, torch.nn as nn, numpy as np
from packaging import version

# ───── Device
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[test] device → {DEVICE.upper()}")

# ───── BESS constants
E_BESS, P_UNIT = 10.0, 5.0
SOC_MIN, SOC_MAX = .2, .9
INIT_SOC = 0.5

# ───── NoisyLinear 
class NoisyLinear(nn.Module):
    def __init__(self,i,o): super().__init__(); self.W=nn.Linear(i,o)
    def forward(self,x): return self.W(x)          # noise = 0 in eval

# ───── Q-network 
class QNet(nn.Module):
    def __init__(self,in_dim=4,n_act=3):
        super().__init__()
        self.f=nn.Sequential(nn.Linear(in_dim,128), nn.SiLU(),
                             nn.Linear(128,128),    nn.SiLU())
        self.V=nn.Linear(128,1); self.A=nn.Linear(128,n_act)
    def forward(self,x):
        x=self.f(x); v=self.V(x); a=self.A(x)
        return v + a - a.mean(1,keepdim=True)

def act(net,state):
    with torch.no_grad():
        st=torch.tensor(state,dtype=torch.float32,device=DEVICE).unsqueeze(0)
        return int(net(st).argmax(1).item())

# ───── Model loading
net=QNet().to(DEVICE)
net.load_state_dict(torch.load("report3.pth",map_location=DEVICE))
net.eval()

# ───── Reading test data
df=pd.read_csv("RL_test1.csv")
if df.shape[1] < 3:
    raise ValueError("Test file must contain at least 3 columns (time, price, load).")

price=df.iloc[:,1].to_numpy(dtype=np.float32)
load =df.iloc[:,2].to_numpy(dtype=np.float32)
T=len(df)

# ───── Prediction loop
soc=INIT_SOC
soc_hist, pbess_hist, pgrid_hist = [], [], []
for t in range(T):
    state=np.array([ t/T, soc, load[t]/10.0, price[t]/100.0 ],dtype=np.float32)
    a_idx=act(net,state)
    pbess=P_UNIT*[-1,0,1][a_idx]               # kW
    soc += 0.25*pbess/E_BESS                   # update
    soc = max(SOC_MIN, min(SOC_MAX, soc))      # bound
    pgrid = load[t] + pbess                    # kW
    pbess_hist.append(pbess); soc_hist.append(soc); pgrid_hist.append(pgrid)

# ───── Writing result
df["Pbess"]=pbess_hist
df["SOC"]  =soc_hist
df["Pgrid"]=pgrid_hist
df.to_csv("RL_result1.csv",index=False)
print("[test] RL_result1.csv written ✓")
