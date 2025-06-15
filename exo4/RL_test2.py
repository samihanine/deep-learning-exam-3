#!/usr/bin/env python3
"""
RL_test2.py  · Deep-Learning Final Exam – Exercise 3 · Task 2
Charge report4.pth, lit RL_test2.csv, prédit Pbess continu,
met à jour SOC, calcule Pgrid et écrit RL_result2.csv.
"""

import math, pandas as pd, torch, torch.nn as nn, numpy as np
DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps"  if torch.backends.mps.is_available()
          else "cpu")
print(f"[test] device → {DEVICE.upper()}")

# ─────────── Constantes BESS
E_BESS, P_UNIT = 10.0, 5.0
SOC_MIN, SOC_MAX = .2, .9
INIT_SOC = 0.5

# ─────────── Réseau Acteur (même archi que train) ─────
class Actor(nn.Module):
    def __init__(self, in_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,128), nn.SiLU(),
            nn.Linear(128,128),    nn.SiLU(),
            nn.Linear(128,1),      nn.Tanh() )
    def forward(self,x): return self.net(x)

actor = Actor().to(DEVICE)
actor.load_state_dict(torch.load("report4.pth", map_location=DEVICE))
actor.eval()

# ─────────── Lecture données test ─────────────────────
df = pd.read_csv("RL_test2.csv")
if df.shape[1] < 3:
    raise ValueError("Le fichier test doit contenir ≥3 colonnes (time, price, load).")
price = df.iloc[:,1].to_numpy(dtype=np.float32)
load  = df.iloc[:,2].to_numpy(dtype=np.float32)
T = len(df)

# ─────────── Boucle de prédiction — actions continues ─
soc = INIT_SOC
Pbess_hist, soc_hist, pgrid_hist = [], [], []
for t in range(T):
    state = np.array([ t/T, soc, load[t]/10.0, price[t]/100.0 ], dtype=np.float32)
    with torch.no_grad():
        a = actor(torch.tensor(state,device=DEVICE).unsqueeze(0)).cpu().numpy()[0,0]
    Pbess = float(np.clip(a, -1, 1) * P_UNIT)         # kW
    soc  += 0.25*Pbess/E_BESS
    soc   = max(SOC_MIN, min(SOC_MAX, soc))
    pgrid = load[t] + Pbess
    Pbess_hist.append(Pbess); soc_hist.append(soc); pgrid_hist.append(pgrid)

# ─────────── Écriture résultat ────────────────────────
df["Pbess"] = Pbess_hist
df["SOC"]   = soc_hist
df["Pgrid"] = pgrid_hist
df.to_csv("RL_result2.csv", index=False)
print("[test] RL_result2.csv written ✓")
