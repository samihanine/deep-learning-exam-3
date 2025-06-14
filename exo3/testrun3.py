#!/usr/bin/env python3
"""
Deep-Learning Final Exam – Exercise 3 · Task 1  (DQN, discrete actions)
Author : <votre nom>
Date   : 2025-06-14

Structure attendue du dossier :
.
├── RL_test1.csv        (données de test fournies)
└── testrun3.py         (ce fichier)

À l’exécution :
  • Entraîne (avec données auto-générées) un DQN discret (actions –1, 0, +1)
  • Sauvegarde les meilleurs poids        → report3.pth
  • Produit la séquence d’actions test    → RL_result1.csv
  • Trace la courbe récompense/perte      → rl_curve.png / rl_curve.pdf
"""

# ╭──────────────────────────── Imports ─────────────────────────────╮
import os, math, random, time, datetime, platform, psutil, gc, csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from packaging import version
from tabulate import tabulate
from tqdm.auto import tqdm

import torch
import torch.nn as nn
# ╰──────────────────────────────────────────────────────────────────╯

# ╭──────────────────── Device & mixed-precision ────────────────────╮
if torch.cuda.is_available():
    DEVICE, AMP_DEVICE = "cuda", "cuda"
elif torch.backends.mps.is_available():
    DEVICE = AMP_DEVICE = "mps"
else:
    DEVICE = AMP_DEVICE = "cpu"
print(f"Using device ➜ {DEVICE.upper()}")

MIXED_PREC = AMP_DEVICE in {"cuda", "mps"}
USE_SCALER  = AMP_DEVICE == "cuda"
USE_COMPILE = AMP_DEVICE == "cuda"

if USE_SCALER:
    from torch.cuda.amp import GradScaler
from torch import autocast
# ╰──────────────────────────────────────────────────────────────────╯

# ╭───────────────────────── Hyper-paramètres ───────────────────────╮
GAMMA          = 0.99
LR             = 3e-4
BATCH          = 256
EPOCHS         = 4000            # nombre maximal d’updates
TARGET_SYNC    = 200             # sync réseau cible
BUFFER_CAP     = 50_000
EPS_START      = 1.0
EPS_END        = 0.05
EPS_DECAY      = 2_000           # steps linéaires
EARLY_STOP     = 400             # patience sur récompense moy.
WD             = 0
SEED           = 42
NUM_WORKERS    = 0
# ╰──────────────────────────────────────────────────────────────────╯

# ╭──────────────────────── Utils : seed & précise ──────────────────╮
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)
set_seed()

if version.parse(torch.__version__) >= version.parse("2.0"):
    torch.set_float32_matmul_precision("high")
# ╰──────────────────────────────────────────────────────────────────╯

# ╭────────────────────── Environnement BESS ────────────────────────╮
class BESSEnv:
    """Environnement simulant la batterie + réseau pour une journée (96 pas)."""
    def __init__(self, load_profile, price_profile):
        self.load = load_profile.astype(np.float32)
        self.price = price_profile.astype(np.float32)
        self.T = len(self.load)                 # 96 pas
        self.ptr = 0
        self.E_bess = 10.0                     # kWh
        self.P_unit = 5.0                      # kW par incrément d’action
        self.soc_min, self.soc_max = 0.20, 0.90
        self.reset()

    # état : [t_norm, SOC, P_load, price]
    def reset(self):
        self.ptr = 0
        self.soc = 0.5                         # SOC initial 50 %
        return self._get_state()

    def _get_state(self):
        return np.array([
            self.ptr / self.T,
            self.soc,
            self.load[self.ptr] / 10.0,        # normalisation grossière
            self.price[self.ptr] / 100.0
        ], dtype=np.float32)

    def step(self, action_idx):
        """
        action_idx ∈ {0,1,2} →  -1,0,+1  =>
        P_BESS' = 5 kW × {-1,0,1}
        """
        a_val = [-1, 0, 1][action_idx]
        P_bess = a_val * self.P_unit           # kW
        # Mise à jour SOC
        self.soc += (0.25 * P_bess) / self.E_bess
        # Contraintes : pénalité si hors bornes
        penalty = 0.0
        if self.soc < self.soc_min:
            penalty -= 10.0 * (self.soc_min - self.soc)
            self.soc = self.soc_min
        elif self.soc > self.soc_max:
            penalty -= 10.0 * (self.soc - self.soc_max)
            self.soc = self.soc_max

        # P_grid' = P_load' + P_BESS'
        P_grid = self.load[self.ptr] + P_bess
        cost = P_grid * self.price[self.ptr] * 0.25          # € (Ts = 15 min)
        reward = -cost + penalty                             # on maximise
        done = (self.ptr == self.T - 1)
        
        # Incrémenter le pointeur après avoir calculé l'état actuel
        self.ptr += 1
        
        # Si l'épisode est terminé, retourner l'état actuel sans appeler _get_state()
        if done:
            # Retourner un état factice ou le dernier état valide
            next_state = np.array([
                1.0,  # t_norm à 1.0 car on est à la fin
                self.soc,
                0.0,  # pas de charge future
                0.0   # pas de prix futur
            ], dtype=np.float32)
        else:
            next_state = self._get_state()
            
        return next_state, reward, done, {}

# util – génère des profils charge/prix stochastiques pour l’agent
def generate_random_day():
    t = np.arange(96)
    load = 3 + 2 * np.sin(2*math.pi*(t-20)/96) + np.random.normal(0,0.5,96)
    price = 60 + 20 * np.sin(2*math.pi*(t+10)/96) + np.random.normal(0,3,96)
    return load.clip(0, None), price.clip(0, None)
# ╰──────────────────────────────────────────────────────────────────╯

# ╭────────────────────────── Replay buffer ──────────────────────────╮
class ReplayBuffer:
    def __init__(self, cap):
        self.cap = cap
        self.mem = []
        self.pos = 0
    def push(self, s, a, r, s2, d):
        if len(self.mem) < self.cap:
            self.mem.append(None)
        self.mem[self.pos] = (s, a, r, s2, d)
        self.pos = (self.pos + 1) % self.cap
    def sample(self, batch):
        idx = random.sample(range(len(self.mem)), batch)
        s,a,r,s2,d = zip(*(self.mem[i] for i in idx))
        return (np.stack(s), np.array(a), np.array(r, dtype=np.float32),
                np.stack(s2), np.array(d, dtype=np.float32))
    def __len__(self): return len(self.mem)
# ╰──────────────────────────────────────────────────────────────────╯

# ╭──────────────────────── Réseau Q-valeurs ─────────────────────────╮
class QNet(nn.Module):
    def __init__(self, in_dim=4, n_act=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.SiLU(inplace=True),
            nn.Linear(128, 128), nn.SiLU(inplace=True),
            nn.Linear(128, n_act)
        )
    def forward(self, x): return self.net(x)

# ╰──────────────────────────────────────────────────────────────────╯

def select_action(qnet, state, step):
    eps = max(EPS_END, EPS_START - step / EPS_DECAY)
    if random.random() < eps:
        return random.randrange(3), eps
    with torch.no_grad():
        q = qnet(torch.from_numpy(state[None]).to(DEVICE))
        return int(q.argmax(1).item()), eps

# ╭────────────────────────── Entraînement ───────────────────────────╮
def train():
    qnet = QNet().to(DEVICE)
    tgt  = QNet().to(DEVICE)
    tgt.load_state_dict(qnet.state_dict())
    if USE_COMPILE:
        qnet = torch.compile(qnet)

    opt = torch.optim.AdamW(qnet.parameters(), lr=LR, weight_decay=WD)
    buf = ReplayBuffer(BUFFER_CAP)
    scaler = GradScaler() if USE_SCALER else None

    rewards_window, loss_window = [], []
    best_mean, best_ep = -1e9, 0
    global_step = 0

    print("\n────────────── Training ──────────────")
    for epoch in range(1, EPOCHS + 1):
        # génère un jour aléatoire
        load, price = generate_random_day()
        env = BESSEnv(load, price)
        state = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action, eps = select_action(qnet, state, global_step)
            nstate, reward, done, _ = env.step(action)
            buf.push(state, action, reward, nstate, done)
            state = nstate
            ep_reward += reward
            global_step += 1

            if len(buf) < BATCH: continue
            s,a,r,s2,d = buf.sample(BATCH)
            s  = torch.from_numpy(s).to(DEVICE)
            a  = torch.from_numpy(a).long().to(DEVICE)
            r  = torch.from_numpy(r).to(DEVICE)
            s2 = torch.from_numpy(s2).to(DEVICE)
            d  = torch.from_numpy(d).to(DEVICE)

            with autocast(device_type=AMP_DEVICE, enabled=MIXED_PREC):
                q_sa = qnet(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_ns = tgt(s2).max(1)[0]
                    target = r + GAMMA * q_ns * (1 - d)
                loss = nn.functional.mse_loss(q_sa, target)

            opt.zero_grad(set_to_none=True)
            if USE_SCALER:
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()

            if global_step % TARGET_SYNC == 0:
                tgt.load_state_dict(qnet.state_dict())

            loss_window.append(loss.item())

        rewards_window.append(ep_reward)

        # stats toutes 100 épisodes
        if epoch % 100 == 0:
            mean_r = np.mean(rewards_window[-100:])
            mean_l = np.mean(loss_window[-100:]) if loss_window else math.nan
            table = [
                ["episode", epoch],
                ["mean reward (100)", f"{mean_r:.2f}"],
                ["mean loss (100)",   f"{mean_l:.4f}"],
                ["epsilon",           f"{eps:.3f}"],
                ["steps",             global_step],
                ["RAM used (GB)",     f"{psutil.Process().memory_info().rss/1e9: .2f}"],
            ]
            print("\n"+tabulate(table, headers=["metric","value"], tablefmt="github"))
            if mean_r > best_mean + 1e-3:
                best_mean, best_ep = mean_r, epoch
                torch.save(qnet.state_dict(), "report3.pth")
            if epoch - best_ep >= EARLY_STOP:   # early-stop
                print(f"\nEarly-stopping at episode {epoch} (best={best_ep})")
                break

    # courbe recompense
    plt.figure(figsize=(6,4))
    plt.plot(rewards_window)
    plt.xlabel("episode"); plt.ylabel("reward")
    plt.tight_layout(); plt.savefig("rl_curve.png", dpi=150)
    with PdfPages("rl_curve.pdf") as pdf:
        pdf.savefig(); plt.close()
        fig = plt.figure(figsize=(6,4)); plt.axis("off")
        txt = (f"Mean best reward : {best_mean:.2f} @ ep {best_ep}\n"
               f"Device : {DEVICE} | γ = {GAMMA} | LR = {LR}\n"
               f"EPOCHS ={EPOCHS}  BUFFER={BUFFER_CAP}  BATCH={BATCH}")
        plt.text(0.02,0.5,txt,va="center",ha="left",family="monospace",fontsize=9)
        pdf.savefig(); plt.close()
    print("\nTraining finished.  Best mean reward:", best_mean)

# ╰──────────────────────────────────────────────────────────────────╯

# ╭─────────────────── Génère RL_result1.csv ─────────────────────────╮
def inference():
    load_price = pd.read_csv("RL_test1.csv").values
    if load_price.shape[1] < 3:
        raise ValueError("RL_test1.csv doit contenir au moins 3 colonnes : time_step, price, load")
    # Colonnes : 0=time_step, 1=price, 2=load
    price, load = load_price[:,1], load_price[:,2]
    env = BESSEnv(load, price)

    qnet = QNet().to(DEVICE)
    qnet.load_state_dict(torch.load("report3.pth", map_location=DEVICE))
    qnet.eval()

    actions = []
    state = env.reset()
    done = False
    while not done:
        with torch.no_grad():
            q = qnet(torch.from_numpy(state[None]).to(DEVICE))
            act = int(q.argmax(1).item())
        actions.append([-1,0,1][act])
        state, _, done, _ = env.step(act)

    pd.DataFrame(actions, columns=["a_t"]).to_csv("RL_result1.csv", index=False)
    print("RL_result1.csv written.")

# ╰──────────────────────────────────────────────────────────────────╯

if __name__ == "__main__":
    tic = time.time()
    train()
    inference()
    print(f"Total runtime : {time.time() - tic:.1f}s")
