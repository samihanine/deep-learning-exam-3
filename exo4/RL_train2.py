# ─────────── Common imports & utilities ───────────
import math, random, time, numpy as np, pandas as pd, psutil, collections
import torch, torch.nn as nn, torch.nn.functional as F
from packaging import version
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps"  if torch.backends.mps.is_available()
          else "cpu")
print(f"[train] device → {DEVICE.upper()}")

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if DEVICE == "cuda": torch.cuda.manual_seed_all(SEED)
if version.parse(torch.__version__) >= version.parse("2.0"):
    torch.set_float32_matmul_precision("high")

# ─────────── Identical BESS environment ────────────
class BESSEnv:
    E_BESS, P_UNIT = 10.0, 5.0
    SOC_MIN, SOC_MAX = .2, .9
    def __init__(self, load, price):
        self.load, self.price = load.astype(np.float32), price.astype(np.float32)
        self.T = len(load); self.reset()
    def reset(self):
        self.ptr, self.soc = 0, .5
        return self._state()
    def _state(self):
        return np.array([ self.ptr/self.T, self.soc,
                          self.load[self.ptr]/10.0, self.price[self.ptr]/100.0 ],
                        dtype=np.float32)
    def step(self, a_cont):
        a_cont = float(np.clip(a_cont, -1., 1.))
        Pbess = a_cont*self.P_UNIT
        self.soc += 0.25*Pbess/self.E_BESS
        penalty = 0.
        if self.soc < self.SOC_MIN:
            penalty -= 10*(self.SOC_MIN-self.soc); self.soc=self.SOC_MIN
        elif self.soc > self.SOC_MAX:
            penalty -= 10*(self.soc-self.SOC_MAX); self.soc=self.SOC_MAX
        Pgrid = self.load[self.ptr] + Pbess
        reward = -(Pgrid*self.price[self.ptr]*0.25) + penalty
        done = self.ptr == self.T-1
        self.ptr += 1
        nxt = np.array([1.0, self.soc, 0., 0.], np.float32) if done else self._state()
        return nxt, reward, done, {}
def random_day():
    t=np.arange(96)
    load=3+2*np.sin(2*np.pi*(t-20)/96)+np.random.normal(0,.5,96)
    price=60+20*np.sin(2*np.pi*(t+10)/96)+np.random.normal(0,3,96)
    return load.clip(0).astype(np.float32), price.clip(0).astype(np.float32)

# ─────────── Replay buffer (optional priority) ─────
class Replay:
    def __init__(self, cap): self.cap=cap; self.buf=collections.deque(maxlen=cap)
    def __len__(self): return len(self.buf)
    def push(self,*tr): self.buf.append(tr)
    def sample(self,b):
        batch=random.sample(self.buf,b)
        s,a,r,s2,d=map(np.stack,zip(*batch))
        return ( torch.tensor(s ,dtype=torch.float32,device=DEVICE),
                 torch.tensor(a ,dtype=torch.float32,device=DEVICE),
                 torch.tensor(r ,dtype=torch.float32,device=DEVICE),
                 torch.tensor(s2,dtype=torch.float32,device=DEVICE),
                 torch.tensor(d ,dtype=torch.float32,device=DEVICE) )

# ─────────── Actor & critic networks ──────────────────
class Actor(nn.Module):
    def __init__(self, in_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,128), nn.SiLU(),
            nn.Linear(128,128),    nn.SiLU(),
            nn.Linear(128,1),      nn.Tanh() )   # output ∈ [-1,1]
    def forward(self,x): return self.net(x)

class Critic(nn.Module):                 # Q(s,a)
    def __init__(self, in_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim+1,128), nn.SiLU(),
            nn.Linear(128,128),      nn.SiLU(),
            nn.Linear(128,1) )
    def forward(self,s,a):
        return self.net(torch.cat([s,a],1)).squeeze(1)

# ─────────── TD3 hyperparameters & training ─────
GAMMA, TAU          = 0.995, 0.005
LR_ACT, LR_CRIT     = 5e-5, 3e-4
POLICY_FREQ, NOISE  = 2, 0.2          # delay + target noise σ
EPISODES, STEPS_EP  = 6_000, 96
BATCH, BUFFER       = 512, 200_000
START_STEPS         = 5_000
TARGET_NOISE_CLIP   = 0.5
EARLY_STOP, BEST_EP = 800, 0
best_meanR          = -1e9
R_hist              = []

# ─────────── Network & optimizer instantiation ───
actor  = Actor().to(DEVICE)
actor_t= Actor().to(DEVICE); actor_t.load_state_dict(actor.state_dict())
crit1  = Critic().to(DEVICE)
crit2  = Critic().to(DEVICE)
crit1_t= Critic().to(DEVICE); crit1_t.load_state_dict(crit1.state_dict())
crit2_t= Critic().to(DEVICE); crit2_t.load_state_dict(crit2.state_dict())
optA   = torch.optim.AdamW(actor.parameters(),  lr=LR_ACT  , betas=(0.9,0.99))
optC1  = torch.optim.AdamW(crit1.parameters(),  lr=LR_CRIT , betas=(0.9,0.99))
optC2  = torch.optim.AdamW(crit2.parameters(),  lr=LR_CRIT , betas=(0.9,0.99))

buf = Replay(BUFFER)
total_steps = 0

print("\n[train] start TD3")
for ep in range(1, EPISODES+1):
    load,price = random_day(); env=BESSEnv(load,price)
    s = env.reset(); ep_R = 0
    for t in range(STEPS_EP):
        total_steps += 1
        # Exploration / exploitation
        if total_steps < START_STEPS:
            a = np.random.uniform(-1,1,size=1).astype(np.float32)
        else:
            with torch.no_grad():
                a = actor(torch.tensor(s,device=DEVICE).unsqueeze(0)).cpu().numpy()[0]
                a += np.random.normal(0, NOISE, size=a.shape)    # bruit exploratoire
                a = np.clip(a, -1, 1)
        s2, r, done, _ = env.step(a)
        buf.push(s, a, r, s2, float(done))
        s = s2; ep_R += r
        # Apprentissage après START_STEPS
        if len(buf) >= BATCH:
            S,A,R,S2,D = buf.sample(BATCH)
            with torch.no_grad():
                # cible action (plus bruit)
                At = actor_t(S2)
                noise = (torch.randn_like(At)*NOISE).clamp(-TARGET_NOISE_CLIP, TARGET_NOISE_CLIP)
                At = (At + noise).clamp(-1,1)
                Qt1 = crit1_t(S2, At)
                Qt2 = crit2_t(S2, At)
                Qt  = torch.min(Qt1, Qt2)
                y   = R + GAMMA*(1-D)*Qt
            # Critics
            for optC,crit in [(optC1,crit1),(optC2,crit2)]:
                optC.zero_grad()
                lossC = F.mse_loss(crit(S,A), y)
                lossC.backward(); optC.step()
            # Actor
            if total_steps % POLICY_FREQ == 0:
                optA.zero_grad()
                lossA = -crit1(S, actor(S)).mean()
                lossA.backward(); optA.step()
                # Soft-update targets
                with torch.no_grad():
                    for θ_t,θ in zip(actor_t.parameters(), actor.parameters()):
                        θ_t.data = θ_t.data*(1-TAU) + θ.data*TAU
                    for net_t,net in [(crit1_t,crit1),(crit2_t,crit2)]:
                        for θ_t,θ in zip(net_t.parameters(), net.parameters()):
                            θ_t.data = θ_t.data*(1-TAU) + θ.data*TAU
        if done: break
    R_hist.append(ep_R)
    if ep % 100 == 0:
        m = np.mean(R_hist[-100:]); print(f"[ep {ep}] mean_R={m:7.1f}")
        if m > best_meanR:
            best_meanR, BEST_EP = m, ep
            torch.save(actor.state_dict(), "report4.pth")
        if ep - BEST_EP >= EARLY_STOP: break

# Learning curve
plt.figure(); plt.plot(R_hist); plt.xlabel("episode"); plt.ylabel("reward"); plt.tight_layout()
plt.savefig("rl_curve2.png",dpi=150)
with PdfPages("rl_curve2.pdf") as pdf: pdf.savefig()
print(f"[train] finished – best mean_R {best_meanR:.1f} @ ep {BEST_EP}")
