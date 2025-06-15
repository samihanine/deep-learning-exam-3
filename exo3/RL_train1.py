# ───────────────────── Imports & utils ─────────────────────
import math, random, time, psutil, numpy as np, pandas as pd
import torch, torch.nn as nn
from packaging import version
from tabulate import tabulate
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ───────────── Device & AMP ──────────
if torch.cuda.is_available():
    DEVICE, AMP = "cuda", "cuda"
elif torch.backends.mps.is_available():
    DEVICE = AMP = "mps"
else:
    DEVICE = AMP = "cpu"
print(f"[train] device → {DEVICE.upper()}")

MIXED = AMP in {"cuda", "mps"}
USE_SCALER = AMP == "cuda"
if USE_SCALER:
    from torch.cuda.amp import GradScaler
from torch import autocast

# ───────────────────── Hyperparameters ─────────────────────
GAMMA, N_STEP, LR = 0.995, 3, 1e-4
BATCH, EPOCHS = 512, 6_000
TARGET_SYNC, BUFFER = 400, 100_000
ALPHA, BETA0, BETA_INC = 0.6, 0.4, 1e-4
NOISY_SIGMA, EARLY_STOP = 0.5, 600
SEED = 42

def seed_everything(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if DEVICE == "cuda": torch.cuda.manual_seed_all(s)
seed_everything()
if version.parse(torch.__version__) >= version.parse("2.0"):
    torch.set_float32_matmul_precision("high")

# ───────────────────── BESS environment ─────────────────
class BESSEnv:
    E_BESS, P_UNIT = 10.0, 5.0
    SOC_MIN, SOC_MAX = .2, .9
    def __init__(self, load, price):
        self.load, self.price = load.astype(np.float32), price.astype(np.float32)
        self.T = len(self.load); self.reset()
    def reset(self):
        self.ptr, self.soc = 0, .5
        return self._state()
    def _state(self):
        return np.array([ self.ptr/self.T, self.soc,
                          self.load[self.ptr]/10.0, self.price[self.ptr]/100.0 ],
                        dtype=np.float32)
    def step(self, act_idx):
        a_val = [-1,0,1][act_idx]
        P_bess = a_val*self.P_UNIT
        self.soc += 0.25*P_bess/self.E_BESS
        penalty = 0.0
        if self.soc < self.SOC_MIN:
            penalty -= 10*(self.SOC_MIN-self.soc); self.soc=self.SOC_MIN
        elif self.soc > self.SOC_MAX:
            penalty -= 10*(self.soc-self.SOC_MAX); self.soc=self.SOC_MAX
        P_grid = self.load[self.ptr] + P_bess
        reward = -(P_grid*self.price[self.ptr]*0.25) + penalty
        done = self.ptr == self.T-1
        self.ptr += 1
        nxt = np.array([1.0,self.soc,0.0,0.0],np.float32) if done else self._state()
        return nxt, reward, done, {}

def random_day():
    t=np.arange(96)
    load=3+2*np.sin(2*np.pi*(t-20)/96)+np.random.normal(0,.5,96)
    price=60+20*np.sin(2*np.pi*(t+10)/96)+np.random.normal(0,3,96)
    return load.clip(0).astype(np.float32), price.clip(0).astype(np.float32)

# ─────────────────── Prioritized replay buffer ──────────────────────
class PriorReplay:
    def __init__(self, cap, alpha): self.cap=cap; self.a=alpha
    def __len__(self): return getattr(self,"n",0)
    def push(self, tr, td=1.0):
        if not hasattr(self,"m"):
            self.m, self.p, self.pos, self.n = [None]*self.cap, np.zeros(self.cap),0,0
        self.m[self.pos], self.p[self.pos] = tr, (abs(td)+1e-6)**self.a
        self.pos = (self.pos+1)%self.cap; self.n=min(self.n+1,self.cap)
    def sample(self,b,beta):
        P=self.p[:self.n]; P/=P.sum()
        idx=np.random.choice(self.n,b,p=P)
        w=(self.n*P[idx])**(-beta); w/=w.max()
        s,a,r,s2,d=zip(*[self.m[i] for i in idx])
        return (np.stack(s).astype(np.float32), np.array(a),
                np.array(r,np.float32), np.stack(s2).astype(np.float32),
                np.array(d,np.float32),
                torch.tensor(w,dtype=torch.float32,device=DEVICE), idx)
    def update(self,idx,td): self.p[idx]=(abs(td)+1e-6)**self.a

# ───────────────────────── Noisy layer ──────────────────────────────
class NoisyLinear(nn.Module):
    def __init__(self,i,o,sigma=NOISY_SIGMA):
        super().__init__()
        self.mu_W=nn.Parameter(torch.empty(o,i)); self.mu_b=nn.Parameter(torch.empty(o))
        self.sigma_W=nn.Parameter(torch.empty(o,i)); self.sigma_b=nn.Parameter(torch.empty(o))
        self.sigma_init=sigma; self.reset_parameters()
    def reset_parameters(self):
        bound=1/math.sqrt(self.mu_W.size(1))
        self.mu_W.data.uniform_(-bound,bound); self.mu_b.data.uniform_(-bound,bound)
        self.sigma_W.data.fill_(self.sigma_init/math.sqrt(self.mu_W.size(1)))
        self.sigma_b.data.fill_(self.sigma_init/math.sqrt(self.mu_W.size(1)))
    def _noise(self, n): v=torch.randn(n,device=self.mu_W.device); return v.sign()*v.abs().sqrt()
    def forward(self,x):
        if self.training:
            eps_i, eps_o = self._noise(x.size(-1)), self._noise(self.mu_W.size(0))
            W=self.mu_W+self.sigma_W*torch.ger(eps_o,eps_i)
            b=self.mu_b+self.sigma_b*eps_o
        else:
            W,b=self.mu_W,self.mu_b
        return torch.nn.functional.linear(x,W,b)

# ───────────────────────── Q-network ────────────────────────────────
class QNet(nn.Module):
    def __init__(self,in_dim=4, n_act=3):
        super().__init__()
        self.f = nn.Sequential(NoisyLinear(in_dim,128), nn.SiLU(),
                               NoisyLinear(128,128), nn.SiLU())
        self.V = NoisyLinear(128,1); self.A = NoisyLinear(128,n_act)
    def forward(self,x):
        x=self.f(x); v=self.V(x); a=self.A(x)
        return v + a - a.mean(1,keepdim=True)

def act(net,state):
    with torch.no_grad():
        st=torch.from_numpy(state[None]).to(DEVICE,dtype=torch.float32)
        return int(net(st).argmax(1).item())

# ─────────────────────────── Training loop ──────────────────────────
def train():
    online,target=QNet().to(DEVICE),QNet().to(DEVICE)
    target.load_state_dict(online.state_dict())
    if DEVICE=="cuda": online=torch.compile(online)
    opt=torch.optim.AdamW(online.parameters(),lr=LR)
    buf=PriorReplay(BUFFER,ALPHA)
    scaler=GradScaler() if USE_SCALER else None
    beta=BETA0; R_hist=[]; best_R,best_ep=-1e9,0
    step_count = 0

    def push_nstep(queue_r,queue_sa,r_new,s_next,done):
        queue_r.append(r_new)
        if len(queue_r)>=N_STEP:
            R=sum(queue_r[i]*(GAMMA**i) for i in range(N_STEP))
            s0,a0=queue_sa.pop(0); queue_r.pop(0)
            buf.push((s0,a0,R,s_next,done))

    print("\n[train] start")
    for ep in range(1,EPOCHS+1):
        load,price=random_day(); env=BESSEnv(load,price)
        queue_sa,queue_r=[],[]; s=env.reset(); ep_R=0; done=False
        while not done:
            a=act(online,s); s2,r,done,_=env.step(a)
            queue_sa.append((s,a)); push_nstep(queue_r,queue_sa,r,s2,done)
            s=s2; ep_R+=r
            if len(buf)<BATCH: continue
            beta=min(1.,beta+BETA_INC)
            sB,aB,rB,s2B,dB,w,idx=buf.sample(BATCH,beta)
            sB=torch.from_numpy(sB).to(DEVICE); aB=torch.from_numpy(aB).long().to(DEVICE)
            rB=torch.from_numpy(rB).to(DEVICE); s2B=torch.from_numpy(s2B).to(DEVICE)
            dB=torch.from_numpy(dB).to(DEVICE); w=w
            with autocast(device_type=AMP,enabled=MIXED):
                q_sa=online(sB).gather(1,aB.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    a2=online(s2B).argmax(1)
                    q_ns=target(s2B).gather(1,a2.unsqueeze(1)).squeeze(1)
                    tgt=rB+(GAMMA**N_STEP)*q_ns*(1-dB)
                td=tgt-q_sa; loss=(w*td.pow(2)).mean()
            opt.zero_grad(set_to_none=True)
            (scaler.scale(loss) if USE_SCALER else loss).backward()
            if USE_SCALER:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(online.parameters(),10)
                scaler.step(opt); scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(online.parameters(),10); opt.step()
            buf.update(idx,td.detach().cpu().numpy())
            step_count += 1
            if step_count % TARGET_SYNC == 0:
                target.load_state_dict(online.state_dict())

        R_hist.append(ep_R)
        if ep%100==0:
            m=np.mean(R_hist[-100:]); print(f"[ep {ep}] mean_R={m:.1f}")
            if m>best_R: best_R,best_ep=m,ep; torch.save(online.state_dict(),"report3.pth")
            if ep-best_ep>=EARLY_STOP: break

    # curves
    plt.figure(); plt.plot(R_hist); plt.xlabel("episode"); plt.ylabel("reward"); plt.tight_layout()
    plt.savefig("rl_curve.png",dpi=150)
    with PdfPages("rl_curve.pdf") as pdf: pdf.savefig()
    print(f"[train] finished – best mean_R {best_R:.1f} @ ep {best_ep}")

if __name__=="__main__":
    t0=time.time(); train()
    print(f"[train] runtime {time.time()-t0:.1f}s")
