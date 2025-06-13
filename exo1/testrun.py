#!/usr/bin/env python3
# ────────────────────────────────────────────────────────────────────────────────
# testrun1.py (optimised)  ·  Deep-Learning Final Exam – Exercise 1
# ────────────────────────────────────────────────────────────────────────────────
# ‣ Major speed-ups added:
#   • DataLoader with workers & pin-memory
#   • Mixed-precision (autocast + GradScaler)
#   • Larger batch (1024) + LR 3e-4 + shorter training schedule (60 epochs)
#   • Early-stopping patience 10
#   • Validation every 2 epochs
#   • torch.compile() if PyTorch ≥ 2.0
#   • Global model compiled once (improves ~10 %)
#   • Inetplace SiLU activations (cheaper than GELU)
#   • progress-bar retained
# ────────────────────────────────────────────────────────────────────────────────
import os, random, math, datetime
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler

# ───────────────────────── deterministic behaviour ─────────────────────────────
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True
set_seed()

# ───────────────────────────── hyper-parameters ────────────────────────────────
HIST_LEN     = 193          #  t-192 … t
HORIZON      = 48           #  t+1 … t+48
BATCH        = 1024         #  ↗ VRAM-permitting
EPOCHS       = 60
EARLY_STOP   = 10
LR           = 3e-4
WEIGHT_DECAY = 1e-4
VAL_FREQ     = 2            # validate every k epochs
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
MIXED_PREC   = torch.cuda.is_available()          # use AMP if on GPU
NUM_WORKERS  = max(os.cpu_count()-2, 2)
PIN_MEMORY   = torch.cuda.is_available()

# ───────────────────────────── data utilities ─────────────────────────────────
def clean_data(series):
    nan_count = pd.isna(series).sum()
    if nan_count:
        series = (pd.Series(series)
                  .interpolate("linear")
                  .ffill()
                  .bfill())
    return series.values

def make_supervised(series: np.ndarray):
    N = len(series) - HIST_LEN - HORIZON + 1
    X = np.zeros((N, 1, HIST_LEN),  dtype=np.float32)
    Y = np.zeros((N, HORIZON),      dtype=np.float32)
    for i in range(N):
        X[i,0] = series[i:i+HIST_LEN]
        Y[i]   = series[i+HIST_LEN:i+HIST_LEN+HORIZON]
    return X, Y

def zscore_fit(x):
    mu, sigma = x.mean(), x.std() or 1.0
    return (x - mu)/sigma, mu, sigma
zscore_apply = lambda x, m, s: (x-m)/s
zscore_inv   = lambda x, m, s:  x*s + m

# ───────────────────────────── 1-D CNN model ──────────────────────────────────
class PriceCNN(nn.Module):
    def __init__(self, horizon=HORIZON):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=2, dilation=2), nn.SiLU(inplace=True),
            nn.Conv1d(64,64, 3, padding=4, dilation=4), nn.SiLU(inplace=True),
            nn.Conv1d(64,128,3, padding=8, dilation=8), nn.SiLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(1)
        )
        self.mlp = nn.Sequential(
            nn.Flatten(), nn.Linear(128,256), nn.SiLU(inplace=True),
            nn.Linear(256,horizon)
        )
    def forward(self,x): return self.mlp(self.cnn(x))

# ──────────────────────── training & evaluation utils ─────────────────────────
def train_one_epoch(model, loader, optim, scaler, criterion):
    model.train(); running = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optim.zero_grad(set_to_none=True)
        with autocast('cuda', enabled=MIXED_PREC):
            pred = model(xb)
            loss = criterion(pred, yb)
        if torch.isnan(loss): raise RuntimeError("Loss became NaN")
        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optim); scaler.update()
        running += loss.item() * xb.size(0)
    return running / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval(); running = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = model(xb)
        running += criterion(pred, yb).item() * xb.size(0)
    return running / len(loader.dataset)

def mae_rmse(pred,true):
    mae  = np.mean(np.abs(pred-true))
    rmse = math.sqrt(np.mean((pred-true)**2))
    return mae, rmse

# ────────────────────────────────── main ──────────────────────────────────────
def main():
    # ───── load ----------------------------------------------------------------
    train_csv = pd.read_csv("price_train.csv")['Prices (EUR/MWh)'].values.astype(np.float32)
    test_csv  = pd.read_csv("price_test.csv").values.astype(np.float32)

    train_csv = clean_data(train_csv)
    train_norm, mu, sigma = zscore_fit(train_csv)

    X, Y = make_supervised(train_norm)
    idx_split = len(X) - 7*48
    X_train, Y_train = X[:idx_split], Y[:idx_split]
    X_val,   Y_val   = X[idx_split:], Y[idx_split:]

    dl_train = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train)),
        batch_size=BATCH, shuffle=True, drop_last=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    dl_val   = DataLoader(
        TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(Y_val)),
        batch_size=BATCH, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # ───── model ----------------------------------------------------------------
    model = PriceCNN().to(DEVICE)
    if torch.__version__ >= "2": model = torch.compile(model)  # PyTorch 2 speed-up
    criterion  = nn.L1Loss()
    optimiser  = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler  = CosineAnnealingLR(optimiser, T_max=EPOCHS)
    scaler     = GradScaler('cuda', enabled=MIXED_PREC)

    # ───── training loop --------------------------------------------------------
    best_loss, best_ep = float('inf'), 0
    losses_t, losses_v = [], []
    t0 = datetime.datetime.now()
    pbar = tqdm(range(1,EPOCHS+1), unit="epoch", desc="Training")
    for epoch in pbar:
        tloss = train_one_epoch(model, dl_train, optimiser, scaler, criterion)
        vloss = evaluate(model, dl_val, criterion) if epoch%VAL_FREQ==0 else np.nan
        scheduler.step()

        # logging
        losses_t.append(tloss); losses_v.append(vloss)
        if not np.isnan(vloss) and vloss < best_loss - 1e-5:
            best_loss, best_ep = vloss, epoch
            torch.save(model.state_dict(), "report1.pth")
        pbar.set_postfix(train=f"{tloss:.4f}", val=f"{vloss:.4f}", best=f"{best_loss:.4f}")

        # early stop
        if epoch - best_ep >= EARLY_STOP: break
    pbar.close()
    print(f"Finished after {epoch} epochs – best MAE {best_loss:.4f} (epoch {best_ep}) "
          f"in {(datetime.datetime.now()-t0).seconds//60} min.")

    # ───── inference on test ----------------------------------------------------
    X_test = zscore_apply(test_csv, mu, sigma)[:,None,:].astype(np.float32)
    with torch.no_grad():
        model.load_state_dict(torch.load("report1.pth", map_location=DEVICE))
        model.eval()
        preds_norm = model(torch.from_numpy(X_test).to(DEVICE)).cpu().numpy()
    preds = zscore_inv(preds_norm, mu, sigma)
    pd.DataFrame(preds).to_csv("price_ans.csv", index=False, header=False)
    print("price_ans.csv written.")

    # quick self-check
    val_pred = zscore_inv(model(torch.from_numpy(X_val).to(DEVICE)).cpu().numpy(), mu, sigma)
    val_true = zscore_inv(Y_val, mu, sigma)
    mae, rmse = mae_rmse(val_pred, val_true)
    print(f"Validation MAE {mae:.3f} | RMSE {rmse:.3f}")

    # learning curve
    plt.figure(figsize=(6,4))
    plt.plot(losses_t, label="train")
    plt.plot([v for v in losses_v if not np.isnan(v)], label="val")
    plt.xlabel("epoch"); plt.ylabel("MAE"); plt.legend(); plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)

if __name__ == "__main__":
    main()
