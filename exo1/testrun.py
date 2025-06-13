#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────
# testrun1_m3pro.py  ·  Deep-Learning Final Exam – Exercise 1
# (Apple Silicon & GPU-CUDA friendly · verbose diagnostics)
# ──────────────────────────────────────────────────────────────────
import os, random, math, time, datetime
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from packaging import version

# ╭──────────────────────── Device & AMP setup ─────────────────────────╮
if torch.cuda.is_available():
    DEVICE = "cuda"
    AMP_DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    AMP_DEVICE = "mps"
else:
    DEVICE = "cpu"
    AMP_DEVICE = "cpu"          # AMP désactivé sur CPU
print(f"Using device: {DEVICE.upper()}")

MIXED_PREC = AMP_DEVICE in {"cuda", "mps"}          # autocast disponible
USE_SCALER = AMP_DEVICE == "cuda"                   # GradScaler ≠ MPS
USE_COMPILE = AMP_DEVICE == "cuda"                  # torch.compile() instable sur MPS
from torch import autocast
if USE_SCALER:
    from torch.cuda.amp import GradScaler

# ╰─────────────────────────────────────────────────────────────────────╯

# ╭──────────────────────────── Hyper-paramètres ───────────────────────╮
HIST_LEN     = 193
HORIZON      = 48
BATCH        = 1024
EPOCHS       = 60
EARLY_STOP   = 10
LR           = 3e-4
WEIGHT_DECAY = 1e-4
VAL_FREQ     = 2
# Réduction des workers sur MPS pour éviter les crashes
NUM_WORKERS  = 0 if DEVICE == "mps" else min(8, os.cpu_count())
PIN_MEMORY   = DEVICE == "cuda"
# ╰─────────────────────────────────────────────────────────────────────╯


# ╭──────────────────── Fix seed & matmul precision ────────────────────╮
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)
set_seed()
if version.parse(torch.__version__) >= version.parse("2.0"):
    torch.set_float32_matmul_precision("medium")  # Apple Silicon accélère
# ╰─────────────────────────────────────────────────────────────────────╯


# ╭──────────────────────── Data utilities ─────────────────────────────╮
def clean_data(series):
    """Linear interpolation + ffill/bfill – comme conseillé en cours."""
    if pd.isna(series).sum():
        series = (pd.Series(series).interpolate("linear").ffill().bfill())
    return series.values.astype(np.float32)

def make_supervised(series):
    N = len(series) - HIST_LEN - HORIZON + 1
    X = np.zeros((N,1,HIST_LEN), dtype=np.float32)
    Y = np.zeros((N,HORIZON),   dtype=np.float32)
    for i in range(N):
        X[i,0] = series[i:i+HIST_LEN]
        Y[i]   = series[i+HIST_LEN:i+HIST_LEN+HORIZON]
    return X, Y

def zscore_fit(x):
    mu, sigma = x.mean(), x.std() or 1.0
    return (x-mu)/sigma, mu, sigma
zscore_apply = lambda x,m,s: (x-m)/s
zscore_inv   = lambda x,m,s: x*s + m
# ╰─────────────────────────────────────────────────────────────────────╯


# ╭─────────────────────────── Model ───────────────────────────────────╮
class PriceCNN(nn.Module):
    def __init__(self, horizon=HORIZON):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1,64,3,padding=2,dilation=2), nn.SiLU(inplace=True),
            nn.Conv1d(64,64,3,padding=4,dilation=4), nn.SiLU(inplace=True),
            nn.Conv1d(64,128,3,padding=8,dilation=8), nn.SiLU(inplace=True),
            nn.BatchNorm1d(128), nn.AdaptiveAvgPool1d(1)
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(128,256), nn.SiLU(inplace=True),
            nn.Linear(256, horizon)
        )
    def forward(self,x): return self.head(self.cnn(x))
# ╰─────────────────────────────────────────────────────────────────────╯


# ╭─────────────────── Diagnostics helpers ─────────────────────────────╮
def horizon_mae(pred, true, horizons=(1,6,12,24,48)):
    return {h: float(np.mean(np.abs(pred[:,-h]-true[:,-h]))) for h in horizons}

def grad_norm(model):
    norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    return float(np.mean(norms)) if norms else 0.0
# ╰─────────────────────────────────────────────────────────────────────╯


# ╭───────────────────────── Extra monitoring & report ─────────────────────────╮
from matplotlib.backends.backend_pdf import PdfPages
import tabulate, psutil, platform, gc

def print_epoch_report(epoch, tloss, vloss, v_mae, gradn, lr):
    table = [
        ["epoch",           epoch                     ],
        ["train MAE",       f"{tloss:.4f}"            ],
        ["val MAE",         f"{vloss:.4f}"            ],
        ["horizon-1 MAE",   f"{v_mae[1]:.4f}"         ],
        ["horizon-6 MAE",   f"{v_mae[6]:.4f}"         ],
        ["horizon-48 MAE",  f"{v_mae[48]:.4f}"        ],
        ["grad-norm",       f"{gradn:.3f}"            ],
        ["learning-rate",   f"{lr:.2e}"               ],
        ["RAM used (GB)",   f"{psutil.Process().memory_info().rss/1e9: .2f}"],
    ]
    print("\n"+tabulate.tabulate(table, headers=["metric","value"], tablefmt="github"))

def save_plots(losses_t, losses_v):
    plt.figure(figsize=(6,4))
    plt.plot(losses_t, label="train")
    plt.plot([v for v in losses_v if not math.isnan(v)], label="val")
    plt.xlabel("epoch"); plt.ylabel("MAE"); plt.legend(); plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)

    # En­registrer aussi dans un PDF autonome
    with PdfPages("loss_curve.pdf") as pdf:
        pdf.savefig()             # page 1 : courbe de perte
        plt.close()
        # page 2 : config système + hyper-params
        fig = plt.figure(figsize=(6,4)); plt.axis("off")
        txt = (
            f"Machine: {platform.platform()}\n"
            f"Python : {platform.python_version()}\n"
            f"PyTorch: {torch.__version__}\n"
            f"Device : {DEVICE}\n\n"
            f"HIST_LEN  = {HIST_LEN}\n"
            f"HORIZON   = {HORIZON}\n"
            f"BATCH     = {BATCH}\n"
            f"EPOCHS    = {EPOCHS}\n"
            f"LR        = {LR}\n"
            f"WEIGHT_DECAY = {WEIGHT_DECAY}\n"
        )
        plt.text(0.0, 0.5, txt, va="center", ha="left", family="monospace", fontsize=9)
        pdf.savefig(); plt.close()
# ╰─────────────────────────────────────────────────────────────────────────────╯


def main():
    # ───── Load data ───────────────────────────────────────────────────
    train_csv = pd.read_csv("price_train.csv")['Prices (EUR/MWh)'].values
    test_csv  = pd.read_csv("price_test.csv").values
    train_csv = clean_data(train_csv)

    train_norm, mu, sigma = zscore_fit(train_csv)
    X, Y = make_supervised(train_norm)
    val_split = 7*48
    X_tr, Y_tr = X[:-val_split], Y[:-val_split]
    X_val, Y_val = X[-val_split:], Y[-val_split:]

    # ───── Vérifier si les fichiers existent déjà ─────────────────────
    model_exists = os.path.exists("report1.pth")
    results_exist = os.path.exists("price_ans.csv")
    
    if model_exists and results_exist:
        print("\n🔄 Fichiers existants détectés - Chargement du modèle pré-entraîné...")
        model = PriceCNN().to(DEVICE)
        model.load_state_dict(torch.load("report1.pth", map_location=DEVICE))
        print("✅ Modèle chargé depuis report1.pth")
        print("✅ Résultats existants dans price_ans.csv")
        
        # Validation rapide
        model.eval()
        with torch.no_grad():
            val_pred = zscore_inv(
                model(torch.from_numpy(X_val).to(DEVICE)).detach().cpu().numpy(), mu, sigma)
            val_true = zscore_inv(Y_val, mu, sigma)
            mae = np.mean(np.abs(val_pred-val_true))
            rmse= math.sqrt(np.mean((val_pred-val_true)**2))
            print(f"Validation MAE {mae:.3f} | RMSE {rmse:.3f}")
        
        return

    # ───── Entraînement (seulement si fichiers n'existent pas) ─────────
    print("\n🚀 Démarrage de l'entraînement...")
    
    dl_train = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(Y_tr)),
                          batch_size=BATCH, shuffle=True, drop_last=True,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    dl_val   = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val)),
                          batch_size=BATCH, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # ───── Model & optimiser ───────────────────────────────────────────
    model = PriceCNN().to(DEVICE)
    if USE_COMPILE:
        model = torch.compile(model)
    criterion = nn.L1Loss()
    optim     = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched     = CosineAnnealingLR(optim, T_max=EPOCHS)
    scaler    = GradScaler() if USE_SCALER else None  # GradScaler pas impl. en MPS

    # ───── Training loop ───────────────────────────────────────────────
    best_loss, best_ep = float('inf'), 0
    t_data = t_fwd = t_bwd = 0.0
    losses_t, losses_v, lr_hist = [], [], []
    t_global0 = time.perf_counter()

    print("\n────────────── Training ──────────────")
    for epoch in tqdm(range(1, EPOCHS+1), unit="ep", desc="Training"):
        ep_start = time.perf_counter()
        model.train()
        running_loss = 0.0
        
        for xb, yb in dl_train:
            tic = time.perf_counter(); xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            t_data += time.perf_counter()-tic

            optim.zero_grad(set_to_none=True)
            with autocast(device_type=AMP_DEVICE, enabled=MIXED_PREC):
                tic = time.perf_counter()
                pred = model(xb); loss = criterion(pred, yb)
                t_fwd += time.perf_counter()-tic

            if USE_SCALER:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                tic = time.perf_counter(); scaler.step(optim); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                tic = time.perf_counter(); optim.step()
            t_bwd += time.perf_counter()-tic

            running_loss += loss.item()*xb.size(0)

        tloss = running_loss/len(dl_train.dataset)
        sched.step(); lr_hist.append(optim.param_groups[0]['lr'])

        # Validation + diagnostics
        if epoch % VAL_FREQ == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for xb, yb in dl_val:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    val_loss += criterion(model(xb), yb).item()*xb.size(0)
                vloss = val_loss/len(dl_val.dataset)

                # MAE par horizons (simplifié)
                v_pred = model(torch.from_numpy(X_val).to(DEVICE)).detach().cpu().numpy()
                v_mae  = horizon_mae(zscore_inv(v_pred,mu,sigma),
                                     zscore_inv(Y_val,mu,sigma))
                
                gradn = grad_norm(model)
                print_epoch_report(epoch, tloss, vloss, v_mae, gradn, lr_hist[-1])
                
                if epoch % 10 == 0:  # Print détaillé tous les 10 epochs seulement
                    print(f"\nEpoch {epoch:>3d} | train {tloss:.4f} | val {vloss:.4f}")
                    print(f"  MAE horizons : {v_mae}")

        else:
            vloss = float('nan')

        losses_t.append(tloss); losses_v.append(vloss)
        if not math.isnan(vloss) and vloss < best_loss - 1e-5:
            best_loss, best_ep = vloss, epoch
            torch.save(model.state_dict(), "report1.pth")

        if epoch - best_ep >= EARLY_STOP:
            print(f"\nEarly-stop at epoch {epoch} (best={best_ep})")
            break

    total_min = (time.perf_counter()-t_global0)/60
    print(f"\nTraining done in {total_min:.1f} min, "
          f"data/fwd/back ≈ {t_data/epoch:.2f}/{t_fwd/epoch:.2f}/{t_bwd/epoch:.2f}s per epoch")

    # ───── Test inference ──────────────────────────────────────────────
    model.load_state_dict(torch.load("report1.pth", map_location=DEVICE))
    model.eval()
    X_test = zscore_apply(test_csv, mu, sigma)[:,None,:].astype(np.float32)
    with torch.no_grad(), autocast(device_type=AMP_DEVICE, enabled=MIXED_PREC):
        preds_norm = model(torch.from_numpy(X_test).to(DEVICE)).detach().cpu().numpy()
    preds = zscore_inv(preds_norm, mu, sigma)
    pd.DataFrame(preds).to_csv("price_ans.csv", index=False, header=False)
    print("\nprice_ans.csv written.")

    # quick self-check
    val_pred = zscore_inv(
        model(torch.from_numpy(X_val).to(DEVICE)).detach().cpu().numpy(), mu, sigma)
    val_true = zscore_inv(Y_val, mu, sigma)
    mae = np.mean(np.abs(val_pred-val_true))
    rmse= math.sqrt(np.mean((val_pred-val_true)**2))
    print(f"Validation MAE {mae:.3f} | RMSE {rmse:.3f}")

    # curves
    save_plots(losses_t, losses_v)

if __name__ == "__main__":
    main()
