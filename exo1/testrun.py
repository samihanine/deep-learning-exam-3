#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────
# testrun1_improved.py  ·  Deep-Learning Final Exam – Exercise 1
# Version revue : split de validation plus réaliste, loss pondérée par horizon,
# normalisation calculée sur la seule partie « train », monitoring détaillé.
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
    DEVICE, AMP_DEVICE = "cuda", "cuda"
elif torch.backends.mps.is_available():
    DEVICE, AMP_DEVICE = "mps", "mps"
else:
    DEVICE = AMP_DEVICE = "cpu"  # AMP désactivé sur CPU
print(f"Using device: {DEVICE.upper()}")

MIXED_PREC = AMP_DEVICE in {"cuda", "mps"}
USE_SCALER = AMP_DEVICE == "cuda"  # GradScaler ≠ MPS
USE_COMPILE = AMP_DEVICE == "cuda"  # torch.compile() instable sur MPS
from torch import autocast
if USE_SCALER:
    from torch.cuda.amp import GradScaler
# ╰─────────────────────────────────────────────────────────────────────╯

# ╭──────────────────────────── Hyper-paramètres ───────────────────────╮
HIST_LEN = 193       # t … t‑192
HORIZON = 48         # t+1 … t+48
VAL_WEEKS = 4        # plage de validation temporelle
BATCH = 1024
EPOCHS = 60
EARLY_STOP = 10      # sur validation pondérée
LR = 3e-4
WEIGHT_DECAY = 1e-4
VAL_FREQ = 2
NUM_WORKERS = 0 if DEVICE == "mps" else min(8, os.cpu_count())
PIN_MEMORY = DEVICE == "cuda"
# ╰─────────────────────────────────────────────────────────────────────╯

# ╭──────────────────── Fix seed & matmul precision ────────────────────╮
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)
set_seed()
if version.parse(torch.__version__) >= version.parse("2.0"):
    torch.set_float32_matmul_precision("medium")
# ╰─────────────────────────────────────────────────────────────────────╯

# ╭──────────────────────── Data utilities ─────────────────────────────╮

def clean_data(series):
    """Linear interpolation + ffill/bfill – comme conseillé en cours."""
    if pd.isna(series).sum():
        series = pd.Series(series).interpolate("linear").ffill().bfill()
    return series.values.astype(np.float32)


def make_supervised(series):
    N = len(series) - HIST_LEN - HORIZON + 1
    X = np.zeros((N, 1, HIST_LEN), dtype=np.float32)
    Y = np.zeros((N, HORIZON), dtype=np.float32)
    for i in range(N):
        X[i, 0] = series[i : i + HIST_LEN]
        Y[i] = series[i + HIST_LEN : i + HIST_LEN + HORIZON]
    return X, Y

# ╰─────────────────────────────────────────────────────────────────────╯

# ╭─────────────────────────── Model ───────────────────────────────────╮
class PriceCNN(nn.Module):
    def __init__(self, horizon=HORIZON):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=2, dilation=2), nn.SiLU(inplace=True),
            nn.Conv1d(64, 64, 3, padding=4, dilation=4), nn.SiLU(inplace=True),
            nn.Conv1d(64, 128, 3, padding=8, dilation=8), nn.SiLU(inplace=True),
            nn.BatchNorm1d(128), nn.AdaptiveAvgPool1d(1)
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(128, 256), nn.SiLU(inplace=True),
            nn.Linear(256, horizon)
        )

    def forward(self, x):
        return self.head(self.cnn(x))

# ╰─────────────────────────────────────────────────────────────────────╯

# ╭─────────────────── Diagnostics helpers ─────────────────────────────╮

def horizon_mae(pred, true, horizons=(1, 6, 12, 24, 48)):
    return {h: float(np.mean(np.abs(pred[:, -h] - true[:, -h]))) for h in horizons}


def grad_norm(model):
    norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    return float(np.mean(norms)) if norms else 0.0

# ╰─────────────────────────────────────────────────────────────────────╯

# ╭───────────────────────── Extra monitoring & report ─────────────────╮
from matplotlib.backends.backend_pdf import PdfPages
import tabulate, psutil, platform, gc


def print_epoch_report(epoch, tloss, vloss, v_mae, gradn, lr):
    table = [
        ["epoch", epoch],
        ["train MAE", f"{tloss:.4f}"],
        ["val MAE", f"{vloss:.4f}"],
        ["horizon-1 MAE", f"{v_mae[1]:.4f}"],
        ["horizon-6 MAE", f"{v_mae[6]:.4f}"],
        ["horizon-48 MAE", f"{v_mae[48]:.4f}"],
        ["grad-norm", f"{gradn:.3f}"],
        ["learning-rate", f"{lr:.2e}"],
        ["RAM used (GB)", f"{psutil.Process().memory_info().rss/1e9: .2f}"],
    ]
    print("\n" + tabulate.tabulate(table, headers=["metric", "value"], tablefmt="github"))


def save_plots(losses_t, losses_v):
    plt.figure(figsize=(6, 4))
    plt.plot(losses_t, label="train")
    plt.plot([v for v in losses_v if not math.isnan(v)], label="val")
    plt.xlabel("epoch"); plt.ylabel("weighted MAE"); plt.legend(); plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)

    with PdfPages("loss_curve.pdf") as pdf:
        pdf.savefig(); plt.close()
        fig = plt.figure(figsize=(6, 4)); plt.axis("off")
        txt = (
            f"Machine: {platform.platform()}\n"
            f"Python : {platform.python_version()}\n"
            f"PyTorch: {torch.__version__}\n"
            f"Device : {DEVICE}\n\n"
            f"HIST_LEN  = {HIST_LEN}\n"
            f"HORIZON   = {HORIZON}\n"
            f"VAL_WEEKS = {VAL_WEEKS}\n"
            f"BATCH     = {BATCH}\n"
            f"EPOCHS    = {EPOCHS}\n"
            f"LR        = {LR}\n"
            f"WEIGHT_DECAY = {WEIGHT_DECAY}\n"
        )
        plt.text(0.0, 0.5, txt, va="center", ha="left", family="monospace", fontsize=9)
        pdf.savefig(); plt.close()

# ╰─────────────────────────────────────────────────────────────────────╯

# ╭─────────────────────── Weighted‑horizon MAE loss ───────────────────╮
WEIGHTS = torch.linspace(1.0, 0.2, HORIZON)  # décroissant – penalise court‑terme

def weighted_l1(pred, target):
    w = WEIGHTS.to(pred.device).view(1, -1)
    return torch.mean(torch.abs(pred - target) * w)
# ╰─────────────────────────────────────────────────────────────────────╯


def main():
    # ───── Load & split data ───────────────────────────────────────────
    series = pd.read_csv("price_train.csv")["Prices (EUR/MWh)"].values
    series = clean_data(series)

    # 4 semaines de validation (≃ 1344 pas)
    val_steps = VAL_WEEKS * 7 * 48
    mu, sigma = series[: -val_steps].mean(), series[: -val_steps].std() or 1.0
    series_norm = (series - mu) / sigma

    X, Y = make_supervised(series_norm)
    X_tr, Y_tr = X[: -val_steps], Y[: -val_steps]
    X_val, Y_val = X[-val_steps:], Y[-val_steps:]

    test_csv = pd.read_csv("price_test.csv").values

    # ───── Dataloaders ────────────────────────────────────────────────
    dl_train = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(Y_tr)),
        batch_size=BATCH, shuffle=True, drop_last=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )
    dl_val = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val)),
        batch_size=BATCH, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )

    # ───── Model & optimiser ─────────────────────────────────────────
    model = PriceCNN().to(DEVICE)
    if USE_COMPILE:
        model = torch.compile(model)
    criterion = weighted_l1
    optim = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = CosineAnnealingLR(optim, T_max=EPOCHS)
    scaler = GradScaler() if USE_SCALER else None

    # ───── Training loop ─────────────────────────────────────────────
    best_loss, best_ep = float("inf"), 0
    losses_t, losses_v, lr_hist = [], [], []
    t_data = t_fwd = t_bwd = 0.0

    print("\n────────────── Training ──────────────")
    for epoch in tqdm(range(1, EPOCHS + 1), unit="ep", desc="Training"):
        model.train(); running = 0.0
        for xb, yb in dl_train:
            tic = time.perf_counter(); xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            t_data += time.perf_counter() - tic
            optim.zero_grad(set_to_none=True)
            with autocast(device_type=AMP_DEVICE, enabled=MIXED_PREC):
                tic = time.perf_counter(); pred = model(xb); loss = criterion(pred, yb)
                t_fwd += time.perf_counter() - tic
            if USE_SCALER:
                scaler.scale(loss).backward(); scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                tic = time.perf_counter(); scaler.step(optim); scaler.update()
            else:
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                tic = time.perf_counter(); optim.step()
            t_bwd += time.perf_counter() - tic
            running += loss.item() * xb.size(0)

        tloss = running / len(dl_train.dataset)
        sched.step(); lr_hist.append(optim.param_groups[0]["lr"])

        # Validation
        model.eval(); vloss = float("nan"); v_mae = {1: np.nan, 6: np.nan, 48: np.nan}
        if epoch % VAL_FREQ == 0:
            with torch.no_grad():
                summ = 0.0
                for xb, yb in dl_val:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    summ += criterion(model(xb), yb).item() * xb.size(0)
                vloss = summ / len(dl_val.dataset)

                # MAE dé‑scalée par horizons
                v_pred = model(torch.from_numpy(X_val).to(DEVICE)).detach().cpu().numpy()
                v_mae = horizon_mae((v_pred * sigma + mu), (Y_val * sigma + mu))

                print_epoch_report(epoch, tloss, vloss, v_mae, grad_norm(model), lr_hist[-1])

        losses_t.append(tloss); losses_v.append(vloss)
        if not math.isnan(vloss) and vloss < best_loss - 1e-5:
            best_loss, best_ep = vloss, epoch
            torch.save(model.state_dict(), "report1.pth")

        if epoch - best_ep >= EARLY_STOP:
            print(f"\nEarly-stop at epoch {epoch} (best={best_ep})")
            break

    print("\nTraining done.  data/fwd/back per‑ep (s):",
          f"{t_data/epoch:.2f}/{t_fwd/epoch:.2f}/{t_bwd/epoch:.2f}")

    # ───── Test inference ────────────────────────────────────────────
    model.load_state_dict(torch.load("report1.pth", map_location=DEVICE))
    X_test = ((test_csv - mu) / sigma)[:, None, :].astype(np.float32)
    with torch.no_grad(), autocast(device_type=AMP_DEVICE, enabled=MIXED_PREC):
        preds = model(torch.from_numpy(X_test).to(DEVICE)).detach().cpu().numpy()
    preds = preds * sigma + mu
    pd.DataFrame(preds).to_csv("price_ans.csv", index=False, header=False)
    print("\nprice_ans.csv written.")

    # quick self‑check sur la vraie MAE / RMSE
    val_pred = model(torch.from_numpy(X_val).to(DEVICE)).detach().cpu().numpy() * sigma + mu
    val_true = Y_val * sigma + mu
    mae = np.mean(np.abs(val_pred - val_true))
    rmse = math.sqrt(np.mean((val_pred - val_true) ** 2))
    print(f"Validation MAE {mae:.3f} | RMSE {rmse:.3f}")

    save_plots(losses_t, losses_v)


if __name__ == "__main__":
    main()
