import os, random, math, json, argparse, datetime
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

# -------------------------- deterministic behaviour --------------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True
set_seed()

# ----------------------------- hyper-parameters ------------------------------
HIST_LEN     = 193          #  t-192 … t
HORIZON      = 48           #  t+1 … t+48
BATCH        = 256
EPOCHS       = 150
EARLY_STOP   = 20
LR           = 1e-4         # Réduit de 3e-4 à 1e-4
WEIGHT_DECAY = 1e-4
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----------------------------- data utilities -------------------------------
def make_supervised(series: np.ndarray):
    """Transform 1-D price series into (X,Y) supervised pairs."""
    N = len(series) - HIST_LEN - HORIZON + 1
    X = np.zeros((N, 1, HIST_LEN),  dtype=np.float32)  # (N,C,L)
    Y = np.zeros((N, HORIZON),      dtype=np.float32)
    for i in range(N):
        window = series[i:i+HIST_LEN]
        target = series[i+HIST_LEN:i+HIST_LEN+HORIZON]
        X[i,0] = window
        Y[i]   = target
    return X, Y

def zscore_fit(x):
    mu, sigma = x.mean(), x.std()
    # Éviter la division par zéro
    if sigma == 0:
        sigma = 1.0
        print(f"Attention: écart-type = 0, utilisation de σ = 1")
    print(f"Normalisation: μ = {mu:.3f}, σ = {sigma:.3f}")
    normalized = (x - mu) / sigma
    # Vérifier les NaN/inf
    if np.any(np.isnan(normalized)) or np.any(np.isinf(normalized)):
        print("ERREUR: Valeurs NaN ou infinies après normalisation!")
    return normalized, mu, sigma

def zscore_apply(x, mu, sigma): return (x - mu) / sigma
def zscore_inv(x, mu, sigma):   return x * sigma + mu

# ------------------------------- 1-D CNN model -------------------------------
class PriceCNN(nn.Module):
    """Causal 1-D ConvNet → GlobalAvgPool → Dense forecast head."""
    def __init__(self, horizon=HORIZON):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=2, dilation=2), nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=4, dilation=4), nn.GELU(),
            nn.Conv1d(64,128, kernel_size=3, padding=8, dilation=8), nn.GELU(),
            nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(1),                # (N,128,1)
            nn.Flatten(),                           # (N,128)
            nn.Linear(128, 256), nn.GELU(),
            nn.Linear(256, horizon)                 # (N,48)
        )
    def forward(self,x): return self.net(x)

# --------------------------- training & evaluation ---------------------------
def train(model, loader, optim, scheduler, criterion):
    model.train(); epoch_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optim.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        
        # Vérifier si la loss est NaN
        if torch.isnan(loss):
            print("ERREUR: Loss est NaN!")
            return float('nan')
            
        loss.backward()
        
        # Gradient clipping pour éviter l'explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optim.step()
        epoch_loss += loss.item() * xb.size(0)
    scheduler.step()
    return epoch_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval(); loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = model(xb)
        loss += criterion(pred, yb).item() * xb.size(0)
    return loss / len(loader.dataset)

def mae_rmse(pred, true):
    mae  = np.mean(np.abs(pred-true))
    rmse = math.sqrt(np.mean((pred-true)**2))
    return mae, rmse

# ---------------------------------- main ------------------------------------
def main():
    print("===> Loading data")
    train_csv = pd.read_csv("price_train.csv")['Prices (EUR/MWh)'].values.astype(np.float32)
    test_csv  = pd.read_csv("price_test.csv").values.astype(np.float32)  # shape (400,193)

    # Vérifications initiales
    print(f"Train data: {len(train_csv)} points, range [{train_csv.min():.2f}, {train_csv.max():.2f}]")
    print(f"Test data: shape {test_csv.shape}")

    # Fit scaler on the whole training series
    train_norm, mu, sigma = zscore_fit(train_csv)

    # Prepare supervised train/val splits (last 7 days reserved for validation)
    X, Y = make_supervised(train_norm)
    
    # Vérifications datasets
    print(f"Supervised data: X shape {X.shape}, Y shape {Y.shape}")
    print(f"X range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"Y range: [{Y.min():.3f}, {Y.max():.3f}]")
    
    VAL_DAYS = 7*48
    idx_split = len(X) - VAL_DAYS
    X_train, Y_train = X[:idx_split], Y[:idx_split]
    X_val,   Y_val   = X[idx_split:], Y[idx_split:]

    train_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(Y_train)),
        batch_size=BATCH, shuffle=True, drop_last=True)
    val_loader   = DataLoader(TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(Y_val)),
        batch_size=BATCH, shuffle=False)

    print(f"Training samples: {len(X_train):,}, Validation: {len(X_val):,}")

    # Model, optimiser, scheduler
    model      = PriceCNN().to(DEVICE)
    criterion  = nn.L1Loss()                     # MAE → robust to spikes
    optimiser  = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler  = CosineAnnealingLR(optimiser, T_max=EPOCHS)

    # Vérifications modèle
    print(f"Modèle initialisé sur {DEVICE}")
    print(f"Nombre de paramètres: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test rapide du modèle avec un échantillon
    with torch.no_grad():
        test_input = torch.from_numpy(X_train[:1]).to(DEVICE)
        test_output = model(test_input)
        print(f"Test forward pass: input {test_input.shape} -> output {test_output.shape}")
        print(f"Output range: [{test_output.min().item():.3f}, {test_output.max().item():.3f}]")

    # Training loop with early stopping
    print(f"===> Début entraînement ({EPOCHS} époques max, early stop={EARLY_STOP})")
    best_loss, best_ep, losses_t, losses_v = np.inf, 0, [], []
    for epoch in range(1, EPOCHS+1):
        tloss = train(model, train_loader, optimiser, scheduler, criterion)
        vloss = evaluate(model, val_loader,   criterion)
        
        # Arrêter si NaN détecté
        if np.isnan(tloss) or np.isnan(vloss):
            print(f"ARRÊT: NaN détecté à l'époque {epoch}")
            break
            
        losses_t.append(tloss); losses_v.append(vloss)
        if vloss < best_loss - 1e-5:           # significant improvement
            best_loss, best_ep = vloss, epoch
            torch.save(model.state_dict(), "report1.pth")
        if epoch-best_ep >= EARLY_STOP: 
            print(f"Early stopping à l'époque {epoch} (meilleure: {best_ep})")
            break
        if epoch % 10 == 0 or epoch <= 3:  # Print plus fréquent au début
            print(f"Epoch {epoch:3d} | train {tloss:.4f} | val {vloss:.4f}")
            
    print(f"Entraînement terminé après {epoch} époques")
    
    if best_loss != np.inf:
        print(f"Best validation MAE={best_loss:.4f} (epoch {best_ep})")
    else:
        print("ERREUR: Aucun modèle valide sauvegardé!")
        return

    # ----------------------------- evaluation on test -----------------------------
    print("===> Evaluating on public test set")
    X_test = zscore_apply(test_csv, mu, sigma)[:,None,:].astype(np.float32)  # (400,1,193)
    with torch.no_grad():
        model.load_state_dict(torch.load("report1.pth", map_location=DEVICE))
        model.eval()
        preds_norm = model(torch.from_numpy(X_test).to(DEVICE)).cpu().numpy()
    preds = zscore_inv(preds_norm, mu, sigma)

    # Vérifications finales
    print(f"Prédictions: shape {preds.shape}, range [{preds.min():.2f}, {preds.max():.2f}]")
    if np.any(np.isnan(preds)):
        print("ATTENTION: Prédictions contiennent des NaN!")

    # Save answers
    pd.DataFrame(preds).to_csv("price_ans.csv", index=False, header=False)
    print("price_ans.csv écrit avec succès.")

    # Quick self-check against the validation slice in train set
    val_pred_norm = model(torch.from_numpy(X_val).to(DEVICE)).cpu().numpy()
    val_pred = zscore_inv(val_pred_norm, mu, sigma)
    val_true = zscore_inv(Y_val, mu, sigma)
    mae, rmse = mae_rmse(val_pred, val_true)
    print(f"Validation MAE: {mae:.3f},  RMSE: {rmse:.3f}")

    # Plot learning curve for the report
    plt.figure(figsize=(6,4))
    plt.plot(losses_t, label="train")
    plt.plot(losses_v, label="val")
    plt.xlabel("Epoch"); plt.ylabel("MAE loss"); plt.title("Learning curve")
    plt.legend(); plt.tight_layout(); plt.savefig("loss_curve.png", dpi=150)

if __name__ == "__main__":
    main()
