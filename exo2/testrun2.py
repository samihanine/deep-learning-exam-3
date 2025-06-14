#!/usr/bin/env python3
"""
Deep-Learning Final Exam – Exercise 2
Author : <votre nom>
Date   : 2025-06-13

Structure attendue du dossier :
.
├── generator_dataset
│   ├── train
│   │   ├── class_1
│   │   ├── class_2
│   │   ├── class_3
│   │   └── class_4
│   └── test
│       ├── image_001.png
│       └── … image_400.png
└── testrun2.py   (ce fichier)

Le script crée :
  • report2.pth          – poids du meilleur modèle
  • class_result2.csv    – prédictions sur le dossier « test »
  • loss_acc_curve.png   – courbes perte / accuracies
  • loss_acc_curve.pdf   – PDF commenté (métadonnées + courbes)

Il suit la même philosophie que testrun1.py (exo 1) : 
  – optimisation CosineAnnealingLR
  – mix-precision (AMP) + torch.compile() si CUDA
  – tableau de stats façon tabulate
"""

import os, time, math, random, datetime, platform, psutil, gc
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from packaging import version
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets

# ╭──────────────────────────── Device & AMP ───────────────────────────╮
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
# ╰─────────────────────────────────────────────────────────────────────╯

# ╭────────────────────────── Hyper-paramètres ─────────────────────────╮
IMG_SIZE   = 224
BATCH      = 64
EPOCHS     = 25
EARLY_STOP = 5
LR         = 3e-4
WEIGHT_DECAY = 1e-4
VAL_SPLIT  = 0.15          # pourcentage du train gardé en validation
SEED       = 42
NUM_WORKERS = 0 if DEVICE == "mps" else min(8, os.cpu_count())
PIN_MEMORY  = DEVICE == "cuda"
VAL_FREQ    = 1
# ╰─────────────────────────────────────────────────────────────────────╯


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)

set_seed()

if version.parse(torch.__version__) >= version.parse("2.0"):
    torch.set_float32_matmul_precision("high")

# ╭───────────────────────── Data transforms ───────────────────────────╮
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.3,0.3,0.3,0.1)], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize(int(IMG_SIZE*1.15)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
# ╰─────────────────────────────────────────────────────────────────────╯

# ╭────────────────────── Datasets & Dataloaders ───────────────────────╮
DATA_ROOT = Path("generator_dataset")
train_dir = DATA_ROOT/"train"
test_dir  = DATA_ROOT/"test"

full_ds = datasets.ImageFolder(train_dir, transform=train_tf)
class_names = full_ds.classes          # ➜ ['class_1', …, 'class_4']
NUM_CLASSES = len(class_names)

# split train/val
val_size = int(len(full_ds)*VAL_SPLIT)
train_size = len(full_ds) - val_size
train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_size, val_size],
                                                generator=torch.Generator().manual_seed(SEED))
# val dataset must use val_tf
val_ds.dataset.transform = val_tf

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

# test dataset (no labels)
class TestFolder(datasets.ImageFolder):
    def __init__(self, root, transform):
        # fake class folder to reuse ImageFolder
        super().__init__(root, transform)
        self.imgs  = self.samples = [(p, 0) for p in sorted(Path(root).glob("*.png"))]

test_ds = TestFolder(test_dir, val_tf)
test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
# ╰─────────────────────────────────────────────────────────────────────╯

# ╭──────────────────────────── Model ───────────────────────────────────╮
def get_model(num_classes=NUM_CLASSES):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # freeze early layers (optionnel)
    for name, param in model.named_parameters():
        if "layer4" not in name:   # fine-tune only the last block + fc
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

model = get_model().to(DEVICE)
if USE_COMPILE:
    model = torch.compile(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = GradScaler() if USE_SCALER else None
# ╰─────────────────────────────────────────────────────────────────────╯

# ╭─────────────────────── Helpers & Monitoring ────────────────────────╮
from tabulate import tabulate
def accuracy(out, y):
    _, pred = torch.max(out, 1)
    return (pred == y).float().mean().item()

def print_epoch(epoch, tl, ta, vl, va, lr):
    table = [
        ["epoch", epoch],
        ["train loss", f"{tl:.4f}"],
        ["train acc",  f"{ta*100:.2f}%"],
        ["val loss",   f"{vl:.4f}"],
        ["val acc",    f"{va*100:.2f}%"],
        ["learning-rate", f"{lr:.2e}"],
        ["RAM used (GB)", f"{psutil.Process().memory_info().rss/1e9: .2f}"],
    ]
    print("\n"+tabulate(table, headers=["metric","value"], tablefmt="github"))
# ╰─────────────────────────────────────────────────────────────────────╯

# ╭──────────────────────────── Train loop ──────────────────────────────╮
best_acc, best_ep = 0.0, 0
hist = {"tl":[], "vl":[], "ta":[], "va":[]}

print("\n────────────── Training ──────────────")
for epoch in tqdm(range(1, EPOCHS+1), unit="ep", desc="Training"):
    # --- training ---
    model.train()
    running_loss, running_acc = 0.0, 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=AMP_DEVICE, enabled=MIXED_PREC):
            out = model(xb)
            loss = criterion(out, yb)
        if USE_SCALER:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running_loss += loss.item()*xb.size(0)
        running_acc  += accuracy(out, yb)*xb.size(0)
    tl = running_loss/len(train_loader.dataset)
    ta = running_acc/len(train_loader.dataset)

    # --- validation ---
    if epoch % VAL_FREQ == 0:
        model.eval()
        with torch.no_grad():
            vl_sum, va_sum = 0.0, 0.0
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                vl_sum += criterion(out, yb).item()*xb.size(0)
                va_sum += accuracy(out, yb)*xb.size(0)
        vl = vl_sum/len(val_loader.dataset)
        va = va_sum/len(val_loader.dataset)
    else:
        vl, va = math.nan, math.nan

    scheduler.step()

    hist["tl"].append(tl); hist["ta"].append(ta)
    hist["vl"].append(vl); hist["va"].append(va)

    if not math.isnan(va) and va > best_acc+1e-4:
        best_acc, best_ep = va, epoch
        torch.save(model.state_dict(), "report2.pth")

    print_epoch(epoch, tl, ta, vl, va, scheduler.get_last_lr()[0])

    if epoch-best_ep >= EARLY_STOP:
        print(f"\nEarly-stopping at epoch {epoch} (best={best_ep})")
        break

print(f"\nBest val acc: {best_acc*100:.2f}% at epoch {best_ep}")

# ╭──────────────────────── Courbes (PNG + PDF) ────────────────────────╮
plt.figure(figsize=(6,4))
plt.plot(hist["tl"], label="train-loss")
plt.plot([v for v in hist["vl"] if not math.isnan(v)], label="val-loss")
plt.ylabel("loss"); plt.xlabel("epoch"); plt.legend(); plt.tight_layout()
plt.savefig("loss_acc_curve.png", dpi=150)

with PdfPages("loss_acc_curve.pdf") as pdf:
    pdf.savefig(); plt.close()
    fig = plt.figure(figsize=(6,4)); plt.axis("off")
    txt = (f"Model : ResNet18 fine-tune\n"
           f"Device: {DEVICE}\n"
           f"Best  : {best_acc*100:.2f}% @ {best_ep}\n"
           f"Date  : {datetime.datetime.now()}\n"
           f"Python: {platform.python_version()} | Torch {torch.__version__}\n"
           f"IMG_SIZE={IMG_SIZE}, BATCH={BATCH}, LR={LR}, WD={WEIGHT_DECAY}\n")
    plt.text(0.0,0.5,txt,va="center",ha="left",family="monospace",fontsize=9)
    pdf.savefig(); plt.close()
# ╰─────────────────────────────────────────────────────────────────────╯


# ╭────────────────────────── Inference test ───────────────────────────╮
print("\n──────────── Inference on test set ────────────")
model.load_state_dict(torch.load("report2.pth", map_location=DEVICE))
model.eval()
preds = []
with torch.no_grad():
    for xb,_ in tqdm(test_loader, desc="Predict"):
        xb = xb.to(DEVICE)
        out = model(xb)
        _, pred = torch.max(out, 1)
        preds.extend(pred.cpu().numpy())
# map idx→class names
pred_labels = [class_names[idx] for idx in preds]
# filenames dans l'ordre
file_names = [p.name for p,_ in test_ds.samples]
pd.DataFrame({"filename":file_names, "class":pred_labels}).to_csv(
        "class_result2.csv", index=False)
print("class_result2.csv written.")

# ╭──────────────────────── Fin & récapitulatif ────────────────────────╮
print("\nDone – fichiers générés :")
for f in ["report2.pth", "class_result2.csv", "loss_acc_curve.png", "loss_acc_curve.pdf"]:
    print(" •", f, f"({Path(f).stat().st_size/1e6:.2f} MB)" if Path(f).exists() else "(missing)")
# ╰─────────────────────────────────────────────────────────────────────╯
