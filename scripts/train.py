import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bpm.data_loading import load_windows_for_subjects, subjects_by_dataset_from_manifest
from bpm.dataset import WindowDataset, make_weighted_sampler
from bpm.model import CNNBiLSTM
from bpm.split import subject_wise_split

MANIFEST_PATH = ROOT / "cache" / "manifest.csv"
CACHE_DIR = ROOT / "cache" / "windows"
CHECKPOINT_PATH = ROOT / "cache" / "best.pt"

BATCH_SIZE = 64
MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 10
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def mae(pred, target):
    return (pred - target).abs().mean().item()


def run_epoch(model, loader, optimizer, loss_fn, train):
    model.train(train)
    total_loss, total_mae, n = 0.0, 0.0, 0
    for windows, labels in loader:
        windows, labels = windows.to(DEVICE), labels.to(DEVICE)
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            preds = model(windows)
            loss = loss_fn(preds, labels)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(labels)
        total_mae += mae(preds, labels) * len(labels)
        n += len(labels)
    return total_loss / n, total_mae / n


def main():
    torch.manual_seed(42)  # model init, dropout and WeightedRandomSampler all draw from torch's global RNG
    torch.backends.cudnn.deterministic = True  # seed alone isn't enough on CUDA: cuDNN conv/LSTM kernels drift run to run
    subjects_by_dataset = subjects_by_dataset_from_manifest(MANIFEST_PATH)
    split = subject_wise_split(subjects_by_dataset, seed=42)

    train_data = load_windows_for_subjects(sorted(split["train"]), CACHE_DIR)
    val_data = load_windows_for_subjects(sorted(split["val"]), CACHE_DIR)

    train_ds = WindowDataset(
        train_data["windows"], train_data["labels"], augment=True
    )
    val_ds = WindowDataset(
        val_data["windows"], val_data["labels"], augment=False
    )

    sampler = make_weighted_sampler(train_data["dataset_source"])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"training on {DEVICE}")
    model = CNNBiLSTM().to(DEVICE)
    loss_fn = nn.HuberLoss(delta=5.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )

    best_val_mae = float("inf")
    epochs_since_best = 0

    for epoch in range(MAX_EPOCHS):
        train_loss, train_mae = run_epoch(model, train_loader, optimizer, loss_fn, train=True)
        _, val_mae = run_epoch(model, val_loader, optimizer, loss_fn, train=False)
        scheduler.step(val_mae)

        print(f"epoch {epoch}: train_loss={train_loss:.3f} train_mae={train_mae:.2f} val_mae={val_mae:.2f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            epochs_since_best = 0
            CHECKPOINT_PATH.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), CHECKPOINT_PATH)
        else:
            epochs_since_best += 1
            if epochs_since_best >= EARLY_STOP_PATIENCE:
                print(f"early stopping at epoch {epoch}, best val_mae={best_val_mae:.2f}")
                break


if __name__ == "__main__":
    main()
