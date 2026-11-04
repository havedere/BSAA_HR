import argparse
import sys

import numpy as np
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
BOOTSTRAP_DIR = ROOT / "cache" / "bootstrap"
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


def bootstrap_letters(value):
    if set(value) - set("HL"):  # empty = argparse applying type to the default
        raise argparse.ArgumentTypeError("use H, L or HL (H = high BPM, L = low BPM)")
    return "".join(sorted(set(value)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", choices=["huber", "mae", "mse"], default="huber")
    parser.add_argument("--delta", type=float, default=5.0, help="Huber delta (BPM), ignored for mae/mse")
    parser.add_argument(
        "--bootstrap", "-bootstrap", type=bootstrap_letters, default="",
        help="add time-warped train windows from cache/bootstrap/: H (high BPM), L (low BPM) or HL",
    )
    args = parser.parse_args()
    # default run keeps writing best.pt, others get a tagged checkpoint
    tag = args.loss if args.loss in ("mae", "mse") else f"huber{args.delta:g}"
    if args.bootstrap:
        tag += f"_boot{args.bootstrap}"
    checkpoint_path = CHECKPOINT_PATH if tag == "huber5" else CHECKPOINT_PATH.with_name(f"best_{tag}.pt")

    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True  # cuDNN kernels vary between runs otherwise
    subjects_by_dataset = subjects_by_dataset_from_manifest(MANIFEST_PATH)
    split = subject_wise_split(subjects_by_dataset, seed=42)

    train_data = load_windows_for_subjects(sorted(split["train"]), CACHE_DIR)
    val_data = load_windows_for_subjects(sorted(split["val"]), CACHE_DIR)

    is_synthetic = np.zeros(len(train_data["labels"]), bool)
    for letter in args.bootstrap:
        boot = load_windows_for_subjects(sorted(split["train"]), BOOTSTRAP_DIR / letter)
        print(f"bootstrap {letter}: {len(boot['labels'])} synthetic windows")
        for key in train_data:
            train_data[key] = np.concatenate([train_data[key], boot[key]])
        is_synthetic = np.concatenate([is_synthetic, np.ones(len(boot["labels"]), bool)])

    train_ds = WindowDataset(
        train_data["windows"], train_data["labels"], augment=True
    )
    val_ds = WindowDataset(
        val_data["windows"], val_data["labels"], augment=False
    )

    sampler = make_weighted_sampler(train_data["dataset_source"], is_synthetic)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    # info only: does the high tail improve? early stopping stays on overall val MAE
    high = val_data["labels"] >= 110
    val_high_loader = DataLoader(
        WindowDataset(val_data["windows"][high], val_data["labels"][high]),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    ) if high.any() else None

    print(f"training on {DEVICE}, loss={tag}, checkpoint={checkpoint_path.name}")
    model = CNNBiLSTM().to(DEVICE)
    loss_fn = {"mae": nn.L1Loss(), "mse": nn.MSELoss()}.get(args.loss) or nn.HuberLoss(delta=args.delta)
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
        high_msg = ""
        if val_high_loader:
            _, val_high_mae = run_epoch(model, val_high_loader, optimizer, loss_fn, train=False)
            high_msg = f" val_mae_110+={val_high_mae:.2f}"

        print(f"epoch {epoch}: train_loss={train_loss:.3f} train_mae={train_mae:.2f} val_mae={val_mae:.2f}{high_msg}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            epochs_since_best = 0
            checkpoint_path.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_since_best += 1
            if epochs_since_best >= EARLY_STOP_PATIENCE:
                print(f"early stopping at epoch {epoch}, best val_mae={best_val_mae:.2f}")
                break


if __name__ == "__main__":
    main()
