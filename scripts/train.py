import argparse
import json
import subprocess
import sys

import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bpm.data_loading import load_windows_for_subjects, subjects_by_dataset_from_manifest
from bpm.dataset import WindowDataset, lds_weights, make_weighted_sampler, sampler_weights
from bpm.model import CNNBiLSTM
from bpm.split import fold_split, load_folds, subject_wise_split

MANIFEST_PATH = ROOT / "cache" / "manifest.csv"
CACHE_DIR = ROOT / "cache" / "windows"
BOOTSTRAP_DIR = ROOT / "cache" / "bootstrap"
CHECKPOINT_PATH = ROOT / "cache" / "best.pt"
FOLDS_PATH = ROOT / "cache" / "folds.json"
RUNS_DIR = ROOT / "cache" / "runs"

BATCH_SIZE = 64
MAX_EPOCHS = 100
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def mae(pred, target):
    return (pred - target).abs().mean().item()


def run_epoch(model, loader, optimizer, loss_fn, train):
    """loss_fn has reduction="none"; batches with a weight column use mean(w * loss).
    Returns (loss, mae, per-window preds in loader order)."""
    model.train(train)
    total_loss, total_mae, n = 0.0, 0.0, 0
    all_preds = []
    for batch in loader:
        windows, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            preds = model(windows)
            per_window = loss_fn(preds, labels)
            loss = (per_window * batch[2].to(DEVICE)).mean() if len(batch) == 3 else per_window.mean()
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(labels)
        total_mae += mae(preds, labels) * len(labels)
        n += len(labels)
        all_preds.append(preds.detach().cpu().numpy())
    return total_loss / n, total_mae / n, np.concatenate(all_preds)


def balanced_mae(preds, labels, dataset_source):
    """Mean over sub-datasets of each one's window-pooled MAE."""
    errors = np.abs(preds - labels)
    return float(np.mean([errors[dataset_source == d].mean() for d in np.unique(dataset_source)]))


def run_tag(loss, delta, bootstrap, lds_power, fold, seed):
    tag = loss if loss in ("mae", "mse") else f"huber{delta:g}"
    if bootstrap:
        tag += f"_boot{bootstrap}"
    if lds_power:
        tag += f"_lds{lds_power:g}"
    if fold is not None:
        tag += f"_fold{fold}_seed{seed}"
    elif seed != 42:
        tag += f"_seed{seed}"
    return tag


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


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
    parser.add_argument("--seed", type=int, default=42, help="init/shuffle/augment seed; the split stays at seed 42")
    parser.add_argument("--patience", type=int, default=20, help="early-stopping patience (epochs)")
    parser.add_argument("--fold", type=int, choices=range(5), help="CV fold from cache/folds.json; default = fixed split")
    parser.add_argument("--lds-power", type=float, default=0.0, help="LDS loss-weight power p (0 = off)")
    args = parser.parse_args()
    # default run keeps writing best.pt, others get a tagged checkpoint
    tag = run_tag(args.loss, args.delta, args.bootstrap, args.lds_power, args.fold, args.seed)
    checkpoint_path = CHECKPOINT_PATH if tag == "huber5" else CHECKPOINT_PATH.with_name(f"best_{tag}.pt")

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True  # cuDNN kernels vary between runs otherwise
    subjects_by_dataset = subjects_by_dataset_from_manifest(MANIFEST_PATH)
    folds_hash, bootstrap_dir = None, BOOTSTRAP_DIR
    if args.fold is None:
        split = subject_wise_split(subjects_by_dataset, seed=42)
    else:
        groups, folds_hash = load_folds(FOLDS_PATH)
        split = fold_split(groups, args.fold)
        bootstrap_dir = BOOTSTRAP_DIR / f"fold{args.fold}"

    train_data = load_windows_for_subjects(sorted(split["train"]), CACHE_DIR)
    val_data = load_windows_for_subjects(sorted(split["val"]), CACHE_DIR)

    is_synthetic = np.zeros(len(train_data["labels"]), bool)
    for letter in args.bootstrap:
        boot = load_windows_for_subjects(sorted(split["train"]), bootstrap_dir / letter)
        print(f"bootstrap {letter}: {len(boot['labels'])} synthetic windows")
        for key in train_data:
            train_data[key] = np.concatenate([train_data[key], boot[key]])
        is_synthetic = np.concatenate([is_synthetic, np.ones(len(boot["labels"]), bool)])

    # LDS on real windows only, density counted with sampler weights; synthetic windows weight 1
    loss_weights = np.ones(len(train_data["labels"]))
    if args.lds_power:
        real = ~is_synthetic
        s = sampler_weights(train_data["dataset_source"], is_synthetic)[real]
        w = lds_weights(train_data["labels"][real], args.lds_power, s)
        loss_weights[real] = w
        print(f"LDS p={args.lds_power:g}: clipped {np.mean(w >= 20 * (1 - 1e-9)):.2%}, "
              f"max {w.max():.2f}, max/min {w.max() / w.min():.1f}")
    train_ds = WindowDataset(
        train_data["windows"], train_data["labels"], augment=True, seed=args.seed, weights=loss_weights
    )
    val_ds = WindowDataset(
        val_data["windows"], val_data["labels"], augment=False
    )

    sampler = make_weighted_sampler(train_data["dataset_source"], is_synthetic)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    # info only: does the high tail improve? early stopping stays on balanced val MAE
    high = val_data["labels"] >= 110
    val_high_loader = DataLoader(
        WindowDataset(val_data["windows"][high], val_data["labels"][high]),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    ) if high.any() else None

    print(f"training on {DEVICE}, loss={tag}, checkpoint={checkpoint_path.name}")
    model = CNNBiLSTM().to(DEVICE)
    loss_fn = (nn.HuberLoss(reduction="none", delta=args.delta) if args.loss == "huber"
               else {"mae": nn.L1Loss, "mse": nn.MSELoss}[args.loss](reduction="none"))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )

    best_val, best_epoch = float("inf"), -1
    hit_max_epochs = True

    for epoch in range(MAX_EPOCHS):
        train_loss, train_mae, _ = run_epoch(model, train_loader, optimizer, loss_fn, train=True)
        _, val_mae, val_preds = run_epoch(model, val_loader, optimizer, loss_fn, train=False)
        val_bal = balanced_mae(val_preds, val_data["labels"], val_data["dataset_source"])
        scheduler.step(val_bal)
        high_msg = ""
        if val_high_loader:
            _, val_high_mae, _ = run_epoch(model, val_high_loader, optimizer, loss_fn, train=False)
            high_msg = f" val_mae_110+={val_high_mae:.2f}"

        saved = val_bal < best_val
        print(f"epoch {epoch}: train_loss={train_loss:.3f} train_mae={train_mae:.2f} "
              f"val_mae={val_mae:.4f} val_bal={val_bal:.4f}{high_msg}{' saved' if saved else ''}")

        if saved:
            best_val, best_epoch = val_bal, epoch
            checkpoint_path.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
        elif epoch - best_epoch >= args.patience:
            print(f"early stopping at epoch {epoch}, best val_bal={best_val:.4f} (epoch {best_epoch})")
            hit_max_epochs = False
            break

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    sidecar = {
        **vars(args), "tag": tag, "checkpoint": checkpoint_path.name,
        "best_epoch": best_epoch, "best_val": best_val, "stop_epoch": epoch,
        "hit_max_epochs": hit_max_epochs, "collapsed": best_epoch <= 2,
        "git_hash": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
        "folds_hash": folds_hash,
    }
    (RUNS_DIR / f"{tag}.json").write_text(json.dumps(sidecar, indent=1))


if __name__ == "__main__":
    main()
