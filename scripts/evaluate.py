import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bpm.data_loading import load_subject_npz, subjects_by_dataset_from_manifest
from bpm.dataset import zscore_window
from bpm.model import BIN_MIN, N_BINS, CNNBiLSTM, decode_bpm, dist_std
from bpm.split import fold_split, load_folds, subject_wise_split

MANIFEST_PATH = ROOT / "cache" / "manifest.csv"
CACHE_DIR = ROOT / "cache" / "windows"
PLOTS_DIR = ROOT / "cache" / "plots"
FOLDS_PATH = ROOT / "cache" / "folds.json"
PREDS_DIR = ROOT / "cache" / "preds"
HR_CURVE_DIR = ROOT / "cache" / "hr_curves"  # written by build_bootstrap.py
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HEAD_BY_OUT = {1: "reg", N_BINS: "dist", 2: "gauss"}  # last-layer width -> head


def to_bpm(head, out, log_prior=None, tau=0):
    if head == "dist":
        return decode_bpm(out, log_prior=log_prior, tau=tau)
    return out[:, 0] if head == "gauss" else out


def log_prior_from(subject_ids):
    """log of (Gaussian-smoothed (sigma 2) 1-BPM label counts + 1), normalised over the N_BINS bins."""
    labels = np.concatenate([np.load(CACHE_DIR / f"{s}.npz")["labels"] for s in subject_ids
                             if (CACHE_DIR / f"{s}.npz").exists()])
    idx = np.clip(np.round(labels).astype(int) - BIN_MIN, 0, N_BINS - 1)
    p = gaussian_filter1d(np.bincount(idx, minlength=N_BINS).astype(float), 2.0) + 1
    return torch.from_numpy(np.log(p / p.sum())).float().to(DEVICE)


def predict(models, subject_data, log_prior=None, tau=0, mc=0, with_logp=False):
    """Ensemble prediction for one subject (all models share one head) -> dict of numpy (n,) arrays:
    pred, scores (primary: reg spread, dist curve std, gauss sigma), score_spread, score_std, score_mc
    (NaN where not applicable).
    dist: per-model logit adjustment -> softmax -> mean probs -> decode. gauss: mixture mean / variance.
    mc > 0: std of BPM over mc dropout passes pooled across all models (BatchNorm eval; includes the
    between-model spread for ensembles). with_logp (dist only): also "logp" (n, N_BINS), log of the mean probs."""
    head = models[0].head_type
    assert all(m.head_type == head for m in models), "ensemble mixes heads"
    x = torch.from_numpy(zscore_window(subject_data["windows"])).float().to(DEVICE)
    nan = torch.full((len(x),), float("nan"), device=DEVICE)
    with torch.no_grad():
        outs = torch.stack([m(x) for m in models])
        bpm = torch.stack([to_bpm(head, o, log_prior, tau) for o in outs])
        spread = bpm.std(0, correction=0) if len(models) > 1 else nan
        std = nan
        if head == "dist":
            logits = outs - tau * log_prior if tau else outs
            lp = torch.softmax(logits, dim=2).mean(0).log()
            pred, std = decode_bpm(lp), dist_std(lp)
        elif head == "gauss":
            mean, var = outs[..., 0], outs[..., 1].clamp(-10, 10).exp()
            pred = mean.mean(0)
            std = ((var + mean**2).mean(0) - pred**2).clamp(min=0).sqrt()
        else:
            pred = bpm.mean(0)
        score_mc = nan
        if mc:
            passes = []
            for m in models:
                for d in m.modules():
                    if isinstance(d, nn.Dropout):
                        d.train()
                passes += [to_bpm(head, m(x), log_prior, tau) for _ in range(mc)]
                m.eval()
            score_mc = torch.stack(passes).std(0, correction=0)
    out = {"pred": pred, "scores": spread if head == "reg" else std,
           "score_spread": spread, "score_std": std, "score_mc": score_mc}
    if with_logp and head == "dist":
        out["logp"] = lp
    return {k: v.float().cpu().numpy() for k, v in out.items()}


def bland_altman_plot(truth, pred, title, out_path):
    mean = (truth + pred) / 2
    diff = pred - truth
    bias, sd = diff.mean(), diff.std()
    fig, ax = plt.subplots()
    ax.scatter(mean, diff, alpha=0.3, s=5)
    ax.axhline(bias, color="red", linestyle="-")
    ax.axhline(bias + 1.96 * sd, color="red", linestyle="--")
    ax.axhline(bias - 1.96 * sd, color="red", linestyle="--")
    ax.set_xlabel("mean of ECG-derived and predicted BPM")
    ax.set_ylabel("predicted - ECG-derived BPM")
    ax.set_title(title)
    fig.savefig(out_path)
    plt.close(fig)


def load_model(ckpt):
    state = torch.load(ckpt, map_location=DEVICE)
    model = CNNBiLSTM(head=HEAD_BY_OUT[state["head.3.weight"].shape[0]]).to(DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def ckpt_config(ckpts, fold):
    """(config id, seed) of fold-k checkpoints named best_<config>_fold<k>_seed<s>.pt; seed None if several."""
    tags = [c.stem.removeprefix("best_") for c in ckpts]
    parsed = [re.fullmatch(r"(.+)_fold(\d+)_seed(\d+)", t) for t in tags]
    assert all(parsed), f"ckpt names must end in _fold<k>_seed<s>: {tags}"
    assert {int(m[2]) for m in parsed} == {fold}, f"ckpts not all fold {fold}: {tags}"
    assert len({m[1] for m in parsed}) == 1, f"ckpts differ in config id: {tags}"
    return parsed[0][1], int(parsed[0][3]) if len(ckpts) == 1 else None


def evaluate_fold(fold, ckpts, tau=0, mc=0):
    """Predict fold val+test subjects, write cache/preds/<tag>.npz (1 ckpt) or
    cache/preds/ens_<config id>_fold<k>.npz (several, combined per head, see predict).
    With 1 reg ckpt scores are NaN (no ensemble spread). tau=1 puts _tau1 into the config id."""
    config, seed = ckpt_config(ckpts, fold)
    config += "_tau1" if tau else ""
    out_name = f"{config}_fold{fold}_seed{seed}.npz" if seed is not None else f"ens_{config}_fold{fold}.npz"

    groups, _ = load_folds(FOLDS_PATH)
    split = fold_split(groups, fold)
    models = [load_model(c) for c in ckpts]
    if tau and models[0].head_type != "dist":
        sys.exit("--tau needs dist checkpoints")
    log_prior = log_prior_from(sorted(split["train"])) if tau else None
    cols = defaultdict(list)
    for split_name in ("val", "test"):
        for subject_id in sorted(split[split_name]):
            data = load_subject_npz(subject_id, CACHE_DIR)
            if data is None or len(data["labels"]) == 0:
                continue
            p = predict(models, data, log_prior, tau, mc)
            n = len(data["labels"])
            cols["subject"].append(np.full(n, subject_id))
            cols["dataset_source"].append(np.asarray(data["dataset_source"]).astype(str))
            cols["split"].append(np.full(n, split_name))
            cols["truth"].append(data["labels"])
            for k, v in p.items():
                cols[k].append(v)
    out = {k: np.concatenate(v) for k, v in cols.items()}
    PREDS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(PREDS_DIR / out_name, **out)

    print(f"fold {fold}, {len(ckpts)} ckpt(s), wrote {PREDS_DIR / out_name}")
    print("MAE per split / sub-dataset (macro-averaged across subjects):")
    err = np.abs(out["pred"] - out["truth"])
    for split_name in ("val", "test"):
        for sub_dataset in sorted(np.unique(out["dataset_source"])):
            rows = (out["split"] == split_name) & (out["dataset_source"] == sub_dataset)
            subj_maes = [err[rows & (out["subject"] == s)].mean() for s in np.unique(out["subject"][rows])]
            print(f"  {split_name} {sub_dataset}: MAE={np.mean(subj_maes):.2f} (n_subj={len(subj_maes)})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, nargs="+", required=True,
                        help="one or more checkpoints; several (needs --fold) = seed ensemble")
    parser.add_argument("--fold", type=int, choices=range(5), help="CV fold from cache/folds.json; writes cache/preds/")
    parser.add_argument("--tau", type=int, choices=[0, 1], default=0,
                        help="dist only: subtract tau * log train-label prior before decoding")
    parser.add_argument("--mc", type=int, default=0, help="MC dropout passes per model (score_mc), 0 = off")
    args = parser.parse_args()
    if args.fold is not None:
        evaluate_fold(args.fold, args.ckpt, args.tau, args.mc)
        return
    if len(args.ckpt) > 1:
        parser.error("several --ckpt need --fold")
    if args.mc:
        parser.error("--mc needs --fold")
    args.ckpt = args.ckpt[0]
    plots_dir = PLOTS_DIR / args.ckpt.stem

    subjects_by_dataset = subjects_by_dataset_from_manifest(MANIFEST_PATH)
    split = subject_wise_split(subjects_by_dataset, seed=42)
    test_subjects = sorted(split["test"])

    model = load_model(args.ckpt)
    if args.tau and model.head_type != "dist":
        parser.error("--tau needs a dist checkpoint")
    log_prior = log_prior_from(sorted(split["train"])) if args.tau else None

    plots_dir.mkdir(parents=True, exist_ok=True)
    per_dataset_subject_mae = defaultdict(list)
    per_dataset_subject_rmse = defaultdict(list)
    per_dataset_truth_pred = defaultdict(lambda: ([], []))

    for subject_id in test_subjects:
        data = load_subject_npz(subject_id, CACHE_DIR)
        if data is None or len(data["labels"]) == 0:
            continue
        pred, truth = predict([model], data, log_prior, args.tau)["pred"], data["labels"]
        sub_dataset = str(data["dataset_source"][0])

        mae = float(np.abs(pred - truth).mean())
        rmse = float(np.sqrt(((pred - truth) ** 2).mean()))
        per_dataset_subject_mae[sub_dataset].append(mae)
        per_dataset_subject_rmse[sub_dataset].append(rmse)
        per_dataset_truth_pred[sub_dataset][0].append(truth)
        per_dataset_truth_pred[sub_dataset][1].append(pred)

    print("Per-sub-dataset MAE / RMSE (macro-averaged across subjects):")
    combined_maes = []
    for sub_dataset in sorted(per_dataset_subject_mae):
        macro_mae = float(np.mean(per_dataset_subject_mae[sub_dataset]))
        macro_rmse = float(np.mean(per_dataset_subject_rmse[sub_dataset]))
        combined_maes.append(macro_mae)
        print(f"  {sub_dataset}: MAE={macro_mae:.2f} RMSE={macro_rmse:.2f}")

        truth_list, pred_list = per_dataset_truth_pred[sub_dataset]
        bland_altman_plot(
            np.concatenate(truth_list),
            np.concatenate(pred_list),
            f"Bland-Altman ({sub_dataset})",
            plots_dir / f"bland_altman_{sub_dataset}.png",
        )

    print(f"Combined (equal-weighted across sub-datasets): MAE={np.mean(combined_maes):.2f}")

    print("MAE per 10-BPM truth bin (window-level, pooled per sub-dataset):")
    for sub_dataset in sorted(per_dataset_truth_pred):
        truth = np.concatenate(per_dataset_truth_pred[sub_dataset][0])
        pred = np.concatenate(per_dataset_truth_pred[sub_dataset][1])
        bins = (truth // 10 * 10).astype(int)
        cells = [
            f"{lo}:{np.abs(pred - truth)[bins == lo].mean():.1f}(n={(bins == lo).sum()})"
            for lo in np.unique(bins)
        ]
        print(f"  {sub_dataset}: " + " ".join(cells))

    all_truth = np.concatenate([t for tl, _ in per_dataset_truth_pred.values() for t in tl])
    all_pred = np.concatenate([p for _, pl in per_dataset_truth_pred.values() for p in pl])
    bland_altman_plot(all_truth, all_pred, "Bland-Altman (overall)", plots_dir / "bland_altman_overall.png")


if __name__ == "__main__":
    main()
