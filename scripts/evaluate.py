import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bpm.data_loading import load_subject_npz, subjects_by_dataset_from_manifest
from bpm.dataset import WindowDataset
from bpm.model import CNNBiLSTM
from bpm.split import subject_wise_split

MANIFEST_PATH = ROOT / "cache" / "manifest.csv"
CACHE_DIR = ROOT / "cache" / "windows"
CHECKPOINT_PATH = ROOT / "cache" / "best.pt"
PLOTS_DIR = ROOT / "cache" / "plots"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def predict_subject(model, subject_data):
    ds = WindowDataset(
        subject_data["windows"], subject_data["labels"], augment=False
    )
    with torch.no_grad():
        windows = torch.from_numpy(ds.windows).float().to(DEVICE)
        preds = model(windows).cpu().numpy()
    return preds, subject_data["labels"]


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


def main():
    subjects_by_dataset = subjects_by_dataset_from_manifest(MANIFEST_PATH)
    split = subject_wise_split(subjects_by_dataset, seed=42)
    test_subjects = sorted(split["test"])

    model = CNNBiLSTM().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    per_dataset_subject_mae = defaultdict(list)
    per_dataset_subject_rmse = defaultdict(list)
    per_dataset_truth_pred = defaultdict(lambda: ([], []))

    for subject_id in test_subjects:
        data = load_subject_npz(subject_id, CACHE_DIR)
        if data is None or len(data["labels"]) == 0:
            continue
        pred, truth = predict_subject(model, data)
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
            PLOTS_DIR / f"bland_altman_{sub_dataset}.png",
        )

    print(f"Combined (equal-weighted across sub-datasets): MAE={np.mean(combined_maes):.2f}")

    all_truth = np.concatenate([t for tl, _ in per_dataset_truth_pred.values() for t in tl])
    all_pred = np.concatenate([p for _, pl in per_dataset_truth_pred.values() for p in pl])
    bland_altman_plot(all_truth, all_pred, "Bland-Altman (overall)", PLOTS_DIR / "bland_altman_overall.png")


if __name__ == "__main__":
    main()
