"""Shared cells for the CV story notebooks: imports, paths, style. Imported by the gen_nb_*.py generators."""

SETUP = r'''
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bpm.aggregate import BIN_EDGES, decide, ltgo_folds, macro, parse_tag, subject_mae

PREDS = ROOT / "cache" / "preds"
RESULTS = ROOT / "cache" / "results"
OUT = ROOT / "cache" / "deck" / "cv_story"   # figures + values.json for the deck
OUT.mkdir(parents=True, exist_ok=True)
DATASETS = ["multisite-ppg", "wrist-finger-ppg"]
'''

STYLE_MD = r'''
## Style

Single source of truth for every plot below. Colors come from a colorblind-safe palette and are bound to **roles**: a config keeps its color in every figure of both notebooks. Plot cells never use a raw hex color or font size.
'''

STYLE = r'''
# palette of docs/hr_cnn_bilstm_results.odp (the deck these figures get merged into):
# ink #212a30, accent green #1e7f5c, red #b03a2e, secondary #555b60, muted #7a7f84, light #c9d1cd, card #edf2f0
COLORS = {
    # configs (role = model family): baselines slate/grey, MSE + boot red, dist + boot green
    "mse": "#212a30", "huber10": "#555b60", "mae": "#7a7f84",
    "mse_bootHL": "#b03a2e", "mse_bootHL_lds0.5": "#b03a2e", "mse_bootHL_lds1": "#b03a2e",
    "dist_bootHL": "#1e7f5c", "dist_bootHL_lds0.5": "#1e7f5c", "dist_bootHL_lds1": "#1e7f5c",
    # datasets
    "multisite-ppg": "#212a30", "wrist-finger-ppg": "#1e7f5c",
    # scores (line style separates the greys, see SCORE_LS)
    "score_std": "#1e7f5c", "score_mc": "#b03a2e", "score_spread": "#212a30",
    "prominence": "#7a7f84", "skew": "#7a7f84",
    # smoothers
    "raw": "#7a7f84", "median": "#555b60", "kalman": "#1e7f5c", "hmm": "#212a30",
    # signals
    "truth": "#212a30", "ideal": "#7a7f84", "bad": "#b03a2e", "good": "#1e7f5c", "neutral": "#c9d1cd",
    # splits (train vs test): test carries the accent
    "train": "#212a30", "val": "#7a7f84", "test": "#1e7f5c",
    # ink and surface
    "text": "#212a30", "muted": "#7a7f84", "grid": "#e8ecea", "surface": "#ffffff",
}
SCORE_LS = {"score_std": "-", "score_mc": "-", "score_spread": "-", "prominence": "-", "skew": "--"}
# LDS power shown by marker (scatter) / line style (curves), so 3 hues cover all 9 configs
MARKERS = {"": "o", "lds0.5": "s", "lds1": "^"}
LINESTYLES = {"": "-", "lds0.5": "--", "lds1": ":"}
LABELS = {
    "mse": "MSE", "huber10": "Huber δ10", "mae": "MAE",
    "mse_bootHL": "MSE + boot", "mse_bootHL_lds0.5": "MSE + boot + LDS 0.5", "mse_bootHL_lds1": "MSE + boot + LDS 1",
    "dist_bootHL": "dist + boot", "dist_bootHL_lds0.5": "dist + boot + LDS 0.5", "dist_bootHL_lds1": "dist + boot + LDS 1",
}
STYLE = {"fig_w": 10, "row_h": 4.2, "dpi": 110, "save_dpi": 200, "font": 11, "title": 13,
         "lw": 1.8, "lw_thin": 1.0, "marker": 70, "alpha": 0.25}

plt.rcParams.update({
    "figure.dpi": STYLE["dpi"], "figure.facecolor": COLORS["surface"], "axes.facecolor": COLORS["surface"],
    "figure.constrained_layout.use": True,
    "font.size": STYLE["font"], "axes.titlesize": STYLE["title"], "axes.labelsize": STYLE["font"],
    "legend.fontsize": STYLE["font"] - 1, "legend.frameon": False,
    "text.color": COLORS["text"], "axes.labelcolor": COLORS["text"], "axes.edgecolor": COLORS["muted"],
    "xtick.color": COLORS["muted"], "ytick.color": COLORS["muted"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": COLORS["grid"], "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": STYLE["lw"],
    "font.family": "Lato",  # the deck's body font as rendered here (Calibri falls back to it)
})
from matplotlib.colors import LinearSegmentedColormap
DENSITY_CMAP = LinearSegmentedColormap.from_list("deck", ["#edf2f0", "#1e7f5c", "#212a30"])


def figsize(rows=1, w=None):
    return (w or STYLE["fig_w"], STYLE["row_h"] * rows)


def lds_of(cfg):
    return next((p for p in ("lds0.5", "lds1") if cfg.endswith(p)), "")


def save(fig, name):
    """Figure for the deck: cache/deck/cv_story/<name>.png."""
    fig.savefig(OUT / f"{name}.png", dpi=STYLE["save_dpi"], facecolor=COLORS["surface"])


VALUES_PATH = OUT / "values.json"
VALUES = json.loads(VALUES_PATH.read_text()) if VALUES_PATH.exists() else {}


def put(section, **kv):
    """Record numbers the deck cites (values.json, merged across both notebooks)."""
    conv = lambda v: v.item() if hasattr(v, "item") else v
    VALUES.setdefault(section, {}).update({k: conv(v) for k, v in kv.items()})
    VALUES_PATH.write_text(json.dumps(VALUES, indent=1, default=float))


BIN_LABELS = [f"{lo}-{hi}" if np.isfinite(hi) else f"{lo}+" for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:])]


def bin_of(truth):
    return pd.cut(truth, BIN_EDGES, right=False, labels=BIN_LABELS)
'''
