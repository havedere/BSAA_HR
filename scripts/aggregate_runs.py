"""Aggregate CV runs: cache/runs/*.json + cache/preds/*.npz -> cache/results/{runs,summary}.csv."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bpm.aggregate import K, decide, ltgo_folds, parse_tag, run_metrics, summarize
from bpm.split import load_folds

RUNS_DIR = ROOT / "cache" / "runs"
PREDS_DIR = ROOT / "cache" / "preds"
RESULTS_DIR = ROOT / "cache" / "results"
FOLDS_PATH = ROOT / "cache" / "folds.json"


def load_preds(path, config, fold, seed):
    z = np.load(path)
    return pd.DataFrame({
        "config": config, "fold": fold, "seed": -1 if seed is None else seed,
        "split": z["split"], "dataset": z["dataset_source"], "subject": z["subject"],
        "truth": z["truth"], "err": np.abs(z["pred"] - z["truth"]),
    })


def loss_verdict(d):
    """Loss of the argmin config, or "indistinguishable" if every candidate loss is in the tie set."""
    loss = lambda configs: {c.split("_")[0] for c in configs}
    if len(loss(d.index)) > 1 and loss(d.index[d["tie"]]) == loss(d.index):
        return "indistinguishable"
    return loss(d.index[d["argmin"]]).pop()


def main():
    runs = pd.DataFrame([json.loads(p.read_text()) for p in sorted(RUNS_DIR.glob("*.json"))])
    if runs.empty:
        sys.exit(f"no sidecars in {RUNS_DIR}")
    runs = runs[runs["tag"].map(parse_tag).notna()].copy()  # CV runs only
    runs["config"] = runs["tag"].map(lambda t: parse_tag(t)[0])

    _, folds_hash = load_folds(FOLDS_PATH)
    stale = runs.loc[runs["folds_hash"] != folds_hash, "tag"].tolist()
    if stale:
        print(f"WARNING: {len(stale)} run(s) trained on another folds.json: {stale}")

    tables, metrics = {}, []
    for p in sorted(PREDS_DIR.glob("*.npz")):
        parsed = parse_tag(p.stem)
        if parsed is None:
            continue
        tables[p.stem] = t = load_preds(p, *parsed)
        metrics.append({"tag": p.stem, "config": parsed[0], "fold": parsed[1], "seed": parsed[2],
                        "ensemble": parsed[2] is None, **run_metrics(t)})
    if not tables:
        sys.exit(f"no CV preds in {PREDS_DIR}")
    is_ens = lambda tag: parse_tag(tag)[2] is None
    w = pd.concat([t for tag, t in tables.items() if not is_ens(tag)], ignore_index=True)
    ens = pd.concat([t for tag, t in tables.items() if is_ens(tag)] or [w.iloc[:0]], ignore_index=True)

    no_preds = sorted(set(runs["tag"]) - set(tables))
    if no_preds:
        print(f"WARNING: {len(no_preds)} sidecar(s) without preds: {no_preds}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    runs_csv = runs.merge(pd.DataFrame(metrics), on=["tag", "config", "fold", "seed"], how="outer")
    runs_csv.convert_dtypes().to_csv(RESULTS_DIR / "runs.csv", index=False)  # keep ints int next to ens NaN rows
    summary = summarize(w, ens, runs)
    summary.to_csv(RESULTS_DIR / "summary.csv", index_label="config")
    print(f"wrote {RESULTS_DIR / 'runs.csv'} ({len(runs_csv)} rows), "
          f"{RESULTS_DIR / 'summary.csv'} ({len(summary)} configs)")

    for name, folds in [*((f"k{k}", ltgo_folds(k)) for k in range(K)), ("global (optimistic)", range(K))]:
        d = decide(w, folds)
        if d.empty:
            print(f"{name}: no val rows")
            continue
        top3 = d[d["top3"]].sort_values("key").index.tolist()
        print(f"{name}: top3={top3} win={d.index[d['win']][0]} tie={d.index[d['tie']].tolist()} "
              f"loss={loss_verdict(d)}")


if __name__ == "__main__":
    main()
