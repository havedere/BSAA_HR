"""Checkpoint -> ONNX (for Netron / Zetane) + one real z-scored PPG window as .npy input.

.venv/bin/python viz/export_onnx.py --ckpt cache/best_huber10_fold0_seed42.pt [--subject multisite_P1]
Writes cache/viz/<ckpt stem>.onnx and cache/viz/<subject>_window.npy (shape (1, 1, 1000)).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bpm.data_loading import load_subject_npz  # noqa: E402
from bpm.dataset import zscore_window  # noqa: E402
from evaluate import load_model  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--subject", default="multisite_P1")
    args = p.parse_args()

    out_dir = ROOT / "cache" / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_subject_npz(args.subject, ROOT / "cache" / "windows")
    assert data is not None, f"no cache/windows/{args.subject}.npz"
    x = zscore_window(data["windows"][:1]).astype(np.float32)  # (1, 1, 1000)
    np.save(out_dir / f"{args.subject}_window.npy", x)

    model = load_model(args.ckpt).cpu()
    onnx_path = out_dir / f"{args.ckpt.stem}.onnx"
    torch.onnx.export(
        model, (torch.from_numpy(x),), onnx_path,
        input_names=["ppg"], output_names=["out"],
        dynamo=True, external_data=False,
    )
    with torch.no_grad():
        pred = model(torch.from_numpy(x))
    print(f"wrote {onnx_path}\nwrote {out_dir / f'{args.subject}_window.npy'}  label={data['labels'][0]:.1f}  pred={pred.flatten()[:3].tolist()}")


if __name__ == "__main__":
    main()
