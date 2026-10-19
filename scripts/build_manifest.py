import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bpm.edf_io import read_header
from bpm.manifest import build_manifest

DATA_ROOT = Path(__file__).resolve().parent.parent / "harmonized_ecg_ppg_databases"
OUT_PATH = Path(__file__).resolve().parent.parent / "cache" / "manifest.csv"


def main():
    records = build_manifest(DATA_ROOT, header_reader=read_header)
    OUT_PATH.parent.mkdir(exist_ok=True)

    fieldnames = sorted({k for r in records for k in r})
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    print(f"wrote {len(records)} records to {OUT_PATH}")


if __name__ == "__main__":
    main()
