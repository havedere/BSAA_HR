import numpy as np

# Plain EDF only (not EDF+/BDF): 256-byte main header, then per-signal header
# fields stored field-by-field across all signals, then int16 data records.
_SIGNAL_FIELDS = [
    ("label", 16), ("transducer", 80), ("phys_dim", 8), ("phys_min", 8),
    ("phys_max", 8), ("dig_min", 8), ("dig_max", 8), ("prefilter", 80),
    ("n_per_record", 8), ("reserved", 32),
]


def _parse_header(f):
    main = f.read(256).decode("ascii")
    n_records = int(main[236:244])
    record_s = float(main[244:252])
    ns = int(main[252:256])
    fields = {}
    for name, width in _SIGNAL_FIELDS:
        raw = f.read(width * ns).decode("ascii")
        fields[name] = [raw[i * width : (i + 1) * width].strip() for i in range(ns)]
    return n_records, record_s, fields


def read_header(file_path):
    with open(file_path, "rb") as f:
        n_records, record_s, fields = _parse_header(f)
    n_per_record = int(fields["n_per_record"][0])
    return {"n_samples": n_records * n_per_record, "fs": n_per_record / record_s}


def read_channels(file_path, labels):
    with open(file_path, "rb") as f:
        n_records, record_s, fields = _parse_header(f)
        raw = np.fromfile(f, dtype="<i2")

    available = fields["label"]
    n_per_record = [int(n) for n in fields["n_per_record"]]
    records = raw[: n_records * sum(n_per_record)].reshape(n_records, sum(n_per_record))
    offsets = np.cumsum([0] + n_per_record)

    signals = {}
    for label in labels:
        if label not in available:
            raise ValueError(f"channel {label!r} missing in {file_path}")
        i = available.index(label)
        digital = records[:, offsets[i] : offsets[i + 1]].ravel().astype(np.float64)
        pmin, pmax = float(fields["phys_min"][i]), float(fields["phys_max"][i])
        dmin, dmax = float(fields["dig_min"][i]), float(fields["dig_max"][i])
        signals[label] = (digital - dmin) * (pmax - pmin) / (dmax - dmin) + pmin
    return signals, n_per_record[0] / record_s
