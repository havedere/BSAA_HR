import numpy as np
import pytest

from bpm.edf_io import read_channels, read_header


def _write_edf(path, signals, n_per_record, phys, dig):
    """Minimal plain-EDF writer: `signals` maps label -> int16 digital samples."""
    labels = list(signals)
    n_records = len(next(iter(signals.values()))) // n_per_record
    ns = len(labels)

    def field(values, width):
        return "".join(str(v).ljust(width)[:width] for v in values)

    header = (
        "0".ljust(8) + " " * 80 + " " * 80 + "01.01.01" + "00.00.00"
        + str(256 * (ns + 1)).ljust(8) + " " * 44
        + str(n_records).ljust(8) + "1".ljust(8) + str(ns).ljust(4)
        + field(labels, 16) + field([""] * ns, 80) + field(["uV"] * ns, 8)
        + field([phys[0]] * ns, 8) + field([phys[1]] * ns, 8)
        + field([dig[0]] * ns, 8) + field([dig[1]] * ns, 8)
        + field([""] * ns, 80) + field([n_per_record] * ns, 8) + field([""] * ns, 32)
    )
    records = np.stack(
        [signals[l].reshape(n_records, n_per_record) for l in labels], axis=1
    )
    path.write_bytes(header.encode("ascii") + records.astype("<i2").tobytes())


def test_reads_physical_values_across_records(tmp_path):
    ecg = np.arange(-4, 4, dtype=np.int16)  # 8 samples = 2 records of 4
    ppg = np.arange(8, dtype=np.int16) * 100
    path = tmp_path / "x.edf"
    _write_edf(path, {"ECG": ecg, "PPG": ppg}, n_per_record=4, phys=(-10, 10), dig=(-1000, 1000))

    signals, fs = read_channels(path, ["PPG", "ECG"])

    assert fs == 4.0
    assert read_header(path) == {"n_samples": 8, "fs": 4.0}
    np.testing.assert_allclose(signals["ECG"], ecg / 100)
    np.testing.assert_allclose(signals["PPG"], ppg / 100)


def test_raises_on_missing_channel(tmp_path):
    path = tmp_path / "x.edf"
    _write_edf(path, {"ECG": np.zeros(4, np.int16)}, n_per_record=4, phys=(-1, 1), dig=(-1, 1))

    with pytest.raises(ValueError, match="PPG"):
        read_channels(path, ["PPG"])
