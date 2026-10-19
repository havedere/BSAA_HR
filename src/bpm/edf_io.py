import pyedflib


def read_header(file_path):
    with pyedflib.EdfReader(str(file_path)) as f:
        return {"n_samples": f.getNSamples()[0], "fs": f.getSampleFrequency(0)}


def read_channels(file_path, labels):
    with pyedflib.EdfReader(str(file_path)) as f:
        available = f.getSignalLabels()
        fs = f.getSampleFrequency(0)
        signals = {}
        for label in labels:
            if label not in available:
                raise ValueError(f"channel {label!r} missing in {file_path}")
            signals[label] = f.readSignal(available.index(label))
        return signals, fs
