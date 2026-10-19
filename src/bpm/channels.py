import numpy as np

CHANNEL_ORDER = ["PPG", "ACC_X", "ACC_Y", "ACC_Z", "has_motion"]


def assemble_channels(ppg, acc=None):
    """Build the fixed 5-channel model input: PPG, ACC_X/Y/Z, has_motion.

    `acc` is a dict with keys ACC_X/ACC_Y/ACC_Z when the sub-dataset has
    accelerometer data; None for the two sub-datasets that don't (zero-fill
    + has_motion=0, see plan.md section 3).
    """
    n = len(ppg)
    has_motion = 1.0 if acc is not None else 0.0
    zeros = np.zeros(n, dtype=ppg.dtype)
    return {
        "PPG": ppg,
        "ACC_X": acc["ACC_X"] if acc is not None else zeros,
        "ACC_Y": acc["ACC_Y"] if acc is not None else zeros,
        "ACC_Z": acc["ACC_Z"] if acc is not None else zeros,
        "has_motion": np.full(n, has_motion, dtype=ppg.dtype),
    }
