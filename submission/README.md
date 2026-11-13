# Heart rate estimation from PPG

CNN-BiLSTM ensemble that estimates heart rate (BPM) from a raw PPG signal.

## Setup

```
pip install -r requirements.txt
```

Python 3.14 tested. Runs on CUDA if available, otherwise CPU.

## Usage

```python
from estimate_heartrate import estimate_heartrate

heartrate = estimate_heartrate(ppg, fs)
```

- `ppg`: raw PPG signal, 1-D array-like (NaN gaps are interpolated)
- `fs`: sampling rate in Hz (any rate, resampled internally to 100 Hz)
- returns: heart rate in BPM, sampled at 2 Hz, `int(len(ppg) / fs * 2)` values

## Method

1. Resample to 100 Hz, bandpass 0.5-8 Hz (4th-order Butterworth, zero-phase).
2. 10 s windows with 2 s stride, z-scored per window.
3. 15 models (5 cross-validation folds x 3 seeds, `weights/`) each predict a
   distribution over 30-220 BPM; the softmax curves are averaged.
4. Windows with curve std > 16.205 BPM are treated as unreliable and get no weight.
5. Kalman filter + RTS smoother over the window estimates, interpolated to 2 Hz.
