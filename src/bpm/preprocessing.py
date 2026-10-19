from scipy.signal import butter, sosfiltfilt


def bandpass_filter(signal, fs, low=0.5, high=8.0, order=4):
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, signal)
