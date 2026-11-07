import torch
import torch.nn as nn

BIN_MIN = 30
N_BINS = 191  # 1-BPM bin centres 30..220
_HEAD_OUT = {"reg": 1, "dist": N_BINS, "gauss": 2}


def _centres(like):
    return torch.arange(BIN_MIN, BIN_MIN + N_BINS, dtype=like.dtype, device=like.device)


def gaussian_targets(labels, sigma=2.0):
    """(B,) BPM -> (B, N_BINS) Gaussian over bin centres, rows sum to 1."""
    labels = labels.float()
    d = _centres(labels)[None, :] - labels[:, None]
    return torch.softmax(-0.5 * (d / sigma) ** 2, dim=1)


def argmax_bpm(logits):
    return logits.argmax(dim=1).to(logits.dtype) + BIN_MIN


def decode_bpm(logits, window=10, log_prior=None, tau=0):
    """Expected BPM within +-window bins of the argmax; logits - tau*log_prior first."""
    if log_prior is not None and tau:
        logits = logits - tau * log_prior
    p = torch.softmax(logits, dim=1)
    c = _centres(p)[None, :]
    peak = c[0, p.argmax(dim=1)][:, None]
    q = p * ((c - peak).abs() <= window)
    return (q * c).sum(1) / q.sum(1)


def dist_std(logits):
    """Std of the softmax curve over the bin centres (dist uncertainty score)."""
    p = torch.softmax(logits, dim=1)
    c = _centres(p)[None, :]
    mean = (p * c).sum(1, keepdim=True)
    return ((p * (c - mean) ** 2).sum(1)).sqrt()


class CNNBiLSTM(nn.Module):
    def __init__(self, in_channels=1, lstm_hidden=64, dropout=0.3, head="reg"):
        super().__init__()
        self.head_type = head
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding="same"),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding="same"),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding="same"),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(128, lstm_hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, _HEAD_OUT[head]),
        )

    def forward(self, x):
        features = self.cnn(x)  # (B, 128, T)
        features = features.transpose(1, 2)  # (B, T, 128)
        lstm_out, _ = self.lstm(features)  # (B, T, hidden*2)
        pooled = lstm_out.mean(dim=1)  # (B, hidden*2)
        out = self.head(pooled)
        return out.squeeze(-1) if self.head_type == "reg" else out
