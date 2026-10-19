import torch.nn as nn


class CNNBiLSTM(nn.Module):
    def __init__(self, in_channels=5, lstm_hidden=64, dropout=0.3):
        super().__init__()
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
            nn.Linear(32, 1),
        )

    def forward(self, x):
        features = self.cnn(x)  # (B, 128, T)
        features = features.transpose(1, 2)  # (B, T, 128)
        lstm_out, _ = self.lstm(features)  # (B, T, hidden*2)
        pooled = lstm_out.mean(dim=1)  # (B, hidden*2)
        return self.head(pooled).squeeze(-1)
