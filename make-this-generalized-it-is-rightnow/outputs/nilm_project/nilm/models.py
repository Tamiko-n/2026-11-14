"""Step 7 - ANN architectures.

Every network maps a window of whole-house power (B, W) to, for each of the A
appliances, an ON/OFF logit and a scaled power estimate:  (B, A), (B, A).
They share the same two-head design so the comparison isolates the trunk.
"""
import torch
import torch.nn as nn


def two_channel(x):
    """(B, W) whole-house power -> (B, 2, W): level + first difference (x10).
    Derived only from the aggregate, so the input is still *only* total power,
    but small appliances (fridge ~100 W) become visible as switch edges."""
    d = torch.diff(x, dim=1, prepend=x[:, :1]) * 10.0
    return torch.stack([x, d], 1)


class Heads(nn.Module):
    def __init__(self, d, n_app):
        super().__init__()
        self.cls = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, n_app))
        self.reg = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, n_app))

    def forward(self, h):
        return self.cls(h), self.reg(h)


class MLP(nn.Module):
    """Model A (baseline): fully-connected net on the flattened window."""

    def __init__(self, window, n_app, hidden=(256, 128), dropout=0.2):
        super().__init__()
        layers, d = [], 2 * window
        for hdim in hidden:
            layers += [nn.Linear(d, hdim), nn.BatchNorm1d(hdim), nn.ReLU(), nn.Dropout(dropout)]
            d = hdim
        self.body = nn.Sequential(*layers)
        self.heads = Heads(d, n_app)

    def forward(self, x):
        return self.heads(self.body(two_channel(x).flatten(1)))


class CNN1D(nn.Module):
    """Model B1: 1-D convolutions learn local edge/step shapes (appliance switch events)."""

    def __init__(self, window, n_app, ch=(32, 64, 64), k=5, dropout=0.2):
        super().__init__()
        layers, c_in = [], 2
        for c in ch:
            layers += [nn.Conv1d(c_in, c, k, padding=k // 2), nn.BatchNorm1d(c), nn.ReLU()]
            c_in = c
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(c_in * 9, 128), nn.ReLU())
        self.heads = Heads(128, n_app)

    def forward(self, x):
        h = self.conv(two_channel(x))
        # add the last time-step features so the "current minute" is never averaged away
        z = torch.cat([self.pool(h).flatten(1), h[:, :, -1]], 1)
        return self.heads(self.fc(z))


class _RNN(nn.Module):
    cell = nn.LSTM

    def __init__(self, window, n_app, hidden=64, layers=2, dropout=0.2, bidirectional=False):
        super().__init__()
        self.inp = nn.Conv1d(2, 16, 3, padding=1)    # tiny feature lift: value + local slope
        self.rnn = self.cell(16, hidden, layers, batch_first=True, dropout=dropout,
                             bidirectional=bidirectional)
        d = hidden * (2 if bidirectional else 1)
        self.heads = Heads(d, n_app)

    def forward(self, x):
        h = torch.relu(self.inp(two_channel(x))).transpose(1, 2)
        out, _ = self.rnn(h)
        return self.heads(out[:, -1])


class LSTMNet(_RNN):
    """Model B2: LSTM - gated memory of how long a load has been on."""
    cell = nn.LSTM


class GRUNet(_RNN):
    """Model B3: GRU - lighter recurrent alternative to the LSTM."""
    cell = nn.GRU


class CNNGRU(nn.Module):
    """Model B4: hybrid - CNN extracts switch edges, GRU models their temporal order."""

    def __init__(self, window, n_app, ch=32, hidden=64, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(2, ch, 5, padding=2), nn.BatchNorm1d(ch), nn.ReLU(),
                                  nn.Conv1d(ch, ch, 5, padding=2), nn.BatchNorm1d(ch), nn.ReLU(),
                                  nn.MaxPool1d(2))
        self.gru = nn.GRU(ch, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.heads = Heads(hidden, n_app)

    def forward(self, x):
        h = self.conv(two_channel(x)).transpose(1, 2)
        out, _ = self.gru(h)
        return self.heads(self.drop(out[:, -1]))


class TCN(nn.Module):
    """Dilated temporal convolutions with residual connections."""

    def __init__(self, window, n_app):
        super().__init__()
        self.input = nn.Conv1d(2, 16, 1)
        self.blocks = nn.ModuleList([
            nn.Conv1d(16, 16, 3, dilation=dilation, padding=dilation)
            for dilation in (1, 2, 4, 8)
        ])
        self.heads = Heads(32, n_app)

    def forward(self, x):
        h = self.input(two_channel(x)[:, :, ::2])
        for layer in self.blocks:
            h = h + torch.relu(layer(h))
        return self.heads(torch.cat([h.mean(-1), h[:, :, -1]], dim=1))


class Transformer(nn.Module):
    """Self-attention across ordered six-minute patches."""

    def __init__(self, window, n_app):
        super().__init__()
        patch_width = min(6, window)
        self.patch = nn.Conv1d(2, 32, patch_width, stride=patch_width)
        self.position = nn.Parameter(torch.zeros(1, window // patch_width, 32))
        self.attention = nn.TransformerEncoderLayer(32, 4, 64, batch_first=True)
        self.heads = Heads(64, n_app)

    def forward(self, x):
        patch_width = self.patch.kernel_size[0]
        offset = x.shape[1] % patch_width
        tokens = self.patch(two_channel(x)[:, :, offset:]).transpose(1, 2)
        h = self.attention(tokens + self.position)
        return self.heads(torch.cat([h.mean(1), h[:, -1]], dim=1))


class Inception1D(nn.Module):
    """Parallel temporal filters capture short and sustained appliance events."""

    def __init__(self, window, n_app):
        super().__init__()
        self.filters = nn.ModuleList([
            nn.Conv1d(2, 12, width, padding=width // 2) for width in (3, 7, 15)
        ])
        self.heads = Heads(72, n_app)

    def forward(self, x):
        signal = two_channel(x)
        h = torch.cat([torch.relu(layer(signal)) for layer in self.filters], dim=1)
        return self.heads(torch.cat([h.mean(-1), h[:, :, -1]], dim=1))


MODELS = {"MLP": MLP, "CNN1D": CNN1D, "LSTM": LSTMNet, "GRU": GRUNet, "CNN-GRU": CNNGRU,
          "TCN": TCN, "Transformer": Transformer, "Inception1D": Inception1D}

DESCRIPTIONS = {
    "MLP": "Baseline. Dense layers on the flattened 60-min window; no notion of temporal order.",
    "CNN1D": "Convolutions detect local step/edge patterns caused by appliances switching.",
    "LSTM": "Recurrent with gated long memory; tracks how long a load has been running.",
    "GRU": "Recurrent, fewer gates than LSTM: faster, often equally accurate.",
    "CNN-GRU": "Hybrid: conv edge detector feeding a GRU that models the event sequence.",
    "TCN": "Residual dilated temporal convolutions capture events at several timescales.",
    "Transformer": "Self-attention relates short patches across the whole input window.",
    "Inception1D": "Parallel convolution widths detect both brief and sustained changes.",
}


def build(name, window, n_app):
    return MODELS[name](window, n_app)


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
