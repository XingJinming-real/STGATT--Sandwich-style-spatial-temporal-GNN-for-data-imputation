import torch.nn as nn
import torch_geometric.nn as gnn
import torch
import torch.nn.functional as F


class FFN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.inProject = nn.Linear(1, input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 1),
        )

    def forward(self, x):
        # x is Batch x Time x 1
        x = self.inProject(x)
        return self.net(x)  # Batch x Time x 1


class BGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.inProject = nn.Linear(1, input_dim)
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            batch_first=True,
            num_layers=num_layers,
            dropout=0.1,
            bidirectional=True,
        )
        self.out = nn.Linear(hidden_dim * 2, 1)  # *2 for bidirectional

    def forward(self, x):
        x = self.inProject(x)
        x, _ = self.gru(x)
        x = F.relu(x)
        x = self.out(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads):
        super().__init__()
        self.inProject = nn.Linear(1, input_dim)
        self.encoderLayer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(self.encoderLayer, num_layers=num_layers)
        self.out = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.inProject(x)
        x = self.encoder(x)
        x = F.relu(x)
        x = self.out(x)
        return x
