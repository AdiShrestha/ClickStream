"""
The Siamese keystroke encoder: a bidirectional LSTM over the 11-key
chronological sequence, producing a fixed-size L2-normalized embedding.
Trained via triplet loss in src/train_encoder.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class KeystrokeEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, embedding_dim=16):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, embedding_dim)

    def forward(self, x):
        # x shape is batch n_keys 3, fixed length 11, no padding needed
        _, (h_n, _) = self.lstm(x)
        h_cat = torch.cat([h_n[-2], h_n[-1]], dim=1)
        embedding = F.normalize(self.fc(h_cat), p=2, dim=1)
        return embedding
