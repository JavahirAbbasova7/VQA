import torch
import torch.nn as nn


class ScoringModel(nn.Module):
    def __init__(self, embed_dim):
        super(ScoringModel, self).__init__()
        self.score_net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.score_net(x)
