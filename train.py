import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=4, dropout=0.6):
        super().__init__()

        self.gat1 = GATConv(
            in_dim,
            hidden_dim,
            heads=heads,
            dropout=dropout,
        )

        # output layer: concatenate heads -> hidden_dim * heads
        self.gat2 = GATConv(
            hidden_dim * heads,
            out_dim,
            heads=1,
            concat=False,
            dropout=dropout,
        )
