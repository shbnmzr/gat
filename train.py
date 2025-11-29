from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from models.gat import GAT
import torch
import torch.nn.functional as F

# Load dataset
def load_dataset():
    dataset = Planetoid(
        root='data/Cora',
        name='Cora',
        transform=NormalizeFeatures(),
    )

    return dataset
