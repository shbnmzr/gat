from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from models.gat import GAT
import torch
import torch.nn.functional as F

dataset = Planetoid(root='data/Cora', name='Cora', transform=NormalizeFeatures())
data = dataset[0]
