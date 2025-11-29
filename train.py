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

def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)

    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    return loss.item()

@torch.no_grad()
def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    accuracies = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = pred[mask].eq(data.y[mask]).sum().item()
        acc = correct / mask.sum().item()
        accuracies.append(acc)

    return accuracies
