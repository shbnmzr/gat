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


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = load_dataset()
    data = dataset[0].to(device)

    model = GAT(
        in_dim=dataset.num_node_features,
        hidden_dim=8,
        out_dim=dataset.num_classes,
        heads=8
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.005,
        weight_decay=5e-4
    )

    print('Training GAT on Cora...')
    for epoch in range(1, 201):
        loss = train(model, data, optimizer)
        train_acc, val_acc, test_acc = test(model, data)

        print(f'Epoch: {epoch:03d} | \n'
              f'Loss: {loss:.4f} | \n'
              f'Val Accuracy: {val_acc:.4f} | \n'
              f'Test Accuracy: {test_acc:.4f}')


if __name__=='__main__':
    main()
