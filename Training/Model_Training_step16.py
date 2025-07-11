
import torch
import torch.nn.functional as F
from torch_geometric.nn import FastRGCNConv
import torch.optim as optim
import matplotlib.pyplot as plt
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

#Mask test node features
data.x_masked = data.x_selected.clone()
data.x_masked[data.test_idx] = 0.0

#model input dimensions
in_channels = data.x_selected.shape[1]
num_relations = int(data.edge_type.max()) + 1
num_classes = int(data.train_y.max()) + 1

class FastRGCN(torch.nn.Module):
    def __init__(self, in_channels, num_relations, num_classes):
        super().__init__()
        self.conv1 = FastRGCNConv(in_channels, 16, num_relations, num_bases=30)
        self.conv2 = FastRGCNConv(16, num_classes, num_relations, num_bases=30)

    def forward(self, x, edge_index, edge_type=None, **kwargs):
        if edge_type is None:
            edge_type = kwargs.get('edge_type', None)
        if edge_type is None:
            raise ValueError("Missing edge_type in forward()")

        x = self.conv1(x, edge_index, edge_type).relu()
        x = self.conv2(x, edge_index, edge_type)
        return F.log_softmax(x, dim=1)

# Initialize model
model = FastRGCN(in_channels, num_relations, num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training and testing functions
def train():
    model.train()
    optimizer.zero_grad()
    #out = model(data.x_selected, data.edge_index, data.edge_type)
    out = model(data.x_masked, data.edge_index, data.edge_type)
    loss = F.nll_loss(out[data.train_idx], data.train_y)
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test():
    model.eval()
    out = model(data.x_selected, data.edge_index, data.edge_type)
    pred = out.argmax(dim=1)
    train_acc = (pred[data.train_idx] == data.train_y).float().mean().item()
    test_acc = (pred[data.test_idx] == data.test_y).float().mean().item()
    return train_acc, test_acc

# Training loop
losses = []
train_accuracies = []
test_accuracies = []
times = []

for epoch in range(1, 51):
    start = time.time()
    loss = train()
    train_acc, test_acc = test()
    losses.append(loss)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    times.append(time.time() - start)

    print(f"Epoch {epoch:02d} | Loss: {loss:.4f}")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy : {test_acc:.4f}")
    print("-" * 40)

print(f"Median epoch time: {torch.tensor(times).median():.4f} s")

# Plot training curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(losses)
plt.title('Loss over Epochs')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(train_accuracies)
plt.title('Train Accuracy')
plt.ylim(0, 1)
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(test_accuracies)
plt.title('Test Accuracy')
plt.ylim(0, 1)
plt.grid(True)

plt.tight_layout()
plt.show()

#class WrapperModel(torch.nn.Module):
#    def __init__(self, model, edge_type):
#        super().__init__()
#        self.model = model
#        self.edge_type = edge_type

#    def forward(self, x, edge_index):
# only x and edge_index, edge_type fixed internally
#        return self.model(x, edge_index, self.edge_type)

