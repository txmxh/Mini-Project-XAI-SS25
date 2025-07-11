
import torch
from torch_geometric.data import Data
import pandas as pd

# File paths
task_file = "/content/data/completeDataset.tsv"
train_file = "/content/data/trainingSet.tsv"
test_file = "/content/data/testSet.tsv"
label_header = 'label_affiliation'
nodes_header = 'person'

edge_type = data.edge_type

# Read label file and build label mapping
labels_df = pd.read_csv(task_file, sep='	')
label_set = set(labels_df[label_header].values.tolist())
labels_dict = {lab: i for i, lab in enumerate(sorted(label_set))}

# Load training set
train_labels_df = pd.read_csv(train_file, sep='	')
train_indices, train_labels = [], []
for nod, lab in zip(train_labels_df[nodes_header].values,
                    train_labels_df[label_header].values):
    if nod in nodes_dict:
        train_indices.append(nodes_dict[nod])
        train_labels.append(labels_dict[lab])

# Load test set
test_labels_df = pd.read_csv(test_file, sep='	')
test_indices, test_labels = [], []
for nod, lab in zip(test_labels_df[nodes_header].values,
                    test_labels_df[label_header].values):
    if nod in nodes_dict:
        test_indices.append(nodes_dict[nod])
        test_labels.append(labels_dict[lab])

# Convert to tensors
train_idx = torch.tensor(train_indices, dtype=torch.long)
train_y = torch.tensor(train_labels, dtype=torch.long)
test_idx = torch.tensor(test_indices, dtype=torch.long)
test_y = torch.tensor(test_labels, dtype=torch.long)

# Final Data object
data = Data(
    x=x_combined,
    edge_index=edge_index,
    edge_type=edge_type,
    train_idx=train_idx,
    train_y=train_y,
    test_idx=test_idx,
    test_y=test_y,
    num_nodes=len(nodes)
)

# Metadata
num_nodes = len(nodes)
num_relations = int(edge_type.max().item()) + 1
num_classes = int(train_y.max().item()) + 1

print("Data object created with:")
print(f"Train samples : {len(train_idx)}")
print(f"Test samples  : {len(test_idx)}")
print(f"Node features : {data.x.shape[1]}")
print(f"Num classes   : {num_classes}")
print(f"Num nodes     : {num_nodes}")
print(f"Num relations : {num_relations}")
