
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import degree

N = len(nodes)
edge_index = data.edge_index
max_degree = 2000

# Encode degrees using OneHotDegree
temp_data = Data(edge_index=edge_index, num_nodes=N)
deg = degree(temp_data.edge_index[0], num_nodes=N, dtype=torch.long)
max_deg = int(deg.max().item()) + 1 
transform = T.OneHotDegree(max_degree=max_degree - 1, cat=False)

# Use dynamically computed max_degree
temp_data = OneHotDegree(max_degree=max_deg - 1)(temp_data)
x_deg = temp_data.x
feature_names = [f"degree_{i:04d}" for i in range(x_deg.shape[1])]
print(f"Encoded degree features for {x_deg.shape[0]} nodes with {x_deg.shape[1]} dimensions.")
print(f"Example feature names: {feature_names[:10]}")
