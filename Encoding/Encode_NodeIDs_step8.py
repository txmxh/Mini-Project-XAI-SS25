
def encode_node_ids(nodes_dict):
    nodes = sorted(subjects.union(objects))
    nodes_dict = {n: i for i, n in enumerate(nodes)}
    num_nodes = len(nodes_dict)
    id_tensor = torch.eye(num_nodes)
    id_feature_names = [f"node_{i:05d}" for i in range(num_nodes)]
    return id_tensor, id_feature_names

x_id, id_feature_names = encode_node_ids(nodes_dict)
print(f"Created node ID encoding of shape: {x_id.shape}")

