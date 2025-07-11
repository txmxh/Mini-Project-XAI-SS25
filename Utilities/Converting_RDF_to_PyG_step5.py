
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import OneHotDegree
from torch_geometric.utils import index_sort
from collections import Counter

#Strategy 1
def build_graph_from_rdf_strategy1(rdf_graph, one_hot_max_degree=2000):
    print("Converting RDF graph to PyG format...")

    subjects = set(rdf_graph.subjects())
    objects = set(rdf_graph.objects())
    nodes = sorted(subjects.union(objects))
    predicates = list(rdf_graph.predicates())
    relations = sorted(set(predicates))   

    N = len(nodes)
    R = 2 * len(relations)

    # Mappings
    nodes_dict = {str(n): i for i, n in enumerate(nodes)}
    relations_dict = {p: i for i, p in enumerate(relations)}

    # Create edges (with inverse)
    edges = []
    for s, p, o in rdf_graph:
        if str(s) not in nodes_dict or str(o) not in nodes_dict:
            continue
        src = nodes_dict[str(s)]
        dst = nodes_dict[str(o)]
        rel_id = relations_dict[p]
        edges.append([src, dst, 2 * rel_id])       # forward
        edges.append([dst, src, 2 * rel_id + 1])   # inverse

    if not edges:
        raise ValueError("No valid RDF edges found!")

    edge = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Strategy 1 sort technique here: N * R * src + R * dst + rel
    _, perm = index_sort(N * R * edge[0] + R * edge[1] + edge[2])
    edge = edge[:, perm]
    edge_index, edge_type = edge[:2], edge[2]

    # Generate edge type names as tuples ('v', 'r{rel_id}', 'v')
    edge_type_names = []
    for e in edge_type:
        rel_id = int(e)
        edge_type_names.append(('v', f"r{rel_id}", 'v'))


    # Node features: OneHotDegree
    temp_data = Data(edge_index=edge_index, num_nodes=N)
    temp_data = OneHotDegree(max_degree=one_hot_max_degree - 1)(temp_data)
    x = temp_data.x

    # Final data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        num_nodes=N
    )
    data.edge_type_names = edge_type_names

    print(" Graph construction complete.")
    print(f"Nodes           : {data.num_nodes:,}")
    print(f"Feature dim     : {x.shape[1]}")
    print(f"Total edges     : {data.edge_index.shape[1]:,}")
    print(f"Relation types  : {int(edge_type.max().item()) + 1}")

    return data, nodes_dict, relations_dict

data, nodes_dict, rel_dict = build_graph_from_rdf_strategy1(graph)
