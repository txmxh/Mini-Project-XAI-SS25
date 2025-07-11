
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def visualize_and_save_neighborhood(node_idx, edge_index, save_dir):
    os.makedirs(save_dir, exist_ok=True) 
    edge_index_np = edge_index.cpu().numpy()
    G = nx.Graph()
    for i in range(edge_index_np.shape[1]):
        G.add_edge(int(edge_index_np[0, i]), int(edge_index_np[1, i]), idx=i)
    
    # Get 1-hop neighborhood
    try:
        node_neighbors = list(G.neighbors(node_idx))
        subgraph_nodes = set([node_idx] + node_neighbors)
        subgraph = G.subgraph(subgraph_nodes)
    except nx.NetworkXError:
        print(f"Node {node_idx} not found in graph")
        return
     
    plt.figure(figsize=(4, 4))
    pos = nx.spring_layout(subgraph, seed=42) 
    
    # Visualization with highlighted target node
    nx.draw_networkx_nodes(subgraph, pos,
                         node_color=['orange' if n == node_idx else 'lightblue' for n in subgraph.nodes()],
                         node_size=300)
    nx.draw_networkx_edges(subgraph, pos, width=1.5)
    nx.draw_networkx_labels(subgraph, pos, font_size=8)
    plt.title(f"1-hop Neighborhood of Node {node_idx}")
    plt.axis('off')
    plt.tight_layout()
    filename = f"node_{node_idx}_neighborhood.pdf"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, bbox_inches='tight', format='pdf')
    print(f"Saved to: {save_path}")
    plt.show()
    plt.close()
 
if __name__ == "__main__": 
    test_nodes = [471, 639, 659, 568]
    save_directory = "./neighborhood_visualizations"
     
    for node_idx in test_nodes:
        print(f"
Processing node {node_idx}...")
        visualize_and_save_neighborhood(
            node_idx=node_idx,
            edge_index=edge_index_cpu,
            save_dir=save_directory
        )
