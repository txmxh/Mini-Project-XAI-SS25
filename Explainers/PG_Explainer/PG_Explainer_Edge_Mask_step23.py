
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def visualize_node_subgraph(edge_index, edge_mask, node_idx, threshold=0.01): 
    # Convert to numpy arrays  
    if hasattr(edge_index, 'cpu'):
        edge_index_np = edge_index.cpu().numpy()
    else:
        edge_index_np = np.array(edge_index)
    
    if hasattr(edge_mask, 'cpu'):
        edge_mask_np = edge_mask.cpu().numpy()
    else:
        edge_mask_np = np.array(edge_mask)
    
    # Get important edges
    important_edges_idx = (edge_mask_np > threshold).nonzero()[0]
    
    # Fallback if no edges pass threshold
    if len(important_edges_idx) == 0:
        print(f"Node {node_idx}: No edges > {threshold}, using top-5")
        important_edges_idx = edge_mask_np.argsort()[-5:]

    # Create subgraph
    G = nx.Graph()
    edge_weights = []
    for idx in important_edges_idx:
        u, v = map(int, edge_index_np[:, idx])
        G.add_edge(u, v)
        edge_weights.append(edge_mask_np[idx])
    
    # Visualization
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    
    node_colors = ['orange' if n == node_idx else 'lightblue' for n in G.nodes()]
    node_sizes = [800 if n == node_idx else 400 for n in G.nodes()]
    
    if len(edge_weights) > 0:
        norm_weights = (edge_weights - min(edge_weights)) / (max(edge_weights) - min(edge_weights) + 1e-8)
        nx.draw_networkx_edges(G, pos, width=np.array(norm_weights)*4 + 1,
                             edge_color=plt.cm.Reds(norm_weights))
    
    # Draw nodes and labels
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)
    nx.draw_networkx_labels(G, pos, font_size=10)
    
   
    for (u, v), w in zip(G.edges(), edge_weights):
        plt.text((pos[u][0]+pos[v][0])/2, 
               (pos[u][1]+pos[v][1])/2,
               f"{w:.4f}", 
               bbox=dict(facecolor='white', alpha=0.7),
               fontsize=8, ha='center', va='center')
    
    plt.title(f"Node {node_idx}
Max: {max(edge_weights):.4f}" if edge_weights else f"Node {node_idx}
No important edges")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
 
if __name__ == "__main__":
    # Configuration
    nodes_to_plot = [471, 639, 659, 568]
    threshold = 0.01
    
    # Generate and visualize for each node
    for node_idx in nodes_to_plot: 
        edge_mask = explainer.explain_node(node_idx, x_cpu, edge_index_cpu)
         
        if hasattr(edge_mask, 'cpu'):
            edge_mask = edge_mask.cpu().numpy()
        
        print(f"
Node {node_idx} Edge Mask Stats:")
        print(f"Min: {edge_mask.min():.6f}")
        print(f"Max: {edge_mask.max():.6f}")
        print(f"Mean: {edge_mask.mean():.6f}
")
        
        visualize_node_subgraph(edge_index_cpu, edge_mask, node_idx, threshold)

