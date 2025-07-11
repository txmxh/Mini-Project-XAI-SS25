
import matplotlib.pyplot as plt
import numpy as np
import os

def explain_multiple_nodes(explainer, x, edge_index, node_indices, rhos=[0.1, 0.05, 0.01, 0.015]):
    results = {}
    for rho in rhos:
        print(f"
=== Explaining with rho={rho} ===")
        explainer.rho = rho
        for node in node_indices:
            feat_imp = explainer.explain_node(node, x, edge_index)
            results[(node, rho)] = feat_imp
            nonzero_count = np.count_nonzero(feat_imp)
            print(f"Node {node}: {nonzero_count} non-zero features (rho={rho})")
    return results

def plot_feature_importance(feat_imp, top_k=10, title="Feature Importance", save_dir=None):
    feat_imp = np.abs(feat_imp)
    top_idx = np.argsort(feat_imp)[-top_k:][::-1]
    top_vals = feat_imp[top_idx]

    plt.figure(figsize=(8, 4))
    plt.bar(range(top_k), top_vals, tick_label=top_idx)
    plt.xlabel("Feature index")
    plt.ylabel("Importance")
    plt.title(title)
    
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = title.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".pdf"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()
    plt.close()

# Use 4 nodes from test set
test_nodes = data.test_idx[:4].cpu().tolist()
results = explain_multiple_nodes(explainer, x_cpu, edge_index_cpu, test_nodes)

# Plot feature importance for selected nodes
node_to_plot = 471
rho_to_plot = 0.1
node_to_plot2 = 659
rho_to_plot2 = 0.05
node_to_plot3 = 555
rho_to_plot3 = 0.01
node_to_plot4 = 639
rho_to_plot4 = 0.015

feat_imp_vec = results.get((node_to_plot, rho_to_plot)) 
feat_imp_vec2 = results.get((node_to_plot2, rho_to_plot2))
feat_imp_vec3 = results.get((node_to_plot3, rho_to_plot3))
feat_imp_vec4 = results.get((node_to_plot4, rho_to_plot4))

if feat_imp_vec is not None:
    save_dir = './Explanations_Visuals/GraphLIME_Node_Visuals_with_rho'
    nodes_to_plot = [
        (feat_imp_vec, node_to_plot, rho_to_plot),
        (feat_imp_vec2, node_to_plot2, rho_to_plot2),
        (feat_imp_vec3, node_to_plot3, rho_to_plot3),
        (feat_imp_vec4, node_to_plot4, rho_to_plot4)
    ]
    
    for feat_imp, node_id, rho in nodes_to_plot:
        if feat_imp is not None:
            plot_feature_importance(
                feat_imp, 
                top_k=3,
                title=f"GraphLIME importance for node {node_id} (rho={rho})",
                save_dir=save_dir  # Pass the save directory here
            )
        else:
            print(f"No explanation found for node {node_id} (rho={rho})")
else:
    print(f"No explanation found for node {node_to_plot} with rho={rho_to_plot}")

