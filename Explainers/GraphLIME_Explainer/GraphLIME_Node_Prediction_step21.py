
import os
import matplotlib.pyplot as plt
import numpy as np
from torch import no_grad

@no_grad()
def get_prediction_probs(model, x, edge_index, nodes):
    model.eval()
    logits = model(x, edge_index)   
    probs = logits.exp()
    return {node: probs[node].cpu().numpy() for node in nodes}

def plot_top_classes(node, probs, top_k=3, save_dir=None):
    # Get exactly top_k classes (not k+1)
    top_indices = np.argsort(probs)[-top_k:][::-1]  
    top_probs = probs[top_indices]
    
    # Create figure with proper dimensions
    plt.figure(figsize=(10, 5))
    bars = plt.bar(range(top_k), top_probs, 
                  tick_label=top_indices,
                  color='skyblue',
                  edgecolor='navy')
    
    # Customize plot
    plt.xlabel("Class Index", fontsize=12)
    plt.ylabel("Probability", fontsize=12)
    plt.title(f"Top {top_k} Predicted Classes for Node {node}", pad=20)
    plt.ylim(0, min(1.1, max(top_probs)*1.2))  
     
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom',
                fontsize=10)
    
    plt.grid(axis='y', alpha=0.4)
    plt.tight_layout()   
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"node_{node}_top_{top_k}_predicted_classes.pdf"
        plt.savefig(os.path.join(save_dir, filename), 
                   dpi=300, 
                   bbox_inches='tight')
    
    plt.show()
    plt.close()
 
save_directory = "./Explanations_Visuals/GraphLIME_Node_Prediction_Visuals"
os.makedirs(save_directory, exist_ok=True)

# Get and plot predictions
subset_nodes = list(explanations_subset.keys())
pred_probs = get_prediction_probs(wrapped_model, x_cpu, edge_index_cpu, subset_nodes)

for node, probs in pred_probs.items():
    plot_top_classes(node, probs, top_k=3, save_dir=save_directory)
