
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch_geometric.explain import Explainer, PGExplainer
from torch_geometric.utils import k_hop_subgraph
import os
from rdflib import URIRef

#Setup 
device = torch.device("cpu")
model_cpu = model.to(device).eval()
x_cpu = data.x_selected.cpu()
edge_index_cpu = data.edge_index.cpu()
edge_type_cpu = data.edge_type.cpu()

class WrapperModelForPG(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, **kwargs):
        edge_type = kwargs.get("edge_type", data.edge_type)
        return self.model(x, edge_index, edge_type)

wrapped_model_pg = WrapperModelForPG(model_cpu).to(device)

#Get Predictions 
with torch.no_grad():
    out = model_cpu(x_cpu, edge_index_cpu, edge_type=edge_type_cpu)
    preds = out.argmax(dim=1)

train_idx_cpu = data.train_idx.cpu()
train_targets = preds[train_idx_cpu]

#Train PGExplainer 
subset_size = 5
train_idx_subset = train_idx_cpu[:subset_size]
train_targets_subset = train_targets[:subset_size]

pg_explainer = Explainer(
    model=wrapped_model_pg,
    algorithm=PGExplainer(epochs=50, lr=0.01),
    explanation_type="phenomenon",
    node_mask_type=None,
    edge_mask_type="object",
    model_config=dict(
        mode="multiclass_classification",
        task_level="node",
        return_type="log_probs"
    )
)

print(f"Training PGExplainer on {subset_size} nodes for 50 epochs...")
for epoch in range(50):
    for node_idx, target in zip(train_idx_subset, train_targets_subset):
        node_idx = int(node_idx.item())
        target_label = int(target.item())

        idx_tensor = torch.tensor([0])
        target_tensor = torch.tensor([target_label])
        subset, sub_edge_index, _, edge_mask = k_hop_subgraph(
            node_idx=node_idx, num_hops=2, edge_index=edge_index_cpu,
            relabel_nodes=True, num_nodes=x_cpu.size(0)
        )
        sub_x = x_cpu[subset]
        sub_edge_type = edge_type_cpu[edge_mask]

        pg_explainer.algorithm.train(
            epoch,
            wrapped_model_pg,
            sub_x,
            sub_edge_index,
            index=idx_tensor,
            target=target_tensor,
            edge_type=sub_edge_type
        )
print("PGExplainer training done.")

# Explanation Output Format 
os.makedirs("Explanations_Visuals/PGExplainer_Node_Visuals", exist_ok=True)

name_pred = URIRef("http://swrc.ontoware.org/ontology#name")

def get_rdf_name(uri):
    subj = URIRef(uri)
    for s, p, o in graph.triples((subj, name_pred, None)):
        return str(o)
    return "(no name found)"

for node_idx in selected_nodes:
    node_idx_tensor = torch.tensor([node_idx])
    pred_class = int(preds[node_idx].item())
    true_class = int(data.test_y[data.test_idx == node_idx][0].item())
    status = "Correct" if pred_class == true_class else "Wrong"

    explanation = pg_explainer(
        x=x_cpu,
        edge_index=edge_index_cpu,
        edge_type=edge_type_cpu,
        index=node_idx_tensor,
        target=pred_class
    )
    edge_mask = explanation.edge_mask.detach().cpu().numpy()

    top_k = 5
    top_indices = edge_mask.argsort()[-top_k:][::-1]

    out = model(data.x_selected, data.edge_index, data.edge_type)
    probs = out[node_idx].softmax(dim=0)

    true_uri = inv_label_map[true_class]
    pred_uri = inv_label_map[pred_class]
    node_uri = id_to_uri.get(node_idx, f"node_{node_idx}")
    true_name = get_rdf_name(true_uri)
    pred_name = get_rdf_name(pred_uri)

    print(f"
PGExplainer Explanation for Node {node_idx}")
    print(f"• URI            : {node_uri}")
    print(f"• True class URI : {true_uri}")
    print(f"• Pred class URI : {pred_uri}")
    print(f"• Prediction     : {status}")
    print(f"• True RDF Name  : {true_name}")
    print(f"• Pred RDF Name  : {pred_name}")
    print(f"• Explanation    : {'This person was correctly predicted.' if status == 'correct' else f'This person was incorrectly predicted. True class is {true_class}, but predicted {pred_class}.'}")
    print(f"• Top edges      : {top_indices.tolist()}")

    # Save plot
    plt.figure(figsize=(4, 3))
    plt.bar(range(top_k), edge_mask[top_indices])
    plt.xticks(range(top_k), top_indices)
    plt.title(f"Top-{top_k} Important Edges (Node {node_idx})")
    plt.xlabel("Edge Index")
    plt.ylabel("Importance")
    plt.grid(True)
    plt.tight_layout()
    path = f"Explanations_Visuals/PGExplainer_Node_Visuals/node_{node_idx}.pdf"
    plt.savefig(path)
    plt.close()
    print(f"• Saved PDF      : {path}")

