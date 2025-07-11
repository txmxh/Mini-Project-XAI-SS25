
from rdflib import URIRef
from rdflib.namespace import RDF
import os
import csv
import torch
import pandas as pd
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import CaptumExplainer

#Mappings 
id_to_uri = {v: k for k, v in nodes_dict.items()}
rel_id_to_uri = {v: str(k) for k, v in rel_dict.items()}

task_file = '/content/data/completeDataset.tsv'
label_header = 'label_affiliation'
labels_df = pd.read_csv(task_file, sep='	')
label_set = sorted(labels_df[label_header].unique())
label_map = {label: idx for idx, label in enumerate(label_set)}
inv_label_map = {v: k for k, v in label_map.items()}


captum_explainer = Explainer(
    model=model,
    algorithm=CaptumExplainer('IntegratedGradients'),
    explanation_type='model',
    node_mask_type=None,
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='log_probs',
    ),
    threshold_config=dict(threshold_type='topk', value=5)
)

#One Node Per Class Precisely
class_to_node = {}
test_node_ids = data.test_idx.cpu().numpy()
test_labels = data.test_y.cpu().numpy()

for node_id, label in zip(test_node_ids, test_labels):
    if label not in class_to_node:
        class_to_node[label] = node_id
    if len(class_to_node) == len(set(test_labels)):
        break

selected_nodes = list(class_to_node.values())
print("Selected one test node per class:", selected_nodes)

#Extracting Top-k Edges 
def get_top_edges_with_uris(explanation, data, top_k=5):
    edge_mask = explanation.edge_mask
    top_edge_idxs = edge_mask.argsort(descending=True)[:top_k]
    edge_list = []
    for idx in top_edge_idxs.tolist():
        src, dst = data.edge_index[:, idx].tolist()
        rel_id = data.edge_type[idx].item()
        src_uri = id_to_uri.get(src, f"node_{src}")
        dst_uri = id_to_uri.get(dst, f"node_{dst}")
        pred_uri = rel_id_to_uri.get(rel_id, f"rel_{rel_id}")
        edge_list.append((idx, src_uri, dst_uri, pred_uri))
    return edge_list

#Get RDF Group Name 
def get_rdf_group_label(uri):
    subject_uri = URIRef(uri)
    name_pred = URIRef('http://swrc.ontoware.org/ontology#name')
    for _, _, o in graph.triples((subject_uri, name_pred, None)):
        return str(o)
    return "N/A"


os.makedirs("Explanations_Visuals/CaptumExplainer_Node_Visuals", exist_ok=True)
csv_path = "Explainers/Captum_Explanations_log.csv"

with open(csv_path, "w", newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "Node ID", "RDF URI",
        "True Class", "True URI", "True Group Label",
        "Predicted Class", "Predicted URI", "Predicted Group Label",
        "Top Edge Index", "Source URI", "Predicate URI", "Destination URI",
        "Prediction Status"
    ])

    out = model(data.x_selected, data.edge_index, data.edge_type)

    for node_idx in selected_nodes:
        logits = out[node_idx]
        pred_class = logits.argmax().item()
        true_class = data.test_y[data.test_idx == node_idx][0].item()
        correct = pred_class == true_class
        status = "Correct" if correct else "Wrong"

        node_uri = id_to_uri.get(node_idx, f"node_{node_idx}")
        pred_uri = inv_label_map[pred_class]
        true_uri = inv_label_map[true_class]
        pred_label = get_rdf_group_label(pred_uri)
        true_label = get_rdf_group_label(true_uri)

        print(f"
Node {node_idx} explanation:")
        print(f"• URI            : {node_uri}")
        print(f"• True class URI : {true_uri}")
        print(f"• Pred class URI : {pred_uri}")
        print(f"• Prediction     : {status}")
        print(f"• True RDF Name  : {true_label}")
        print(f"• Pred RDF Name  : {pred_label}")
        
        if correct:
            print(f"
→ This person was correctly predicted to belong to group {pred_class}.
")
        else:
            print(f"
→ This person was incorrectly predicted. True class is {true_class}, but predicted {pred_class}.
")


        explanation = captum_explainer(
            x=data.x_selected.clone().detach().requires_grad_(True),
            edge_index=data.edge_index,
            edge_type=data.edge_type,
            index=torch.tensor(node_idx)
        )

        pdf_path = f"Explanations_Visuals/CaptumExplainer_Node_Visuals/Expl_node_{node_idx}.pdf"
        explanation.visualize_graph(path=pdf_path)
        print(f"Saved explanation graph to {pdf_path}")

        top_edges = get_top_edges_with_uris(explanation, data, top_k=5)
        for edge_idx, src_uri, dst_uri, pred_rel in top_edges:
            writer.writerow([
                node_idx, node_uri,
                true_class, true_uri, true_label,
                pred_class, pred_uri, pred_label,
                edge_idx, src_uri, pred_rel, dst_uri,
                status
            ])

