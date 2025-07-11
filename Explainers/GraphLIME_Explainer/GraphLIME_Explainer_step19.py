

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoLars
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph
import os
from rdflib import URIRef

#Setup
class WrapperModel(torch.nn.Module):
    def __init__(self, model, edge_type):
        super().__init__()
        self.model = model
        self.edge_type = edge_type

    def forward(self, x, edge_index):
        return self.model(x, edge_index, self.edge_type)

class GraphLIME:
    def __init__(self, model, hop=2, rho=0.1, cached=True):
        self.hop = hop
        self.rho = rho
        self.model = model
        self.cached = cached
        self.cached_result = None
        self.model.eval()

    def __flow__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    def __subgraph__(self, node_idx, x, y, edge_index, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx, self.hop, edge_index, relabel_nodes=True,
            num_nodes=num_nodes, flow=self.__flow__())
        x = x[subset]
        y = y[subset]
        for key, item in kwargs.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item
        return x, y, edge_index, mapping, edge_mask, kwargs

    def __init_predict__(self, x, edge_index, **kwargs):
        if self.cached and self.cached_result is not None:
            if x.size(0) != self.cached_result.size(0):
                raise RuntimeError('Cached {} nodes, got {}.'.format(
                    self.cached_result.size(0), x.size(0)))
        if not self.cached or self.cached_result is None:
            with torch.no_grad():
                log_logits = self.model(x=x, edge_index=edge_index, **kwargs)
                probas = log_logits.exp()
            self.cached_result = probas
        return self.cached_result

    def __compute_kernel__(self, x, reduce):
        n, d = x.shape
        dist = (x.reshape(1, n, d) - x.reshape(n, 1, d)) ** 2
        if reduce:
            dist = np.sum(dist, axis=-1, keepdims=True)
        std = np.sqrt(d)
        return np.exp(-dist / (2 * std ** 2 * 0.1 + 1e-10))

    def __compute_gram_matrix__(self, x):
        G = x - np.mean(x, axis=0, keepdims=True)
        G = G - np.mean(G, axis=1, keepdims=True)
        G = G / (np.linalg.norm(G, ord='fro', axis=(0, 1), keepdims=True) + 1e-10)
        return G

    def explain_node(self, node_idx, x, edge_index, **kwargs):
        if not isinstance(node_idx, torch.Tensor):
          node_idx = torch.tensor([node_idx])   
        elif node_idx.dim() == 0:
          node_idx = node_idx.unsqueeze(0)
        probas = self.__init_predict__(x, edge_index, **kwargs)
        x_sub, probas_sub, *_ = self.__subgraph__(node_idx, x, probas, edge_index, **kwargs)
        x_sub = x_sub.detach().cpu().numpy()
        y_sub = probas_sub.detach().cpu().numpy()
        n, d = x_sub.shape
        K = self.__compute_kernel__(x_sub, reduce=False)
        L = self.__compute_kernel__(y_sub, reduce=True)
        K_bar = self.__compute_gram_matrix__(K).reshape(n ** 2, d)
        L_bar = self.__compute_gram_matrix__(L).reshape(n ** 2,)
        solver = LassoLars(alpha=self.rho, fit_intercept=False, positive=True)
        solver.fit(K_bar * n, L_bar * n)
        return solver.coef_
 
wrapped_model = WrapperModel(model, data.edge_type).cpu()
x_cpu = data.x_selected.cpu()
edge_index_cpu = data.edge_index.cpu()

explainer = GraphLIME(wrapped_model, hop=2, rho=0.1, cached=True)

os.makedirs("Explanations/GraphLIME_Node_Visuals", exist_ok=True)

# Select one node per class
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

# RDF mapping
name_pred = URIRef("http://swrc.ontoware.org/ontology#name")

def get_rdf_name(uri):
    subj = URIRef(uri)
    for s, p, o in graph.triples((subj, name_pred, None)):
        return str(o)
    return "(no name found)"

# Run GraphLIME on each
for node_to_explain in selected_nodes:
    feat_imp = explainer.explain_node(node_to_explain, x_cpu, edge_index_cpu)

    top_k = 5
    feat_abs = np.abs(feat_imp)
    top_idx = np.argsort(feat_abs)[-top_k:][::-1]
    top_vals = feat_abs[top_idx]

    plt.figure(figsize=(6, 3))
    plt.bar(range(top_k), top_vals, tick_label=top_idx)
    plt.title(f"GraphLIME Feature Importance for Node {node_to_explain}")
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.tight_layout()
    pdf_path = f"Explanations/GraphLIME_Node_Visuals/node_{node_to_explain}.pdf"
    plt.savefig(pdf_path)
    plt.close()

    out = model(data.x_selected, data.edge_index, data.edge_type)
    pred_class = out[node_to_explain].argmax().item()
    true_class = data.test_y[data.test_idx == node_to_explain][0].item()
    status = "Correct" if true_class == pred_class else "Wrong"

    node_uri = id_to_uri.get(node_to_explain, f"node_{node_to_explain}")
    pred_uri = inv_label_map[pred_class]
    true_uri = inv_label_map[true_class]
    pred_name = get_rdf_name(pred_uri)
    true_name = get_rdf_name(true_uri)

    print(f"
GraphLIME Explanation for Node {node_to_explain}")
    print(f"• URI            : {node_uri}")
    print(f"• True class URI : {true_uri}")
    print(f"• Pred class URI : {pred_uri}")
    print(f"• Prediction     : {status}")
    print(f"• True RDF Name  : {true_name}")
    print(f"• Pred RDF Name  : {pred_name}")
    print(f"• Explanation    : {'This person was correctly predicted.' if status == 'Correct' else f'This person was incorrectly predicted. True class is {true_class}, but predicted {pred_class}.'}")
    print(f"• Top features   : {top_idx.tolist()}")
    print(f"• Saved PDF      : {pdf_path}")

