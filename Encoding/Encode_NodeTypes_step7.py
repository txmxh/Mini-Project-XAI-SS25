
import torch
from collections import Counter
from rdflib import URIRef
from rdflib.namespace import RDF as rdf
import rdflib
import pandas as pd
from collections import Counter
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="rdflib")

def encode_node_types(rdf_graph, nodes_dict):
    print("
Encoding RDF node types")

    types = [str(obj) for _, obj in rdf_graph.subject_objects(rdf.type)]
    freq = Counter(types)
    #nodes = sorted(subjects.union(objects))
    #nodes_dict = {n: i for i, n in enumerate(nodes)}
    types_sorted = sorted(subjects.union(objects))  # most frequent first
    type_to_idx =  {n: i for i, n in enumerate(nodes)}


    print(f"Discovered {len(type_to_idx)} distinct RDF types.")

    num_nodes = len(type_to_idx)
    type_tensor = torch.zeros((num_nodes, len(type_to_idx)), dtype=torch.float)

    for node_ref, idx in nodes_dict.items():  # already URIRef
      for obj in rdf_graph.objects(node_ref, rdf.type):
          if not isinstance(obj, URIRef):
              continue
          type_uri = str(obj)
          if type_uri in type_to_idx:
              type_tensor[idx, type_to_idx[type_uri]] = 1.0

    print(f"Created type encoding of shape: {type_tensor.shape}")
    return type_tensor, type_to_idx

x_type, type_to_idx = encode_node_types(graph,nodes_dict)
type_feature_names = [f"type_{i}" for i in range(len(type_to_idx))]
