
import rdflib
import pandas as pd
from collections import Counter
import time
from google.colab import files

start = time.time()

# Load RDF graph
print("Loading RDF graph...")
rdf_path = "data/aifbfixed_complete.n3"
graph = rdflib.Graph()
graph.parse(rdf_path)

print(f"RDF graph loaded successfully.")
print(f"Total RDF triples: {len(graph):,}
")

# Extract RDF components
subjects = set(graph.subjects())
objects = set(graph.objects())
predicates = list(graph.predicates())

nodes = subjects.union(objects)
unique_predicates = set(predicates)

print("Graph Summary:")
print(f"Unique nodes (subjects ∪ objects): {len(nodes):,}")
print(f"Unique predicate types:            {len(unique_predicates):,}")
print(f"Total predicate instances:         {len(predicates):,}
")

# Show all predicate types sorted by frequency
print("Predicate frequency (all types):")
predicate_counts = Counter(predicates)
for idx, (pred, count) in enumerate(predicate_counts.most_common(), 1):
    print(f"{idx:2d}. {str(pred):60} → {count:5,} triples")

# Show only first 20 RDF triples for preview
print("
Sample RDF triples (first 20 shown):")
for idx, (s, p, o) in enumerate(graph):
    print(f"{idx+1:2d}. ({s}, {p}, {o})")
    if idx >= 19:
        break

# Export all the rest of RDF triples to a file including top 20's.
export = True
if export:
    print("
Saving all RDF triples to 'all_triples.txt'...")
    with open("all_triples.txt", "w", encoding="utf-8") as f:
        for s, p, o in graph:
            f.write(f"{s}	{p}	{o}
")
    print("Export complete.")
    files.download("all_triples.txt")

end = time.time()
print(f"
RDF graph analysis completed in {end - start:.2f} seconds.")
