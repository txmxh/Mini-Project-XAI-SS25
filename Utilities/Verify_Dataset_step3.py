# Checking if RDF file exists
import os
rdf_path = "data/aifbfixed_complete.n3"

if os.path.exists(rdf_path):
    print("RDF file is ready:", rdf_path)
else:
    print("RDF file not found. Check your ZIP contents.")
