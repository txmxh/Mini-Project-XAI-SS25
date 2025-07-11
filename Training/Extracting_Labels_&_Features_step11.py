
# Extract features of labeled nodes
labeled_features = data.x[data.train_idx]

# Extract their labels
labeled_labels = data.train_y

print(f"Labeled features shape: {labeled_features.shape}")
print(f"Labeled labels shape: {labeled_labels.shape}")

