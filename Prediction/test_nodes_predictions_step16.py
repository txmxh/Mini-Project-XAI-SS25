
import torch
@torch.no_grad()
def predict_for_test_nodes(model, data, inv_label_map):
    model.eval()
    out = model(data.x_selected, data.edge_index, data.edge_type)
    preds = out[data.test_idx].argmax(dim=1).cpu().numpy()
    node_ids = data.test_idx.cpu().numpy()

    predictions = []
    for node_id, pred_idx in zip(node_ids, preds):
        label_uri = inv_label_map[pred_idx]
        predictions.append((node_id, pred_idx, label_uri))

    return predictions
label_map = {label: idx for idx, label in enumerate(sorted(labels_df['label_affiliation'].unique()))}
inv_label_map = {v: k for k, v in label_map.items()}


# Get predictions
predictions = predict_for_test_nodes(model, data, inv_label_map)

#first few predictions to print
print("Predictions on test nodes:")
for node_id, pred_idx, uri in predictions[:50]:
    print(f"Node {node_id} → Class {pred_idx} → URI: {uri}")

#rest is saving upto 50
output_file = "test_predictions.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write("Predictions on test nodes:
")
    for node_id, pred_idx, uri in predictions:
        f.write(f"Node {node_id} → Class {pred_idx} → URI: {uri}
")

print(f"Saved predictions to {output_file}")
