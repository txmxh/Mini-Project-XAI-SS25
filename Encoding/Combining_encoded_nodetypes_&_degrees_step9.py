
import numpy as np 
# Combine degree and type features only 
x_combined = torch.cat([x_deg, x_type], dim=1)

deg_feature_names = [f"degree_{i:04d}" for i in range(x_deg.shape[1])]
type_feature_names = [f"type_{i}" for i in range(x_type.shape[1])]

feature_names = deg_feature_names + type_feature_names
feature_mask = x_combined.abs().sum(dim=0) > 0
x_filtered = x_combined[:, feature_mask]

# Update metadata
feature_names_arr = np.array(feature_names)
feature_names_filtered = feature_names_arr[feature_mask.cpu().numpy()]
in_channels = x_filtered.shape[1]

# Assign to data
data.x = x_filtered

# Final check and print
print(f"Final feature matrix shape: {x_filtered.shape}")
print(f"Number of active features: {in_channels}")
print(f"Example active features: {feature_names_filtered[:10]}")

