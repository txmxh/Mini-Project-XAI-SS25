
from sklearn.feature_selection import VarianceThreshold
import numpy as np

X_full = x_combined   

#extracting features for training nodes only
X_train_np = X_full[data.train_idx].cpu().numpy()

# Apply VarianceThreshold to reduce dimensions
selector_var = VarianceThreshold(threshold=0.01)
X_reduced = selector_var.fit_transform(X_train_np)
print("After VarianceThreshold:", X_reduced.shape)
