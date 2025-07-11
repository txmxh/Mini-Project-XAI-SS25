
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
var_mask = selector_var.get_support() # masks from VarianceThreshold and SelectFromModel 
model_mask = selector_model.get_support()    
final_mask = np.zeros(len(var_mask), dtype=bool) #Buildimg combined final mask over original full feature set
final_mask[var_mask] = model_mask            
X_full = x_combined   #Applying mask to full node feature matrix (x_combined)
x_selected_all = X_full[:, final_mask] 
data.x_selected = x_selected_all.to(device)
print("x_selected shape:", data.x_selected.shape)
