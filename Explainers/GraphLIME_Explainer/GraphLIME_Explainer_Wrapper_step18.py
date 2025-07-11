
# Wrapper to fix the model interface for GraphLIME
class WrapperModel(torch.nn.Module):
    def __init__(self, model, edge_type):
        super().__init__()
        self.model = model
        self.edge_type = edge_type

    def forward(self, x, edge_index):
        return self.model(x, edge_index, self.edge_type)

# Using the selected features
wrapped_model = WrapperModel(model, data.edge_type).cpu()
x_cpu = data.x_selected.cpu()
edge_index_cpu = data.edge_index.cpu()

