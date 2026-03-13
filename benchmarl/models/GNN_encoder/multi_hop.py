import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, LayerNorm, Sequential

class GNNMultiHop(nn.Module):
    def __init__(self, gnn_kwargs, model_params):
        super().__init__()
        layer_sizes = model_params["layer_sizes"]
        weight_standardization = model_params["weight_standardization"]
        base_model = model_params["base_model"]
        assert len(layer_sizes) >= 1
        self.input_size, self.representation_size = (
            gnn_kwargs["in_channels"],
            layer_sizes[-1],
        )
        self.edge_dim = gnn_kwargs.get("edge_dim", None)
        layer_sizes.insert(0, gnn_kwargs["in_channels"])
        self.weight_standardization = weight_standardization

        layers = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            gnn_kwarg = {"in_channels": in_dim, "out_channels": out_dim}
            if self.edge_dim is not None:
                gnn_kwarg.update({"edge_dim": self.edge_dim})
            layers.append(
                (base_model(**gnn_kwarg), "x, edge_index -> x")
            )
            layers.append(LayerNorm(out_dim))
            layers.append(nn.PReLU())
        print(layers)
        self.model = Sequential("x, edge_index", layers)

    def forward(self, x, edge_index, edge_attr=None):
        if self.weight_standardization:
            self.standardize_weights()
        if edge_attr is None:
            return self.model(x, edge_index)
        
        return self.model(x, edge_index, edge_attr)

    def reset_parameters(self):
        self.model.reset_parameters()

    def standardize_weights(self):
        skipped_first_conv = False
        for m in self.model.modules():
            if isinstance(m, MessagePassing):
                if not skipped_first_conv:
                    skipped_first_conv = True
                    continue
                weight = m.lin.weight.data
                var, mean = torch.var_mean(weight, dim=1, keepdim=True)
                weight = (weight - mean) / (torch.sqrt(var + 1e-5))
                m.lin.weight.data = weight
