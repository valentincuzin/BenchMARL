import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, LayerNorm, Sequential

class GNNMultiHop(nn.Module):
    def __init__(self, gnn_kwargs, param):
        super().__init__()
        layer_sizes = param["layer_sizes"]
        weight_standardization = param["weight_standardization"]
        base_model = param["base_model"]
        assert len(layer_sizes) >= 1
        self.input_size, self.representation_size = gnn_kwargs["in_channels"], layer_sizes[-1]
        layer_sizes.insert(0, gnn_kwargs["in_channels"])
        # self.edge_dim = gnn_kwargs["edge_dim"]  # TODO gérér les paramètre et la edge dim correctement
        self.weight_standardization = weight_standardization

        layers = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append((base_model(in_dim, out_dim), "x, edge_index -> x"))
            layers.append(LayerNorm(out_dim))
            layers.append(nn.PReLU())
        print(layers)
        self.model = Sequential("x, edge_index", layers)

    def forward(self, x, edge_index):
        if self.weight_standardization:
            self.standardize_weights()
        return self.model(x, edge_index)

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
