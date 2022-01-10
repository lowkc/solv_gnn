import torch.nn as nn

class UnifySize(nn.Module):
    """
    A layer to unify the feature size of nodes of different types.
    Each feature uses a linear fc layer to map the size.
    NOTE, after this transformation, each data point is just a linear combination of its
    feature in the original feature space (x_new_ij = x_ik w_kj), there is not mixing of
    feature between data points.
    Args:
        input_dim (dict): feature sizes of nodes with node type as key and size as value
        output_dim (int): output feature size, i.e. the size we will turn all the
            features to
    """

    def __init__(self, input_dim, output_dim):
        super(UnifySize, self).__init__()

        self.linears = nn.ModuleDict(
            {k: nn.Linear(size, output_dim, bias=False) for k, size in input_dim.items()}
        )
    
    def forward(self, feats):
        """
        Args:
            feats (dict): features dict with node type as key and feature as value
        Returns:
            dict: size adjusted features
        """
        return {k: self.linears[k](x) for k, x in feats.items()}

class LinearN(nn.Module):
    """
    N stacked linear layers.

    Args:
        in_size (int): input feature size
        out_sizes (list): size of each layer
        activations (list): activation function of each layer
        use_bias (list): whether to use bias for the linear layer
    """

    def __init__(self, in_size, out_sizes, activations, use_bias):
        super(LinearN, self).__init__()

        self.fc_layers = nn.ModuleList()
        for out, act, b in zip(out_sizes, activations, use_bias):
            self.fc_layers.append(nn.Linear(in_size, out, bias=b))
            self.fc_layers.append(act)
            in_size = out

    def forward(self, x):
        for layer in self.fc_layers:
            x = layer(x)
        return x

class MLP(nn.Module):
    """
    MLP with linear output
    """

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.input_dim = input_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = nn.ModuleList()
            self.batch_norms = nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)