import torch
from torch import nn
from einops import einsum


class LinearChain(nn.Module):
    """A sequence of linear layers with effective weights I + W."""

    def __init__(self, residual_weights):
        super().__init__()
        self.weights = nn.ParameterList([
            nn.Parameter(weight.clone()) for weight in residual_weights
        ])

    def forward(self, inputs):
        hidden = inputs
        for weight in self.weights:
            residual = einsum(
                hidden, weight,
                "run sample input, run output input -> run sample output",
            )
            hidden = hidden + residual
        return hidden


class ResidualBlocks(nn.Module):
    """Blocks h + V phi(Uh); direct blocks fix U=I, factored blocks train U."""

    def __init__(self, residual_weights, *, factored, activation):
        super().__init__()
        if activation not in ("linear", "relu"):
            raise ValueError(f"Unknown activation: {activation}")
        self.activation = nn.ReLU() if activation == "relu" else nn.Identity()
        self.input_weights = nn.ParameterList()
        self.output_weights = nn.ParameterList()
        for residual in residual_weights:
            if factored:
                # R = L diag(s) R^T. Share each singular value equally between V and U.
                left, singular_values, right_transpose = torch.linalg.svd(residual)
                root = singular_values.sqrt()
                input_weight = einsum(
                    root, right_transpose, "run mode, run mode input -> run mode input",
                )
                output_weight = einsum(
                    left, root, "run output mode, run mode -> run output mode",
                )
            else:
                dimension = residual.shape[-1]
                input_weight = torch.eye(dimension, device=residual.device).expand_as(residual)
                output_weight = residual
            self.input_weights.append(nn.Parameter(input_weight.clone(), requires_grad=factored))
            self.output_weights.append(nn.Parameter(output_weight.clone()))

    def branch(self, hidden, layer):
        features = einsum(
            hidden, self.input_weights[layer],
            "run sample input, run feature input -> run sample feature",
        )
        return einsum(
            self.activation(features), self.output_weights[layer],
            "run sample feature, run output feature -> run sample output",
        )

    def forward(self, inputs):
        hidden = inputs
        for layer in range(len(self.input_weights)):
            hidden = hidden + self.branch(hidden, layer)
        return hidden
