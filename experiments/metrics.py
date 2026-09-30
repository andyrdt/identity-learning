import torch
from einops import einsum, reduce


def rms(values):
    squared_norm = reduce(values.square(), "run sample coord -> run sample", "sum")
    return reduce(squared_norm, "run sample -> run", "mean").sqrt()


@torch.no_grad()
def functional(model, *, samples, seed):
    """Sum of relative branch RMS values, and relative clean function error."""
    weight = next(model.parameters())
    runs, dimension, _ = weight.shape
    random = torch.Generator(device=weight.device).manual_seed(seed)
    inputs = torch.randn(runs, samples, dimension, generator=random, device=weight.device)
    input_rms = rms(inputs)
    hidden = inputs
    branch_rms = torch.zeros(runs, device=weight.device)
    for layer in range(len(model.input_weights)):
        residual = model.branch(hidden, layer)
        branch_rms += rms(residual) / input_rms
        hidden = hidden + residual
    function_error = rms(hidden - inputs) / input_rms
    return branch_rms.cpu().numpy(), function_error.cpu().numpy()


@torch.no_grad()
def residual_norm(model):
    """Sum of Frobenius norms of the residual maps VU, per run."""
    norms = []
    for input_weight, output_weight in zip(model.input_weights, model.output_weights, strict=True):
        residual = einsum(
            output_weight, input_weight,
            "run output feature, run feature input -> run output input",
        )
        norms.append(reduce(residual.square(), "run output input -> run", "sum").sqrt())
    return torch.stack(norms).sum(0).cpu().numpy()
