from copy import deepcopy

import torch
from einops import einsum, reduce

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gaussian_weights(runs, dimension, scale, *, seed, depth):
    """Residual entries have standard deviation scale / sqrt(dimension)."""
    random = torch.Generator(device=DEVICE).manual_seed(seed)
    return [
        torch.randn(runs, dimension, dimension, generator=random, device=DEVICE)
        * (scale / dimension**0.5)
        for _ in range(depth)
    ]


def matched_weights(runs, dimension, *, seed):
    """Two nonzero residuals with ||W1||_F=1 and (I+W2)(I+W1)=I."""
    first = gaussian_weights(runs, dimension, 1, seed=seed, depth=1)[0]
    norm = reduce(first.square(), "run row col -> run 1 1", "sum").sqrt()
    first = first / norm
    identity = torch.eye(dimension, device=DEVICE)
    second = torch.linalg.inv(identity + first) - identity
    return [first, second]


def train(model, noise_variance, *, steps, rate, batch, seed, decay=0,
          record=None, record_steps=()):
    """Train model in place and return a separate model with final-1% mean weights.

    `record(step, model)` observes current weights, including step zero if requested.
    The loss averages over samples and sums over coordinates AND independent runs.
    Averaging across runs instead would change each network's learning rate.
    """
    parameter = next(model.parameters())
    runs, dimension, _ = parameter.shape
    device = parameter.device
    random = torch.Generator(device=device).manual_seed(seed)
    noise_std = torch.as_tensor(noise_variance, device=device, dtype=parameter.dtype).sqrt()
    optimizer = torch.optim.SGD(model.parameters(), lr=rate)
    average = deepcopy(model)
    for weight in average.parameters():
        weight.requires_grad_(False)
        weight.zero_()
    average_steps = max(1, round(steps * 0.01))
    record_steps = set(record_steps)
    if record is not None and 0 in record_steps:
        with torch.no_grad():
            record(0, model)

    for step in range(1, steps + 1):
        inputs = torch.randn(runs, batch, dimension, generator=random, device=device)
        noise = torch.randn(runs, batch, dimension, generator=random, device=device)
        noise = einsum(noise, noise_std, "run sample coord, coord -> run sample coord")
        target = inputs + noise
        prediction = model(inputs)
        squared_error = (prediction - target).square()
        loss_per_sample = reduce(squared_error, "run sample coord -> run sample", "sum")
        loss = reduce(loss_per_sample, "run sample -> run", "mean").sum()
        if decay:
            loss = loss + decay * sum(weight.square().sum() for weight in model.parameters())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if step > steps - average_steps:
                for total, weight in zip(average.parameters(), model.parameters(), strict=True):
                    total.add_(weight)
            if record is not None and step in record_steps:
                record(step, model)
        if step % max(1, steps // 10) == 0:
            print(f"  {step:,}/{steps:,} steps", flush=True)

    for weight in average.parameters():
        weight.div_(average_steps)
        if not torch.isfinite(weight).all():
            raise RuntimeError("Training produced nonfinite weights")
    return average
