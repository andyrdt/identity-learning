import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import torch
from einops import einsum, repeat

from experiments.common import BLUE, ORANGE, GRAY, folders, save_data, save_figure, style
from experiments.theory import predicted_gram
from experiments.train import DEVICE

CACHE, FIGURES = folders(__file__)
CONDITIONS = (3., 5., 10.)
ANGLES = (0., 45., 90.)
RATES = (0.02, 0.01, 0.005)
RUNS, BATCH, SEED = 8, 8, 94107
DURATION, INTERVAL = 20., 0.05  # Slow time tau = rate**2 * step / batch.


def cache_files():
    return tuple(CACHE / f"rate_{rate:g}.npz" for rate in RATES)


def geometry():
    """Covariance factors L (with L L^T = covariance) and nine Gram predictions."""
    cases, input_roots, noise_roots, predictions = [], [], [], []
    for condition in CONDITIONS:
        eigenvalues = np.array([2 / (1 + condition), 2 * condition / (1 + condition)])
        noise_covariance = np.diag(eigenvalues[::-1])
        for degrees in ANGLES:
            angle = np.deg2rad(degrees)
            rotation = np.array([[np.cos(angle), -np.sin(angle)],
                                 [np.sin(angle), np.cos(angle)]])
            input_root = einsum(rotation, np.sqrt(eigenvalues), "output mode, mode -> output mode")
            input_covariance = einsum(input_root, input_root, "row mode, col mode -> row col")
            cases.append((condition, degrees))
            input_roots.append(input_root)
            noise_roots.append(np.diag(np.sqrt(eigenvalues[::-1])))
            predictions.append(predicted_gram(input_covariance, noise_covariance))
    return np.array(cases), np.array(input_roots), np.array(noise_roots), np.array(predictions)


def simulate(rate):
    """Autograd SGD on effective weights; all cases share standardized draws."""
    cases, input_roots, noise_roots, predictions = geometry()
    input_roots = torch.tensor(input_roots, device=DEVICE, dtype=torch.float32)
    noise_roots = torch.tensor(noise_roots, device=DEVICE, dtype=torch.float32)
    identity = torch.eye(2, device=DEVICE)
    first = torch.nn.Parameter(repeat(identity, "row col -> case run row col", case=len(cases), run=RUNS).clone())
    second = torch.nn.Parameter(first.detach().clone())
    optimizer = torch.optim.SGD([first, second], lr=rate)
    random = torch.Generator(device=DEVICE).manual_seed(SEED)
    stride = round(INTERVAL * BATCH / rate**2)
    records = round(DURATION / INTERVAL)
    assert np.isclose(stride * rate**2 / BATCH, INTERVAL)
    grams = []

    def record_gram():
        # Measure the current Gram, not a product of time-averaged weights.
        weights = first.detach().double()
        gram = einsum(weights, weights, "case run hidden row, case run hidden col -> case run row col")
        grams.append(gram.cpu().numpy())

    record_gram()
    for record in range(1, records + 1):
        for _ in range(stride):
            standard_inputs = torch.randn(RUNS, BATCH, 2, generator=random, device=DEVICE)
            standard_noise = torch.randn(RUNS, BATCH, 2, generator=random, device=DEVICE)
            inputs = einsum(standard_inputs, input_roots,
                            "run sample mode, case input mode -> case run sample input")
            noise = einsum(standard_noise, noise_roots,
                           "run sample mode, case output mode -> case run sample output")
            hidden = einsum(inputs, first,
                            "case run sample input, case run hidden input -> case run sample hidden")
            prediction = einsum(hidden, second,
                                "case run sample hidden, case run output hidden -> case run sample output")
            target = inputs + noise
            # Mean over samples; sum over coordinates and independent models.
            loss = (prediction - target).square().sum() / BATCH
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        record_gram()
        if record % 40 == 0:
            print(f"  rate={rate}, slow time={record * INTERVAL:g}/{DURATION:g}", flush=True)
    return dict(cases=cases, gram=np.array(grams), predicted_gram=predictions,
                tau=np.arange(records + 1) * INTERVAL, rate=rate, batch=BATCH,
                seed=SEED, steps=records * stride)


def generate():
    """Train all covariance pairs to slow time 20 at three learning rates."""
    for rate in RATES:
        save_data(CACHE / f"rate_{rate:g}.npz", **simulate(rate))


def errors(data):
    """Relative error of the mean Gram over the final quarter, for each case and run."""
    final_quarter = data["tau"] >= 0.75 * data["tau"][-1]
    mean_gram = data["gram"][final_quarter].mean(axis=0)            # (case, run, 2, 2)
    prediction = data["predicted_gram"][:, None]                    # (case, 1, 2, 2)
    return np.linalg.norm(mean_gram - prediction, axis=(-2, -1)) / np.linalg.norm(prediction, axis=(-2, -1))


def plot():
    percent_errors = {}
    for rate in RATES:
        with np.load(CACHE / f"rate_{rate:g}.npz") as data:
            cases = data["cases"]
            percent_errors[rate] = 100 * errors(data)               # (case, run)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 2.9), sharey=True)
    ymax = 0.
    for ax, condition in zip(axes, CONDITIONS, strict=True):
        style(ax)
        ax.set_title(rf"condition number $\kappa={condition:g}$")
        ax.set_xlabel(r"Learning rate $\eta$")
        ax.set_xticks([.005, .01, .02], ["0.005", "0.01", "0.02"])
        ax.set_xlim(.0035, .0215)
        for angle, color, marker in zip(ANGLES, (BLUE, ORANGE, GRAY), ("o", "s", "^"), strict=True):
            index = np.flatnonzero((cases[:, 0] == condition) & (cases[:, 1] == angle))[0]
            means = []
            for rate in RATES:
                samples = percent_errors[rate][index]
                mean, sem = samples.mean(), samples.std(ddof=1) / np.sqrt(len(samples))
                ax.errorbar(rate, mean, yerr=sem, marker=marker, color=color,
                            markersize=4, capsize=3, lw=1, zorder=3)
                means.append(mean)
                ymax = max(ymax, mean + sem)
            # Lines guide the eye between measured learning rates only.
            ax.plot(RATES, means, color=color, lw=.8)
            kind = "noncommuting" if angle == 45 else "commuting"
            ax.plot([], [], color=color, marker=marker, lw=.8, label=rf"${angle:g}^\circ$ ({kind})")
    axes[0].set_ylim(-.3, 1.12 * ymax)
    axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    axes[0].set_ylabel(r"Relative error in $\overline{G}$")
    axes[-1].legend(fontsize=8, title=r"rotation angle $\theta$",
                    title_fontsize=9, loc="center left",
                    bbox_to_anchor=(1.03, 0.5), frameon=True, framealpha=0.95)
    fig.tight_layout()
    save_figure(fig, FIGURES / "fig_noncommute_controlled")
