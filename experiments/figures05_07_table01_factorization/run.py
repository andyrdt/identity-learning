import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from experiments.common import BLUE, GRAY, band, folders, log_steps, save_data, save_figure, spectrum, style
from experiments.metrics import functional, residual_norm
from experiments.models import ResidualBlocks
from experiments.train import matched_weights, train

CACHE, FIGURES = folders(__file__)
DIMENSION, RUNS, STEPS = 16, 6, 800_000
RATE, BATCH = 0.02, 64
INIT_SEED, DATA_SEED, EVALUATION_SEED = 12345, 99991, 20240101
TRAJECTORY_SAMPLES, TABLE_SAMPLES = 2048, 8192


def cache(parametrization, noise, activation):
    return CACHE / f"{parametrization}_{noise}_{activation}.npz"


def cache_files():
    return tuple(cache(param, noise, activation)
                 for param in ("direct", "factored")
                 for noise in ("iso", "aniso")
                 for activation in ("linear", "relu"))


def generate():
    """Compare direct and factored blocks under linear and ReLU activations."""
    initial_weights = matched_weights(RUNS, DIMENSION, seed=INIT_SEED)
    for parametrization in ("direct", "factored"):
        for noise in ("iso", "aniso"):
            for activation in ("linear", "relu"):
                print(f"{parametrization}, {noise}, {activation}", flush=True)
                generate_setting(initial_weights, parametrization, noise, activation)


def generate_setting(initial_weights, parametrization, noise, activation):
    """Train one setting, tracing current weights and evaluating the averaged model."""
    model = ResidualBlocks(initial_weights, factored=parametrization == "factored",
                           activation=activation)
    steps, norms, branches, errors = [], [], [], []

    def record(step, model):
        branch, error = functional(model, samples=TRAJECTORY_SAMPLES, seed=EVALUATION_SEED)
        steps.append(step)
        norms.append(residual_norm(model))
        branches.append(branch)
        errors.append(error)

    variance = spectrum(DIMENSION, anisotropic=noise == "aniso")
    average = train(model, variance, steps=STEPS, rate=RATE, batch=BATCH,
                    seed=DATA_SEED, record=record, record_steps=log_steps(STEPS, 140))
    table_branch, table_error = functional(average, samples=TABLE_SAMPLES, seed=EVALUATION_SEED)
    save_data(cache(parametrization, noise, activation),
              step=steps, residual_norm=norms, branch=branches, function_error=errors,
              table_branch=table_branch, table_error=table_error,
              steps=STEPS, rate=RATE, batch=BATCH, init_seed=INIT_SEED,
              data_seed=DATA_SEED, evaluation_seed=EVALUATION_SEED)


def plot_linear():
    """Compare both noise settings on one set of axes using cached linear runs."""
    fig, ax = plt.subplots(figsize=(6.0, 2.4))
    largest_residual = 0.0
    for noise, ls in (("iso", "-"), ("aniso", "--")):
        for param, color in (("direct", GRAY), ("factored", BLUE)):
            with np.load(cache(param, noise, "linear")) as data:
                keep = data["step"] >= 1
                band(ax, data["step"][keep], data["residual_norm"][keep], color, ls=ls)
                largest_residual = max(largest_residual, float(data["residual_norm"][keep].max()))
    upper = np.ceil(1.12 * largest_residual * 2) / 2
    ax.set(xscale="log", ylim=(-0.02 * upper, upper), xlabel="training step",
           ylabel=r"residual map norm $\sum_\ell\|R_\ell\|_F$")
    style(ax)
    fig.legend(handles=[
        Line2D([], [], color=GRAY, label=r"$W_\ell$ (direct)"),
        Line2D([], [], color=BLUE, label=r"$A_\ell B_\ell$ (factored)"),
    ], title="Weight parameterization", title_fontsize=7.5, fontsize=7.5,
        loc="upper left", bbox_to_anchor=(1.01, 1.01), bbox_transform=ax.transAxes,
        alignment="left", framealpha=0.95, borderpad=0.35, labelspacing=0.25,
        handletextpad=0.5, borderaxespad=0.2)
    ax.legend(handles=[
        Line2D([], [], color="0.2", ls="-", label="isotropic"),
        Line2D([], [], color="0.2", ls="--", label="anisotropic"),
    ], title="Label noise distribution", title_fontsize=7.5, fontsize=7.5,
        loc="upper left", bbox_to_anchor=(1.01, 0.71),
        alignment="left", framealpha=0.95, borderpad=0.35, labelspacing=0.25,
        handletextpad=0.5, borderaxespad=0.2)
    fig.subplots_adjust(left=0.13, right=0.75, bottom=0.28, top=0.96)
    save_figure(fig, FIGURES / "fig_flip")


def plot_relu():
    """Actual ReLU branch outputs and clean function errors from cached iterates."""
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6), sharex=True, sharey="row")
    for col, noise in enumerate(("iso", "aniso")):
        for param, color in (("direct", GRAY), ("factored", BLUE)):
            with np.load(cache(param, noise, "relu")) as data:
                keep = data["step"] >= 1
                steps = data["step"][keep]
                for row, key in enumerate(("branch", "function_error")):
                    values = data[key][keep]
                    ax = axes[row, col]
                    ax.plot(steps, values.mean(1), color=color, lw=1.5, label=param)
                    ax.fill_between(steps, values.min(1), values.max(1),
                                    color=color, alpha=0.15, lw=0)
        axes[0, col].set_title(f"{'isotropic' if noise == 'iso' else 'anisotropic'} noise")
        axes[1, col].set_xlabel("training step")
    axes[0, 0].set_ylabel("relative branch RMS")
    axes[1, 0].set_ylabel("relative function error")
    for ax in axes.flat:
        ax.set(xscale="log", yscale="log")
        style(ax)
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    save_figure(fig, FIGURES / "fig_relu_functional")


def write_table():
    lines = [r"\begin{tabular}{lllcc}", r"\toprule",
             r"parametrization & activation & noise & relative branch RMS & relative function error \\",
             r"\midrule"]
    for parametrization in ("direct", "factored"):
        if parametrization == "factored":
            lines.append(r"\midrule")
        for noise in ("iso", "aniso"):
            for activation in ("linear", "relu"):
                with np.load(cache(parametrization, noise, activation)) as data:
                    branch = data["table_branch"].mean()
                    error = data["table_error"].mean()
                label = "ReLU" if activation == "relu" else "linear"
                lines.append(f"{parametrization} & {label} & {noise} & ${branch:.4f}$ & ${error:.4f}$ " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    FIGURES.mkdir(parents=True, exist_ok=True)
    (FIGURES / "table_factored_matched.tex").write_text("\n".join(lines) + "\n")


def plot():
    plot_linear()
    plot_relu()
    write_table()
