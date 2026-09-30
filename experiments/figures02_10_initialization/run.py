import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

from einops import rearrange

from experiments.common import BLUE, ORANGE, GRAY, folders, save_data, save_figure, style
from experiments.models import LinearChain
from experiments.train import gaussian_weights, train

CACHE, FIGURES = folders(__file__)
DATA = CACHE / "results.npz"
DIMENSION, RUNS, STEPS = 16, 16, 2_000_000
RATE, BATCH = 0.02, 64
VARIANCES = (0.1, 1., 2., 4.)
INIT_SEED, DATA_SEED = 83, 1900


def cache_files():
    return (DATA,)


def generate():
    """Four initialization scales, with and without weight decay."""
    initial_weights = []
    for layer in range(2):
        groups = []
        for index, variance in enumerate(VARIANCES):
            weights = gaussian_weights(RUNS, DIMENSION, variance**0.5,
                                       seed=INIT_SEED + 4 * layer + index, depth=1)
            groups.append(weights[0])
        initial_weights.append(rearrange(torch.stack(groups),
                                       "scale run row col -> (scale run) row col"))

    results = dict(variance_multipliers=VARIANCES, steps=STEPS, rate=RATE,
                   batch=BATCH, init_seed=INIT_SEED, data_seed=DATA_SEED)
    for name, decay in (("no_decay", 0.), ("decay", 0.01)):
        print(f"weight decay = {decay}", flush=True)
        model = LinearChain(initial_weights)
        average = train(model, np.ones(DIMENSION), steps=STEPS, rate=RATE,
                        batch=BATCH, seed=DATA_SEED, decay=decay)
        identity = torch.eye(DIMENSION, device=average.weights[0].device)
        effective_weight = identity + average.weights[0]
        results[name] = rearrange(effective_weight, "(scale run) row col -> scale run row col",
                                 scale=len(VARIANCES)).cpu().numpy()
    save_data(DATA, **results)


def run_metrics(matrices):
    """Return Frobenius distances to O(d) and to the identity."""
    d = matrices.shape[-1]
    eye = np.eye(d)
    # Orthogonal Procrustes: dist_F(Q, O(d))^2
    # = sum_i (sigma_i(Q) - 1)^2.
    singular_values = np.linalg.svd(matrices, compute_uv=False)
    orthogonality = np.linalg.norm(singular_values - 1.0, axis=-1)
    identity = np.linalg.norm(matrices - eye, axis=(-2, -1))
    return orthogonality, identity


def median_mark(ax, x, values, color):
    """Draw a short median rule over a jittered seed cloud."""
    ax.plot([x - 0.105, x + 0.105], [np.median(values)] * 2,
            color=color, lw=1.6, solid_capstyle="round", zorder=4)


def plot_distances_and_eigenvalues():
    with np.load(DATA) as dat:
        index = np.flatnonzero(np.isclose(dat["variance_multipliers"], 1.0))[0]
        no_wd = dat["no_decay"][index].astype(float)
        wd = dat["decay"][index].astype(float)
    g0_metrics = run_metrics(no_wd)
    wd_metrics = run_metrics(wd)

    fig = plt.figure(figsize=(8.0, 2.95))
    outer = fig.add_gridspec(
        1, 2, width_ratios=(1.15, 2.0),
        left=0.075, right=0.985, bottom=0.25, top=0.90, wspace=0.32,
    )
    ax_summary = fig.add_subplot(outer[0, 0])
    eig_grid = outer[0, 1].subgridspec(1, 2, wspace=0.22)
    ax_g0 = fig.add_subplot(eig_grid[0, 0])
    ax_wd = fig.add_subplot(eig_grid[0, 1])

    # (a) Direct tests of orthogonality and proximity to the identity.
    rng = np.random.default_rng(17)
    centers = np.array([0.0, 1.0])
    offset = 0.14
    jitter = 0.026
    for values, dx, color in (
        (g0_metrics, -offset, BLUE),
        (wd_metrics, offset, ORANGE),
    ):
        for x, y in zip(centers + dx, values, strict=True):
            xx = x + rng.normal(0.0, jitter, len(y))
            ax_summary.scatter(
                xx, y, s=15, marker="o", color=color, alpha=0.58,
                edgecolors="none", zorder=3,
            )
            median_mark(ax_summary, x, y, color)

    ax_summary.set_yscale("log")
    ax_summary.set_ylim(1.0e-2, 50.0)
    ax_summary.set_xlim(-0.43, 1.43)
    ax_summary.set_xticks(centers)
    ax_summary.set_xticklabels((
        "to nearest\northogonal matrix",
        "to\nidentity matrix",
    ))
    ax_summary.tick_params(axis="x", which="both", length=0)
    ax_summary.set_ylabel("Frobenius distance")
    style(ax_summary)
    ax_summary.grid(False, axis="x")
    ax_summary.legend(
        handles=[
            Line2D([], [], ls="", marker="o", ms=4.5, color=BLUE,
                   label=r"$\gamma=0$"),
            Line2D([], [], ls="", marker="o", ms=4.5, color=ORANGE,
                   label=r"$\gamma=10^{-2}$"),
        ],
        title="Weight decay strength",
        title_fontsize=7.2,
        loc="upper left", fontsize=7.2, framealpha=0.94,
        alignment="left",
        borderpad=0.35, handletextpad=0.35, labelspacing=0.45,
    )

    # (b) Orientation of the learned effective first layer, split by gamma.
    theta = np.linspace(0.0, 2.0 * np.pi, 500)
    for ax, matrices, color, title in (
        (ax_g0, no_wd, BLUE, r"$\gamma=0$"),
        (ax_wd, wd, ORANGE, r"$\gamma=10^{-2}$"),
    ):
        eigenvalues = np.linalg.eigvals(matrices).ravel()
        ax.plot(
            np.cos(theta), np.sin(theta), color=GRAY, ls="--",
            lw=0.85, zorder=1,
        )
        ax.scatter(
            eigenvalues.real, eigenvalues.imag, s=10.5, color=color,
            alpha=0.50, linewidths=0, zorder=2,
        )
        ax.set_xlim(-1.42, 1.42)
        ax.set_ylim(-1.42, 1.42)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$\mathrm{Re}\,\lambda$")
        ax.set_title(title, fontsize=9.0, pad=4)
        style(ax)
    ax_g0.set_ylabel(r"$\mathrm{Im}\,\lambda$")
    ax_wd.set_yticklabels([])

    for ax, label in ((ax_summary, "a"), (ax_g0, "b")):
        ax.text(
            -0.16, 1.05, rf"\textbf{{({label})}}", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=9.0,
        )

    save_figure(fig, FIGURES / "fig_eig")


def plot_row(dat, arm, gamma_label, filename):
    multipliers = dat["variance_multipliers"]
    matrices = dat[arm]
    theta = np.linspace(0, 2 * np.pi, 400)
    fig, axes = plt.subplots(1, len(multipliers), figsize=(8.8, 2.45),
                             sharex=True, sharey=True)
    for ax, value, runs in zip(axes, multipliers, matrices, strict=True):
        eigenvalues = np.linalg.eigvals(runs).reshape(-1)
        ax.plot(np.cos(theta), np.sin(theta), color=GRAY, ls="--",
                lw=0.9, zorder=1)
        ax.scatter(eigenvalues.real, eigenvalues.imag, s=6, color=BLUE,
                   alpha=0.4, lw=0, zorder=2)
        ax.set_aspect("equal")
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        scale = np.sqrt(value)
        scale_label = f"{scale:g}" if np.isclose(scale, round(scale)) else rf"\sqrt{{{value:g}}}"
        ax.set_title(rf"$s={scale_label}$", fontsize=9.5)
        ax.set_xlabel(r"$\mathrm{Re}\,\lambda$")
        style(ax)
    axes[0].set_ylabel(r"$\mathrm{Im}\,\lambda$")
    fig.suptitle(gamma_label, fontsize=10.5, y=1.01)
    fig.tight_layout()
    save_figure(fig, FIGURES / filename)


def plot():
    plot_distances_and_eigenvalues()
    with np.load(DATA) as data:
        plot_row(data, "no_decay", r"$\gamma=0$", "fig_init_eigs_g0")
        plot_row(data, "decay", r"$\gamma=10^{-2}$", "fig_init_eigs_wd")
