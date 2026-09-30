import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
import numpy as np
import torch

from experiments.common import BLUE, folders, save_data, save_figure, spectrum, style
from experiments.theory import decay_spectrum
from experiments.models import LinearChain
from experiments.train import gaussian_weights, train

CACHE, FIGURES = folders(__file__)
DIMENSION, RUNS, STEPS = 16, 5, 1_500_000
RATE, BATCH = 0.005, 16
DECAYS = (0., 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1.)
INIT_SEED, DATA_SEED = 3, 42


def cache(decay):
    return CACHE / f"decay_{decay:g}.npz"


def cache_files():
    return tuple(cache(decay) for decay in DECAYS)


def generate():
    """Measure the learned spectrum at seven weight-decay strengths."""
    initial_weights = gaussian_weights(RUNS, DIMENSION, 0.3, seed=INIT_SEED, depth=2)
    variance = spectrum(DIMENSION)
    for decay in DECAYS:
        print(f"weight decay = {decay}", flush=True)
        model = LinearChain(initial_weights)
        average = train(model, variance, steps=STEPS, rate=RATE, batch=BATCH,
                        seed=DATA_SEED, decay=decay)
        identity = torch.eye(DIMENSION, device=average.weights[0].device)
        effective_weight = identity + average.weights[0]
        singular_values = torch.linalg.svdvals(effective_weight).cpu().numpy()
        save_data(cache(decay), variance=variance, singular_values=singular_values,
                  steps=STEPS, rate=RATE, batch=BATCH, decay=decay,
                  init_seed=INIT_SEED, data_seed=DATA_SEED)


def plot():
    """Main-text figure: isolate the fourth-root and weight-decay claims."""
    fig, (ax_root, ax_wd) = plt.subplots(
        1, 2, figsize=(7.5, 2.9), sharex=True, sharey=True
    )
    with np.load(cache(0)) as data:
        v = data["variance"]
        rate, batch = float(data["rate"]), int(data["batch"])
        zero_decay_runs = data["singular_values"]
    gammas = np.asarray(DECAYS)
    v_sorted = np.sort(v)

    # Left: the no-weight-decay prediction is exactly the fourth-root law.
    s_runs = np.sort(zero_decay_runs, axis=1)
    ax_root.errorbar(
        v_sorted, s_runs.mean(0), yerr=s_runs.std(0), fmt="o", ms=3.2,
        color=BLUE, ecolor=BLUE, elinewidth=0.65, capsize=0,
        label="empirical (SGD)", zorder=3,
    )
    ax_root.plot(
        v_sorted, v_sorted ** 0.25, "-", color="k", lw=1.15,
        label=r"theory: $s_i=\lambda_i^{1/4}$", zorder=2,
    )
    ax_root.set_title(
        r"without weight decay ($\gamma=0$)", fontsize=9.0, pad=4
    )
    ax_root.legend(
        handles=[
            Line2D([], [], marker="o", color=BLUE, ls="none", ms=4.0,
                   label="empirical (SGD)"),
            Line2D([], [], color="k", lw=1.15,
                   label=r"theory: $s_i=\lambda_i^{1/4}$"),
        ],
        loc="lower right", fontsize=7.2, framealpha=0.95,
    )

    # Right: positive weight decay balances the fourth-root preference against
    # the identity solution.  Color identifies gamma; circles and lines retain
    # the same SGD/theory encoding as in the left panel.
    positive_gammas = gammas[gammas > 0]
    gamma_norm = LogNorm(
        vmin=float(positive_gammas.min()),
        vmax=float(positive_gammas.max()),
    )
    gamma_cmap = LinearSegmentedColormap.from_list(
        "weight_decay", plt.cm.plasma(np.linspace(0.08, 0.78, 256))
    )
    for g in positive_gammas:
        c = gamma_cmap(gamma_norm(g))
        with np.load(cache(g)) as data:
            s_runs = np.sort(data["singular_values"], axis=1)
        s_sgd = s_runs.mean(0)
        s_std = s_runs.std(0)
        s_theory = np.sort(decay_spectrum(v, rate, batch, g))
        ax_wd.errorbar(
            v_sorted, s_sgd, yerr=s_std, fmt="o", ms=3.2,
            color=c, ecolor=c, elinewidth=0.65, capsize=0,
            zorder=3,
        )
        ax_wd.plot(
            v_sorted, s_theory, "-", color=c, lw=1.0, alpha=0.9,
            zorder=2,
        )
    ax_wd.set_title(
        r"with weight decay ($\gamma>0$)", fontsize=9.0, pad=4
    )
    gamma_map = plt.cm.ScalarMappable(norm=gamma_norm, cmap=gamma_cmap)
    cbar = fig.colorbar(
        gamma_map, ax=ax_wd, ticks=positive_gammas,
        fraction=0.055, pad=0.025, aspect=22,
    )
    cbar.set_ticklabels([
        rf"$10^{{{int(round(np.log10(g)))}}}$"
        for g in positive_gammas
    ])
    cbar.minorticks_off()
    cbar.ax.yaxis.set_ticks_position("right")
    cbar.ax.tick_params(
        axis="y", which="major", left=False, right=True,
        labelleft=False, labelright=True, direction="inout",
        labelsize=7.2, length=2.5, width=0.7, pad=2,
    )
    cbar.outline.set_linewidth(0.7)
    cbar.set_label(
        r"weight decay strength $\gamma$", fontsize=7.5, labelpad=4
    )

    for ax in (ax_root, ax_wd):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"ordered noise variance $\lambda_i$")
        style(ax)
        ax.grid(True, which="minor", color="#d0d0d0", lw=0.4, alpha=0.55)
    ax_root.set_ylabel(
        r"ordered singular value $s_i$ of $\widetilde W_1$"
    )
    fig.tight_layout(w_pad=1.2)
    save_figure(fig, FIGURES / "fig_fourthroot")
