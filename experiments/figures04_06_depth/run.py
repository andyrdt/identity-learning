import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter
from matplotlib.lines import Line2D
import numpy as np
import torch

from experiments.common import BLUE, ORANGE, GRAY, folders, save_data, save_figure, spectrum, style
from experiments.theory import chain_scales
from experiments.models import LinearChain
from experiments.train import gaussian_weights, train

CACHE, FIGURES = folders(__file__)
RUNS, STEPS, RATE, BATCH = 6, 1_500_000, 0.02, 64
# depth, dimension, initialization seed
SETTINGS = ((3, 16, 80), (4, 8, 77), (5, 8, 82))


def cache_files():
    return tuple(CACHE / f"depth{depth}.npz" for depth, _, _ in SETTINGS)


def generate():
    """Measure each layer's spectrum in three-, four-, and five-layer networks."""
    for depth, dimension, seed in SETTINGS:
        print(f"depth = {depth}", flush=True)
        weights = gaussian_weights(RUNS, dimension, 0.2, seed=seed, depth=depth)
        model = LinearChain(weights)
        variance = spectrum(dimension)
        average = train(model, variance, steps=STEPS, rate=RATE, batch=BATCH, seed=900 + depth)
        identity = torch.eye(dimension, device=average.weights[0].device)
        singular_values = []
        for weight in average.weights:
            singular_values.append(torch.linalg.svdvals(identity + weight).cpu().numpy())
        save_data(CACHE / f"depth{depth}.npz", variance=variance,
                  singular_values=singular_values, depth=depth, steps=STEPS,
                  rate=RATE, batch=BATCH, init_seed=seed, data_seed=900 + depth)


def plot_depth_three():
    with np.load(CACHE / "depth3.npz") as data:
        v = np.sort(np.asarray(data["variance"], float))
        sv = data["singular_values"]
    D = 3
    c, a = chain_scales(v, D)
    vv = np.geomspace(v.min() * 0.85, v.max() * 1.18, 40)
    fig, ax = plt.subplots(figsize=(5.2, 2.3))
    ax.plot(vv, a * vv ** 0.25, ls="--", lw=1.0, color=BLUE)
    ax.plot(vv, 0 * vv + c, ls="--", lw=1.0, color=GRAY)
    ax.plot(vv, a * vv ** -0.25, ls="--", lw=1.0, color=ORANGE)
    first_runs = np.sort(sv[0], axis=1)
    interior_runs = np.sort(sv[1], axis=1)
    last_runs = np.sort(sv[-1], axis=1)[:, ::-1]
    for runs, color in ((first_runs, BLUE), (interior_runs, GRAY),
                        (last_runs, ORANGE)):
        ax.errorbar(v, runs.mean(0), yerr=runs.std(0), fmt="o", ms=2.8,
                    color=color, ecolor=color, elinewidth=0.55, capsize=1.2,
                    capthick=0.55)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"ordered noise variance $\lambda_i$")
    ax.set_ylabel(r"ordered singular value $s_i(\widetilde W_\ell)$")
    style(ax)
    ax.yaxis.set_major_locator(FixedLocator([0.5, 1.0, 2.0]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.grid(True, which="minor", color="#d0d0d0", lw=0.4, alpha=0.55)

    layer_handles = [
        Line2D([0], [0], marker="o", color=color, lw=0, ms=3.5,
               label=rf"$\ell={ell}$")
        for ell, color in ((1, BLUE), (2, GRAY), (3, ORANGE))
    ]
    layer_legend = ax.legend(handles=layer_handles, title="Layer", fontsize=7.5,
                             title_fontsize=7.5, loc="upper left",
                             bbox_to_anchor=(1.01, 1.01), framealpha=0.95,
                             borderpad=0.35, labelspacing=0.25,
                             handletextpad=0.5, borderaxespad=0.2)
    ax.add_artist(layer_legend)
    style_handles = [
        Line2D([0], [0], marker="o", color="0.35", lw=0, ms=3.5,
               label="empirical"),
        Line2D([0], [0], color="0.35", ls="--", lw=1.0, label="theory"),
    ]
    ax.legend(handles=style_handles, fontsize=7.5, loc="upper left",
              bbox_to_anchor=(1.01, 0.64), framealpha=0.95,
              borderpad=0.35, labelspacing=0.25,
              handletextpad=0.5, borderaxespad=0.2)
    fig.subplots_adjust(left=0.13, right=0.75, bottom=0.24, top=0.96)
    save_figure(fig, FIGURES / "fig_deep")


def plot_deeper_chains():
    """Appendix check: boundary laws and interior concentration at D=4,5."""
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.9),
                             gridspec_kw={"width_ratios": [1, 1, 0.9]})
    boundary_axes, ax_interior = axes[:2], axes[2]
    boundary_axes[1].sharey(boundary_axes[0])
    rng = np.random.default_rng(4)
    for ax_boundary, D in zip(boundary_axes, (4, 5), strict=True):
        with np.load(CACHE / f"depth{D}.npz") as data:
            v = np.sort(np.asarray(data["variance"], float))
            sv = data["singular_values"]
        vv = np.geomspace(v.min() * 0.85, v.max() * 1.18, 60)
        c, a = chain_scales(v, D)
        first_runs = np.sort(sv[0], axis=1) / a
        last_runs = np.sort(sv[-1], axis=1)[:, ::-1] / a
        for runs, color, marker, label, exponent in (
            (first_runs, BLUE, "o", "first", 0.25),
            (last_runs, ORANGE, "s", "last", -0.25),
        ):
            ax_boundary.plot(vv, vv ** exponent, ls="--", lw=1.0, color=color)
            ax_boundary.errorbar(v, runs.mean(0), yerr=runs.std(0),
                                 ls="none", marker=marker, ms=3.5,
                                 color=color, ecolor=color, elinewidth=0.5,
                                 capsize=1.1, label=label)
        ax_boundary.set(xscale="log", yscale="log",
                        xlabel=r"ordered noise variance $\lambda_i$")
        ax_boundary.set_title(rf"boundary layers, $D={D}$", fontsize=9)
        ax_boundary.legend(title="Boundary layer", title_fontsize=7, fontsize=7,
                           loc="upper right", framealpha=0.95)
        ax_boundary.yaxis.set_major_locator(FixedLocator([0.5, 1.0, 2.0]))
        ax_boundary.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax_boundary.yaxis.set_minor_formatter(NullFormatter())
        style(ax_boundary)

        interior = sv[1:-1].reshape(-1) / c
        x = D + rng.uniform(-0.10, 0.10, size=interior.size)
        ax_interior.scatter(x, interior, s=7, color=GRAY, alpha=0.28,
                            edgecolors="none")
        ax_interior.boxplot(
            [interior], positions=[D], widths=0.34, patch_artist=True,
            showfliers=False,
            medianprops={"color": "0.2", "linewidth": 1.0},
            boxprops={"facecolor": "white", "edgecolor": "0.35", "linewidth": 0.8},
            whiskerprops={"color": "0.4", "linewidth": 0.8},
            capprops={"color": "0.4", "linewidth": 0.8})
    boundary_axes[0].set_ylabel(r"boundary singular value $/\,a_D$")
    boundary_axes[1].tick_params(axis="y", labelleft=False)
    ax_interior.axhline(1, ls="--", lw=1.0, color="0.25", label="theory")
    ax_interior.set_xticks([4, 5], [r"$D=4$", r"$D=5$"])
    ax_interior.set(xlim=(3.55, 5.45), ylim=(0.9, 1.1),
                    ylabel=r"interior singular value $/\,c_D$")
    ax_interior.set_yticks([0.9, 0.95, 1, 1.05, 1.1])
    ax_interior.set_title("interior layers", fontsize=9)
    ax_interior.legend(fontsize=7, framealpha=0.95, loc="upper left")
    style(ax_interior)
    fig.tight_layout(w_pad=1.4)
    save_figure(fig, FIGURES / "fig_deep_extra")


def plot():
    plot_depth_three()
    plot_deeper_chains()
