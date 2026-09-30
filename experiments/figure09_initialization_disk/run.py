import matplotlib.pyplot as plt
import numpy as np

from experiments.common import BLUE, ORANGE, folders, save_data, save_figure, style

CACHE, FIGURES = folders(__file__)
DATA = CACHE / "results.npz"
DIMENSION, SEED = 400, 5
SCALES = (0.5, 1., 1.5)


def cache_files():
    return (DATA,)


def generate():
    """Draw one matrix for each initialization scale; no training is needed."""
    random = np.random.default_rng(SEED)
    eigenvalues = []
    for scale in SCALES:
        gaussian = random.standard_normal((DIMENSION, DIMENSION))
        weight = np.eye(DIMENSION) + scale * gaussian / np.sqrt(DIMENSION)
        eigenvalues.append(np.linalg.eigvals(weight))
    save_data(DATA, eigenvalues=eigenvalues, scales=SCALES, dimension=DIMENSION, seed=SEED)


def plot():
    with np.load(DATA) as data:
        scales, eigenvalues = data["scales"], data["eigenvalues"]
    th = np.linspace(0, 2 * np.pi, 240)
    fig, axes = plt.subplots(1, len(scales), figsize=(8.6, 2.7), sharex=True, sharey=True)
    for ax, s, ev in zip(axes, scales, eigenvalues, strict=True):
        real = np.abs(ev.imag) < 1e-9
        ax.plot(1 + s * np.cos(th), s * np.sin(th), color="0.45", ls="--", lw=1.0)
        ax.axvline(0.0, color="0.35", ls=":", lw=1.0)
        ax.scatter(ev[~real].real, ev[~real].imag, s=5, color=BLUE,
                   alpha=0.45, lw=0)
        ax.scatter(ev[real].real, ev[real].imag, s=5, color=ORANGE, lw=0,
                   zorder=4)
        ax.set_aspect("equal")
        ax.set_title(rf"$s={s:g}$", fontsize=10)
        ax.set_xlabel(r"$\mathrm{Re}\,\lambda$")
        style(ax)
    axes[0].set_ylabel(r"$\mathrm{Im}\,\lambda$")
    axes[0].set_xlim(-1.0, 2.9)
    axes[0].set_ylim(-1.7, 1.7)
    fig.tight_layout()
    save_figure(fig, FIGURES / "fig_detspectrum")
