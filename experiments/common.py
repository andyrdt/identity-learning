from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
# Importing SciencePlots registers the "science" style used below.
import scienceplots  # noqa: F401

REPOSITORY = Path(__file__).resolve().parent.parent
BLUE, ORANGE, GRAY = "#0072B2", "#D55E00", "0.45"
plt.style.use(["science", "grid"])


def spectrum(dimension, anisotropic=True):
    """Label-noise variances, normalized to mean one."""
    values = np.logspace(-1, 1, dimension) if anisotropic else np.ones(dimension)
    return values / values.sum() * dimension


def log_steps(steps, count=120):
    sampled = np.geomspace(1, steps, count).round().astype(int)
    return np.unique(np.concatenate(([0], sampled, [steps])))


def smooth(values, window=9):
    """Centered moving average with reflected edges (Figure 1 only)."""
    half = window // 2
    padded = np.pad(values, half, mode="reflect")
    return np.convolve(padded, np.ones(window)/window, mode="valid")


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(top=False, right=False, which="both")
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.grid(True, which="major", color="#c4c4c4", lw=.5, alpha=.55)
    ax.grid(False, which="minor")


def band(ax, steps, values, color, *, ls="-", lw=1.0):
    """Mean across runs and their full range; values are (steps, runs)."""
    ax.plot(steps, values.mean(1), color=color, ls=ls, lw=lw)
    ax.fill_between(steps, values.min(1), values.max(1), color=color, alpha=.18, lw=0)


def folders(script):
    """An experiment's cache/ and figures/ directories, beside its script."""
    here = Path(script).resolve().parent
    return here / "cache", here / "figures"


def save_figure(fig, path):
    """Write path.pdf and path.png."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{extension}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path.relative_to(REPOSITORY)}.pdf and .png", flush=True)


def save_data(path, **arrays):
    """Save numerical arrays, refusing any non-finite result."""
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError(f"Nonfinite result for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    print(f"Saved {path.relative_to(REPOSITORY)}", flush=True)

