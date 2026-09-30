import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np
import torch

from einops import rearrange

from experiments.common import BLUE, folders, log_steps, save_data, save_figure, smooth, style
from experiments.models import LinearChain
from experiments.train import DEVICE, train

CACHE, FIGURES = folders(__file__)
DATA = CACHE / "results.npz"
STEPS, RATE, BATCH, SEED = 80_000, 0.01, 1, 11


def cache_files():
    return (DATA,)


def generate():
    """Four scalar starts with population loss 5, followed for 80,000 steps."""
    first = torch.tensor([2.5, 1.1, -1.15, -2.3], device=DEVICE)
    product = torch.tensor([-1., 3., 3., -1.], device=DEVICE)
    second = product / first
    weights = [rearrange(value - 1, "run -> run 1 1") for value in (first, second)]
    model = LinearChain(weights)
    steps, first_weights, second_weights = [], [], []

    def record(step, model):
        steps.append(step)
        first_weights.append(rearrange(model.weights[0], "run 1 1 -> run").cpu().numpy().copy())
        second_weights.append(rearrange(model.weights[1], "run 1 1 -> run").cpu().numpy().copy())

    train(model, np.ones(1), steps=STEPS, rate=RATE, batch=BATCH, seed=SEED,
          record=record, record_steps=log_steps(STEPS, 400))
    save_data(DATA, step=steps, first=first_weights, second=second_weights,
              steps=STEPS, rate=RATE, batch=BATCH, seed=SEED)


def plot():
    with np.load(DATA) as data:
        w1, w2, steps = data["first"], data["second"], data["step"]  # (T, 4), (T,)
    keep = steps >= 1

    fig = plt.figure(figsize=(7.4, 3.25))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.24,
                          top=0.82, bottom=0.13, left=0.07, right=0.985)
    axL = fig.add_subplot(gs[0, 0])
    axT = fig.add_subplot(gs[0, 1])

    # ---- left: the (w1t, w2t) plane
    lo, hi = -3.0, 3.0
    g = np.linspace(lo, hi, 500)
    M1, M2 = np.meshgrid(g, g)
    axL.contourf(M1, M2, np.log10((M1 * M2 - 1) ** 2 + 1),
                 levels=np.linspace(0, 2.1, 9), cmap="Greys", vmin=0, vmax=6,
                 zorder=0)
    for br in (np.linspace(1 / hi, hi, 300), np.linspace(lo, -1 / hi, 300)):
        axL.plot(br, 1 / br, color="0.6", lw=0.9, ls=":", zorder=2)
    norm = LogNorm(vmin=1, vmax=steps.max())
    cmap = plt.get_cmap("viridis")
    for k in range(w1.shape[1]):
        x, y = smooth(1 + w1[:, k]), smooth(1 + w2[:, k])
        pts = np.stack([x, y], axis=1)[:, None, :]
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=1.6,
                            zorder=3, capstyle="round")
        lc.set_array(np.maximum(steps[:-1], 1))
        axL.add_collection(lc)
        # Mark the true recorded initialization; the displayed path itself is
        # lightly smoothed and its first smoothed point need not equal it.
        axL.plot(1 + w1[0, k], 1 + w2[0, k], "o", ms=3.4,
                 color=cmap(norm(1)), zorder=4)
    for p in ((1, 1), (-1, -1)):
        axL.plot(*p, "*", ms=10, mfc=cmap(1.0), mec="k", mew=0.7, zorder=5)
    axL.set_xlim(lo, hi)
    axL.set_ylim(lo, hi)
    axL.set_xticks([-2, -1, 0, 1, 2])
    axL.set_yticks([-2, -1, 0, 1, 2])
    axL.set_aspect("equal")
    axL.set_xlabel(r"$\widetilde w_1$")
    axL.set_ylabel(r"$\widetilde w_2$")
    style(axL)
    axL.grid(False)

    # ---- right: L and E||grad l||^2 at the measured weights. Off the curve,
    # E||grad l||^2 = 4 (w1t^2 + w2t^2)(3 (1 - w2t w1t)^2 + 1); on it, this is
    # the Section 2.2 expression, with minimum 8.
    pop_loss = lambda a, b: (a * b - 1) ** 2 + 1
    grad_sq = lambda a, b: 4 * (a * a + b * b) * (3 * (1 - a * b) ** 2 + 1)
    for k in range(w1.shape[1]):
        a_r, b_r = 1 + w1[:, k], 1 + w2[:, k]
        for series, col in ((pop_loss(a_r, b_r), "0.45"),
                            (grad_sq(a_r, b_r), BLUE)):
            axT.plot(steps[keep], series[keep], color=col, lw=0.5, alpha=0.35)
            axT.plot(steps[keep], smooth(series)[keep], color=col, lw=1.0)
    axT.axhline(1, color="0.45", ls=":", lw=0.8, zorder=1)
    axT.axhline(8, color=BLUE, ls=":", lw=0.8, zorder=1)
    axT.set_yscale("log")
    axT.set_ylim(0.7, 2000)
    handles = [Line2D([0], [0], color="0.45", lw=1.2),
               Line2D([0], [0], color=BLUE, lw=1.2)]
    labels = [r"$\mathcal L$ (population loss)",
              r"$\mathbb{E}\|\nabla\ell\|^2$ (gradient norm)"]
    axT.legend(handles, labels, fontsize=7, framealpha=0.9,
               loc="upper right", handlelength=1.4)
    axT.set_xscale("log")
    axT.set_xlabel("training step")
    style(axT)

    # Align the square phase portrait with the full-height time-series panel,
    # with a compact horizontal training-step colorbar above the phase portrait.
    bbT = axT.get_position()
    bottom, top = bbT.y0, bbT.y1
    h = top - bottom
    fw, fh = fig.get_size_inches()
    w = h * fh / fw
    x0 = 0.07
    axL.set_position([x0, bottom, w, h])
    right_x0 = x0 + w + 0.065
    axT.set_position([right_x0, bottom, 0.985 - right_x0, h])
    cax = fig.add_axes([x0 + 0.24 * w, top + 0.05, 0.72 * w, 0.028])
    cb = fig.colorbar(lc, cax=cax, orientation="horizontal")
    fig.text(x0 + 0.20 * w, top + 0.064, "training step", fontsize=7.5,
             ha="right", va="center")
    cb.ax.tick_params(axis="x", which="major", labelsize=7, length=2.0,
                      width=0.5, color="0.35", pad=1.5, direction="out")
    save_figure(fig, FIGURES / "fig_oned")
