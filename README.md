# Learning the identity

Code for *Learning the identity: a case study of how SGD selects among
functional decompositions*.

Each directory in `experiments/` produces one or more of the paper's figures. It
contains the experiment (`run.py`), its results (`cache/`), and its
figures (`figures/`).

| Experiment                           | Paper output            |
| ------------------------------------ | ----------------------- |
| `figure01_scalar`                     | Figure 1                |
| `figures02_10_initialization`         | Figures 2 and 10        |
| `figure03_fourthroot`                 | Figure 3                |
| `figures04_06_depth`                  | Figures 4 and 6         |
| `figures05_07_table01_factorization`   | Figures 5 and 7, Table 1 |
| `figure08_covariance`                 | Figure 8                |
| `figure09_initialization_disk`        | Figure 9                |

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and
[LaTeX](https://github.com/garrettj403/SciencePlots/wiki/FAQ#installing-latex)
(used for figure text), then run `uv sync`.

## Usage

```bash
uv run python -m experiments.run           # draw all figures from cache/
uv run python -m experiments.run --fresh   # rerun all experiments first
```

Experiments without cached results are run automatically. Name experiments to
handle only those, e.g. `uv run python -m experiments.run figure03_fourthroot`.
