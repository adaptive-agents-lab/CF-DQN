#!/usr/bin/env python3
"""
Fetch Weights & Biases runs by tags and plot charts/episodic_return (mean ± std per algorithm).

Requires: pip install wandb matplotlib numpy
Login once: wandb login

Usage (import and call with your arguments):

    from plot_wandb_episodic_return import plot_episodic_return

    plot_episodic_return(
        project="Deep-CVI-Experiments",
        entity=None,
        required_tag=[],
        algo_tags=["MoG", "FFT", "dqn"],
        out="figures/breakout_episodic_return.png",
        env_name="Breakout MinAtar",
    )

Each run should carry exactly one of the ``algo_tags`` on W&B.
"""
from __future__ import annotations

import os
from collections import defaultdict

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit("Please install matplotlib: pip install matplotlib") from e

try:
    import wandb
except ImportError as e:
    raise SystemExit("Please install wandb: pip install wandb") from e


def _wandb_path(entity: str | None, project: str) -> str:
    if entity:
        return f"{entity}/{project}"
    return project


def _run_matches_required(run, required: list[str]) -> bool:
    tags = set(run.tags or [])
    return all(t in tags for t in required)


def _algo_group(run, algo_tags: list[str]) -> str | None:
    tags = set(run.tags or [])
    found = [a for a in algo_tags if a in tags]
    if len(found) == 1:
        return found[0]
    if len(found) == 0:
        return None
    raise ValueError(f"Run {run.id} has multiple algo tags {found}; use unique algo tags per run.")


def _load_series(run, metric: str, step_key: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Returns (steps, values) sorted by step; drops NaNs."""
    df = None
    try:
        df = run.history(keys=[metric, step_key], pandas=True)
    except Exception:
        pass
    if df is None or getattr(df, "empty", True):
        try:
            df = run.history(pandas=True)
        except Exception:
            return None
    if df is None or df.empty or step_key not in df.columns or metric not in df.columns:
        return None
    work = df[[step_key, metric]].dropna()
    if work.empty:
        return None
    steps = work[step_key].to_numpy(dtype=np.float64)
    vals = work[metric].to_numpy(dtype=np.float64)
    order = np.argsort(steps)
    return steps[order], vals[order]


def _interp_on_grid(
    curves: list[tuple[np.ndarray, np.ndarray]], grid: np.ndarray
) -> np.ndarray:
    """Each curve (s, v); returns array shape (n_curves, len(grid)) with linear interp and nan outside."""
    out = np.full((len(curves), len(grid)), np.nan, dtype=np.float64)
    for i, (s, v) in enumerate(curves):
        if len(s) == 0:
            continue
        mask = np.isfinite(s) & np.isfinite(v)
        s, v = s[mask], v[mask]
        if len(s) < 2:
            continue
        out[i] = np.interp(grid, s, v, left=np.nan, right=np.nan)
    return out


def _smooth_1d(y: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average (edge-padded). ``window`` is coerced to an odd length ≤ ``len(y)``."""
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if window <= 1 or n < 2:
        return y
    w = max(3, int(window) | 1)
    if w > n:
        w = n if n % 2 == 1 else n - 1
    if w < 3:
        return y
    # Fill NaNs for convolution, then restore NaN where entire window was NaN (optional skip)
    bad = ~np.isfinite(y)
    if np.any(bad) and np.any(~bad):
        idx = np.arange(n)
        y = y.copy()
        y[bad] = np.interp(idx[bad], idx[~bad], y[~bad])
    pad = w // 2
    yp = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=np.float64) / w
    return np.convolve(yp, kernel, mode="valid")


def _pretty_metric_label(metric: str) -> str:
    """Last path segment, underscores → spaces, title case (e.g. charts/episodic_return → Episodic Return)."""
    tail = metric.split("/")[-1]
    return tail.replace("_", " ").strip().title()


def plot_episodic_return(
    *,
    project: str,
    entity: str | None = None,
    required_tag: list[str] | None = None,
    algo_tags: list[str] | None = None,
    metric: str = "charts/episodic_return",
    step_metric: str = "global_step",
    out: str = "wandb_episodic_return.png",
    grid_points: int = 800,
    max_runs: int = 500,
    smooth_window: int = 41,
    env_name: str = "Breakout MinAtar",
) -> None:
    """Fetch W&B runs, group by algo tags, plot mean ± std of ``metric`` vs ``step_metric``.

    ``smooth_window``: centered moving-average length on the interpolation grid (odd; set 0 to disable).
    Larger ``grid_points`` + moderate ``smooth_window`` yields smoother curves.

    Title: metric name on the first line, then ``env_name``
    """
    if required_tag is None:
        required_tag = []
    if algo_tags is None:
        algo_tags = ["MoG", "FFT", "dqn"]

    api = wandb.Api()
    path = _wandb_path(entity, project)
    runs = list(api.runs(path, order="-created_at"))[:max_runs]

    by_algo: dict[str, list] = defaultdict(list)
    for run in runs:
        if not _run_matches_required(run, required_tag):
            continue
        g = _algo_group(run, algo_tags)
        if g is None:
            continue
        series = _load_series(run, metric, step_metric)
        if series is None:
            continue
        by_algo[g].append(series)

    if not by_algo:
        raise RuntimeError(
            "No runs matched. Check entity, project, required_tag, and algo_tags "
            "(runs must include one of the algo tags, e.g. MoG, FFT, or dqn)."
        )

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)

    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(algo_tags), 1)))

    for idx, algo_tag in enumerate(algo_tags):
        curves = by_algo.get(algo_tag, [])
        if not curves:
            print(f"Warning: no runs for algo tag {algo_tag!r}")
            continue
        all_steps = np.concatenate([c[0] for c in curves])
        s_min, s_max = np.nanmin(all_steps), np.nanmax(all_steps)
        if not np.isfinite(s_min) or not np.isfinite(s_max) or s_max <= s_min:
            continue
        grid = np.linspace(s_min, s_max, grid_points)
        mat = _interp_on_grid(curves, grid)
        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0)
        if smooth_window and smooth_window > 1:
            mean = _smooth_1d(mean, smooth_window)
            std = _smooth_1d(std, smooth_window)
        label = "DQN" if algo_tag == "dqn" else algo_tag
        c = colors[idx % len(colors)]
        plt.plot(grid, mean, color=c, linewidth=2.0, label=f"{label} (n={len(curves)})")
        plt.fill_between(grid, mean - std, mean + std, color=c, alpha=0.2)

    plt.xlabel(step_metric)
    plt.ylabel(metric)
    metric_title = _pretty_metric_label(metric)
    plt.title(
        f"{metric_title} in {env_name}",
        fontsize=12,
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    plot_episodic_return(
        project="Deep-CVI-Experiments",
        entity=None,
        required_tag=[],
        algo_tags=["MoG", "FFT", "dqn"],
        metric="charts/episodic_return",
        step_metric="global_step",
        out="figures/breakout_episodic_return.png",
        grid_points=800,
        max_runs=500,
        smooth_window=41,
        env_name="Breakout MinAtar",
    )
