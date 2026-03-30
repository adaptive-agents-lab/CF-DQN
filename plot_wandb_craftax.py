#!/usr/bin/env python3
"""
Plot Craftax benchmark curves from Weights & Biases (mean ± std over seeds per algorithm).

Panels:
  - charts/episodic_return
  - charts/episodic_length
  - losses/q_values
  - losses/q_gap
  - losses/mog_entropy (MoG runs only; others omitted)
  - losses/sigma_mean (MoG only)
  - losses/cf_loss_imag (MoG only)

Requires: wandb, matplotlib, numpy. Login: wandb login
"""
from __future__ import annotations

import os

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit("Please install matplotlib: pip install matplotlib") from e

try:
    import wandb
except ImportError as e:
    raise SystemExit("Please install wandb: pip install wandb") from e

from plot_wandb_episodic_return import (
    _algo_group,
    _draw_algo_curves_on_ax,
    _interp_on_grid,
    _load_series,
    _run_matches_required,
    _smooth_1d,
    _wandb_path,
)


def plot_craftax_benchmark(
    *,
    project: str = "Deep-CVI-Experiments",
    entity: str | None = None,
    experiment_tag: str = "Craftax_50M",
    algo_tags: list[str] | None = None,
    env_id: str | None = None,
    step_metric: str = "global_step",
    out: str = "figures/craftax_50m_benchmark.png",
    grid_points: int = 800,
    max_runs: int = 2000,
    smooth_window: int = 41,
) -> None:
    """Load runs tagged ``experiment_tag`` and one of ``algo_tags``; optional filter by ``config.env_id``."""
    if algo_tags is None:
        algo_tags = ["MoG", "dqn", "C51"]

    required = [experiment_tag]
    api = wandb.Api()
    path = _wandb_path(entity, project)
    runs = list(api.runs(path, order="-created_at"))[:max_runs]

    metrics_rows: list[tuple[str, str, bool]] = [
        ("charts/episodic_return", "Episodic return", False),
        ("charts/episodic_length", "Episodic length", False),
        ("losses/q_values", "Mean Q (logged batch)", False),
        ("losses/q_gap", "Q gap (sharpness)", False),
        ("losses/mog_entropy", "MoG mixture entropy H(pi)", True),
        ("losses/sigma_mean", "MoG sigma mean", True),
        ("losses/cf_loss_imag", "CF loss |Im(residual)|", True),
    ]

    colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(algo_tags), 1)))
    by_metric_algo: dict[str, dict[str, list]] = {m[0]: {a: [] for a in algo_tags} for m in metrics_rows}

    for run in runs:
        if not _run_matches_required(run, required):
            continue
        if env_id is not None:
            cfg = getattr(run, "config", None)
            rid = cfg.get("env_id") if cfg is not None and hasattr(cfg, "get") else None
            if rid is not None and str(rid) != env_id:
                continue
        g = _algo_group(run, algo_tags)
        if g is None:
            continue
        for metric_key, _title, mog_only in metrics_rows:
            if mog_only and g != "MoG":
                continue
            series = _load_series(run, metric_key, step_metric)
            if series is None:
                continue
            by_metric_algo[metric_key][g].append(series)

    n_panels = len(metrics_rows)
    fig_h = max(10.0, 2.8 * n_panels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, fig_h), sharex=True)
    if n_panels == 1:
        axes = [axes]

    for ax, (metric_key, title, mog_only) in zip(axes, metrics_rows, strict=True):
        by_algo = by_metric_algo[metric_key]
        tags_this = ["MoG"] if mog_only else algo_tags
        if not any(len(by_algo.get(t, [])) > 0 for t in tags_this):
            ax.text(0.5, 0.5, f"No data: {metric_key}", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, fontsize=10)
            continue
        _draw_algo_curves_on_ax(
            ax,
            dict(by_algo),
            tags_this,
            colors,
            grid_points=grid_points,
            smooth_window=smooth_window,
            step_metric=step_metric,
            metric=metric_key,
            show_ylabel=True,
            autoscale_y=True,
            y_top_margin=0.12,
            y_bottom=None,
        )
        ax.set_title(title, fontsize=10)

    axes[-1].set_xlabel(step_metric)
    fig.suptitle(f"{experiment_tag} — Craftax benchmark", fontsize=12, y=1.002)
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    plot_craftax_benchmark(
        project="Deep-CVI-Experiments",
        entity=None,
        experiment_tag="Craftax_50M",
        env_id="Craftax-Classic-Symbolic-v1",
        out="figures/craftax_50m_benchmark.png",
    )
