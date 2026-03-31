#!/usr/bin/env python3
"""
Fetch Weights & Biases runs by tags and plot charts/episodic_return (mean ± std per algorithm).

Requires: pip install wandb matplotlib numpy
Login once: wandb login

**Single environment** (e.g. Breakout only): pass ``env_name`` for the title; leave ``env_ids`` unset.

**MinAtar 10M sweep**: pass ``experiment_tag="MinAtar_10M"`` and ``env_ids``. Runs are filtered by
tags (experiment + one algo tag). Environment is taken from ``config.env_id`` when present; otherwise
from the W&B **run name**, which is ``{env_id}__{exp_name}__{seed}__...`` (first segment = env).
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

ALGO_COLORS: dict[str, str] = {
    # Hard-coded 4-model palette requested for benchmark figures.
    "MoG": "#2077b4",  # mid blue
    "dqn": "#2ca02c",  # green
    "QR-DQN": "#e477c2",  # pink
    "IQN": "#15becf",  # light blue
    
    # Fallback colors for any additional algos.
    "C51": "#FF8C42",
    "FQF": "#8A6AE2",
    "FFT": "#9A9A9A",
}


def _algo_colors(algo_tags: list[str]) -> list[str]:
    return [ALGO_COLORS.get(tag, "#444444") for tag in algo_tags]


def _wandb_path(entity: str | None, project: str) -> str:
    if entity:
        return f"{entity}/{project}"
    return project


def _parse_env_id_from_run_name(name: str | None) -> str | None:
    """Training uses ``wandb.init(..., name=f"{env_id}__{exp_name}__{seed}__{time}")`` — env is the first segment."""
    if not name:
        return None
    s = str(name).strip()
    if "__" not in s:
        return None
    return s.split("__", 1)[0].strip() or None


def _get_run_env_id(run, *, use_run_name: bool = True) -> str | None:
    """Prefer ``config.env_id``; if missing, parse from ``run.name`` (see ``_parse_env_id_from_run_name``)."""
    cfg = getattr(run, "config", None)
    if cfg is not None and hasattr(cfg, "get"):
        v = cfg.get("env_id")
        if v is not None and str(v).strip() != "":
            return str(v)
    if use_run_name:
        return _parse_env_id_from_run_name(getattr(run, "name", None))
    return None


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
    tail = metric.split("/")[-1]
    return tail.replace("_", " ").strip().title()


def _pretty_env_title(env_id: str) -> str:
    if env_id.endswith("-MinAtar"):
        return env_id.replace("-MinAtar", " MinAtar").replace("-", " ")
    return env_id.replace("-", " ").replace("_", " ").title()


def _draw_algo_curves_on_ax(
    ax,
    by_algo: dict[str, list],
    algo_tags: list[str],
    colors: list[str],
    *,
    grid_points: int,
    smooth_window: int,
    step_metric: str,
    metric: str,
    show_ylabel: bool,
    autoscale_y: bool = False,
    y_top_margin: float = 0.1,
    y_bottom: float | None = 0.0,
) -> None:
    """If ``autoscale_y``, set y-axis to ``[y_bottom, (max of mean+std) * (1 + y_top_margin)]`` per panel."""
    ymax_track = -np.inf
    ymin_track = np.inf

    for idx, algo_tag in enumerate(algo_tags):
        curves = by_algo.get(algo_tag, [])
        if not curves:
            continue
        all_steps = np.concatenate([c[0] for c in curves])
        s_min, s_max = np.nanmin(all_steps), np.nanmax(all_steps)
        if not np.isfinite(s_min) or not np.isfinite(s_max) or s_max <= s_min:
            continue
        grid = np.linspace(s_min, s_max, grid_points)
        mat = _interp_on_grid(curves, grid)
        n_per_step = np.sum(np.isfinite(mat), axis=0)
        std_valid = n_per_step >= 2
        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0)
        std[~std_valid] = np.nan
        if smooth_window and smooth_window > 1:
            mean = _smooth_1d(mean, smooth_window)
            std = _smooth_1d(std, smooth_window)
            # Keep variance hidden where fewer than 2 runs contribute, even after smoothing.
            std[~std_valid] = np.nan
        upper = mean + std
        lower = mean - std
        ymax_track = max(ymax_track, float(np.nanmax(upper)))
        ymin_track = min(ymin_track, float(np.nanmin(lower)))

        label_map = {
            "dqn": "DQN",
            "C51": "C51",
            "MoG": "MoG",
            "FFT": "FFT",
            "QR-DQN": "QR-DQN",
            "IQN": "IQN",
            "FQF": "FQF",
        }
        label = label_map.get(algo_tag, algo_tag)
        c = colors[idx % len(colors)]
        ax.plot(grid, mean, color=c, linewidth=2.0, label=f"{label} (n={len(curves)})")
        ax.fill_between(grid, lower, upper, color=c, alpha=0.2)

    ax.set_xlabel(step_metric)
    if show_ylabel:
        ax.set_ylabel(metric)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    if autoscale_y and np.isfinite(ymax_track) and ymax_track > 0:
        top = ymax_track * (1.0 + y_top_margin)
        if y_bottom is not None:
            bot = float(y_bottom)
        elif np.isfinite(ymin_track) and ymin_track < 0:
            bot = ymin_track * (1.0 + y_top_margin)
        else:
            bot = 0.0
        ax.set_ylim(bottom=bot, top=top)


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
    experiment_tag: str | None = None,
    env_ids: list[str] | None = None,
    use_run_name_for_env: bool = True,
    multi_env_y_top_margin: float = 0.1,
) -> None:
    """Plot mean ± std episodic return from W&B.

    - **Legacy (single panel):** leave ``env_ids`` as ``None``. Optionally set ``experiment_tag`` to filter runs.
    - **One env, filtered by id:** ``env_ids=[\"Breakout-MinAtar\"]`` and ``experiment_tag`` if needed.
    - **Five-env grid:** pass ``experiment_tag`` (e.g. ``\"MinAtar_10M\"``) and ``env_ids`` with 2+ entries;
      one subplot per env, **independent y-axes** (not shared). Each panel’s top is
      ``(max of mean+std across algos) * (1 + multi_env_y_top_margin)`` (default 10 % headroom) so scales
      match each game.

    If ``config.env_id`` is missing on older runs, set ``use_run_name_for_env=True`` (default) to recover
    env from ``run.name``.
    """
    if required_tag is None:
        required_tag = []
    if algo_tags is None:
        algo_tags = ["MoG", "dqn", "C51", "QR-DQN", "IQN", "FQF"]

    required_effective = list(required_tag)
    if experiment_tag:
        required_effective.append(experiment_tag)

    api = wandb.Api()
    path = _wandb_path(entity, project)
    runs = list(api.runs(path, order="-created_at"))[:max_runs]

    colors = _algo_colors(algo_tags)
    metric_title = _pretty_metric_label(metric)

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)

    # ----- Multi-env grid (e.g. MinAtar_10M × 5 games) -----
    if env_ids is not None and len(env_ids) > 1:
        by_env_algo: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for run in runs:
            if not _run_matches_required(run, required_effective):
                continue
            eid = _get_run_env_id(run, use_run_name=use_run_name_for_env)
            if eid is None or eid not in env_ids:
                continue
            g = _algo_group(run, algo_tags)
            if g is None:
                continue
            series = _load_series(run, metric, step_metric)
            if series is None:
                continue
            by_env_algo[eid][g].append(series)

        if not by_env_algo:
            raise RuntimeError(
                "No runs matched. Check experiment_tag, env_ids, algo tags, and env (config.env_id or run.name)."
            )

        n = len(env_ids)
        fig_w = max(14, 3.2 * n)
        fig, axes = plt.subplots(1, n, figsize=(fig_w, 4.2), sharey=False, squeeze=False)
        ax_flat = axes[0]

        for j, eid in enumerate(env_ids):
            ax = ax_flat[j]
            by_algo = by_env_algo.get(eid, {})
            if not any(len(by_algo.get(t, [])) > 0 for t in algo_tags):
                ax.text(0.5, 0.5, "No runs", ha="center", va="center", transform=ax.transAxes)
            else:
                _draw_algo_curves_on_ax(
                    ax,
                    dict(by_algo),
                    algo_tags,
                    colors,
                    grid_points=grid_points,
                    smooth_window=smooth_window,
                    step_metric=step_metric,
                    metric=metric,
                    show_ylabel=(j == 0),
                    autoscale_y=True,
                    y_top_margin=multi_env_y_top_margin,
                    y_bottom=0.0,
                )
            ax.set_title(_pretty_env_title(eid), fontsize=10)

        et = experiment_tag or ""
        supt = f"{et} {metric_title}  "
        fig.suptitle(supt, fontsize=12, y=1.03)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out}")
        return

    # ----- Single panel -----
    by_algo: dict[str, list] = defaultdict(list)
    for run in runs:
        if not _run_matches_required(run, required_effective):
            continue
        if env_ids is not None and len(env_ids) == 1:
            if _get_run_env_id(run, use_run_name=use_run_name_for_env) != env_ids[0]:
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
            "No runs matched. Check entity, project, experiment_tag, required_tag, env_ids, and algo_tags."
        )

    plt.figure(figsize=(10, 6))
    _draw_algo_curves_on_ax(
        plt.gca(),
        dict(by_algo),
        algo_tags,
        colors,
        grid_points=grid_points,
        smooth_window=smooth_window,
        step_metric=step_metric,
        metric=metric,
        show_ylabel=True,
    )
    plt.title(f"{metric_title} in {env_name}", fontsize=12)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Wrote {out}")


def plot_minatar_10m_grid(
    *,
    project: str = "Deep-CVI-Experiments",
    entity: str | None = None,
    out: str = "figures/minatar_10m_episodic_return.png",
    experiment_tag: str = "MinAtar_10M",
    env_ids: list[str] | None = None,
    include_pong_misc: bool = False,
    use_run_name_for_env: bool = True,
    **kwargs,
) -> None:
    """Convenience wrapper for the MinAtar 10M multi-env figure.

    By default uses **four** MinAtar games (Asterix, Breakout, Freeway, Space Invaders). Set
    ``include_pong_misc=True`` to add ``Pong-misc`` as a fifth panel (Slurm stand-in for Seaquest).
    If ``env_ids`` is passed explicitly, ``include_pong_misc`` is ignored.
    """
    if env_ids is None:
        env_ids = [
            "Asterix-MinAtar",
            "Breakout-MinAtar",
            "Freeway-MinAtar",
            "SpaceInvaders-MinAtar",
        ]
        if include_pong_misc:
            env_ids = [*env_ids, "Pong-misc"]
    plot_episodic_return(
        project=project,
        entity=entity,
        experiment_tag=experiment_tag,
        env_ids=env_ids,
        out=out,
        use_run_name_for_env=use_run_name_for_env,
        **kwargs,
    )


if __name__ == "__main__":
    # Comment out the experiment you are *not* plotting; leave exactly one ``plot_episodic_return`` call active.
    # -------------------------------------------------------------------------

    # (1) Breakout MinAtar — 3 algos (MoG / FFT / dqn), 3M-step sweep (no MinAtar_10M tag).
    # plot_episodic_return(
    #     project="Deep-CVI-Experiments",
    #     entity=None,
    #     required_tag=[],
    #     algo_tags=["MoG", "FFT", "dqn"],
    #     experiment_tag=None,
    #     env_ids=["Breakout-MinAtar"],
    #     use_run_name_for_env=True,
    #     metric="charts/episodic_return",
    #     step_metric="global_step",
    #     out="figures/breakout_3algo_3M_episodic_return.png",
    #     grid_points=800,
    #     max_runs=500,
    #     smooth_window=41,
    #     env_name="Breakout MinAtar",
    # )

    # (2) MinAtar 10M — one panel per game; tags: MinAtar_10M + MoG|dqn|C51|QR-DQN|IQN|FQF.
    #     Set include_pong_misc=True to add the Pong-misc panel (5 columns); default is 4 MinAtar games only.
    plot_minatar_10m_grid(
        project="Deep-CVI-Experiments",
        entity=None,
        experiment_tag="MinAtar_10M",
        include_pong_misc=False,
        use_run_name_for_env=True,
        algo_tags=["MoG", "dqn", "C51", "QR-DQN", "IQN"],
        metric="charts/episodic_return",
        step_metric="global_step",
        out="figures/minatar_10m_episodic_return.png",
        grid_points=800,
        max_runs=2000,
        smooth_window=41,
    )
