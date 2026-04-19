#!/usr/bin/env python3
"""
Plot three paper-style TD3 training figures from td3_training_history.json.

Figures:
1. reward over episode
2. training loss over episode (actor loss + critic loss in one plot)
3. minimum vehicle distance over episode

Usage
-----
python plot_td3_three_curves.py /path/to/td3_training_history.json

Optional
--------
python plot_td3_three_curves.py /path/to/td3_training_history.json \
    --outdir /path/to/output_dir \
    --smooth 9 \
    --tick-step 20
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def set_paper_style():
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 400,
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 0.9,
        "lines.linewidth": 2.0,
        "grid.alpha": 0.28,
        "grid.linewidth": 0.6,
        "grid.linestyle": "--",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def smooth_curve(values, window=1):
    values = np.asarray(values, dtype=np.float32)

    if len(values) == 0 or window <= 1:
        return values

    window = int(max(1, round(window)))
    window = min(window, len(values))

    if window % 2 == 0:
        window = max(1, window - 1)

    if window <= 1:
        return values

    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def fill_missing(values):
    arr = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(arr)

    if not valid.any():
        return np.asarray([], dtype=np.float32)

    fill_value = float(np.nanmedian(arr[valid]))
    return np.where(valid, arr, fill_value).astype(np.float32)


def get_episode_axis(n):
    return np.arange(1, n + 1, dtype=np.int32)


def set_episode_ticks(ax, num_episodes, tick_step=20):
    if num_episodes <= 0:
        return

    tick_step = max(1, int(tick_step))
    end_tick = int(np.ceil(num_episodes / tick_step) * tick_step)
    ticks = np.arange(0, end_tick + 1, tick_step, dtype=np.int32)

    ax.set_xlim(0, max(num_episodes, tick_step))
    ax.set_xticks(ticks)
    ax.set_xlabel("Episode")


def plot_single_curve(values, title, ylabel, save_path, smooth_window=1, tick_step=20, show_raw=True):
    values = fill_missing(values)
    if len(values) == 0:
        print(f"Skip empty figure: {save_path}")
        return

    set_paper_style()
    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    x = get_episode_axis(len(values))
    y_smooth = smooth_curve(values, smooth_window)

    if show_raw and smooth_window > 1 and len(values) > 1:
        ax.plot(x, values, alpha=0.20, linewidth=1.0, label="Raw")

    # ax.plot(x, y_smooth, label="Smoothed")
    ax.plot(x, y_smooth)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.set_axisbelow(True)
    set_episode_ticks(ax, len(values), tick_step=tick_step)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_loss_curve(actor_values, critic_values, title, ylabel, save_path, smooth_window=1, tick_step=20):
    actor_values = fill_missing(actor_values)
    critic_values = fill_missing(critic_values)

    if len(actor_values) == 0 and len(critic_values) == 0:
        print(f"Skip empty figure: {save_path}")
        return

    set_paper_style()
    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    if len(actor_values) > 0:
        x_actor = get_episode_axis(len(actor_values))
        ax.plot(x_actor, smooth_curve(actor_values, smooth_window), label="Actor loss")

    if len(critic_values) > 0:
        x_critic = get_episode_axis(len(critic_values))
        ax.plot(x_critic, smooth_curve(critic_values, smooth_window), label="Critic loss")

    num_episodes = max(len(actor_values), len(critic_values))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.set_axisbelow(True)
    set_episode_ticks(ax, num_episodes, tick_step=tick_step)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def load_history(json_path):
    with open(json_path, "r") as f:
        history = json.load(f)

    if not isinstance(history, list) or len(history) == 0:
        raise ValueError("td3_training_history.json must contain a non-empty list.")

    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=str, help="Path to td3_training_history.json")
    parser.add_argument("--outdir", type=str, default=None, help="Directory to save output plots")
    parser.add_argument("--smooth", type=int, default=9, help="Smoothing window")
    parser.add_argument("--tick-step", type=int, default=100, help="Episode tick interval")
    args = parser.parse_args()

    json_path = Path(args.json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Cannot find file: {json_path}")

    history = load_history(json_path)

    reward = [ep.get("reward", np.nan) for ep in history]
    actor_loss = [ep.get("actor_loss_mean", np.nan) for ep in history]
    critic_loss = [ep.get("critic_loss_mean", np.nan) for ep in history]
    min_vehicle_distance = [ep.get("min_vehicle_distance", np.nan) for ep in history]

    outdir = Path(args.outdir) if args.outdir else json_path.parent / "paper_style_plots"
    outdir.mkdir(parents=True, exist_ok=True)

    plot_single_curve(
        values=reward,
        title="TD3 Training Reward",
        ylabel="Episode Reward",
        save_path=outdir / "td3_reward_over_episode.png",
        smooth_window=args.smooth,
        tick_step=args.tick_step,
        show_raw=False,
    )

    plot_loss_curve(
        actor_values=actor_loss,
        critic_values=critic_loss,
        title="TD3 Training Loss",
        ylabel="Loss",
        save_path=outdir / "td3_training_loss_over_episode.png",
        smooth_window=args.smooth,
        tick_step=args.tick_step,
    )

    plot_single_curve(
        values=min_vehicle_distance,
        title="Minimum Vehicle Distance",
        ylabel="Distance to Vehicle (m)",
        save_path=outdir / "td3_min_vehicle_distance_over_episode.png",
        smooth_window=args.smooth,
        tick_step=args.tick_step,
        show_raw=False,
    )

    print("\\nDone.")
    print(f"Output directory: {outdir}")


if __name__ == "__main__":
    main()
