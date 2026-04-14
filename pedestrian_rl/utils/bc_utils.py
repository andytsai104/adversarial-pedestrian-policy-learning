
import os
import csv
import json
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from ..models.bc_model import BehaviorCloningPolicy
from ..models.cnn_encoder import CNNEncoder



# --- Data Loading functions ---
def save_json(data, save_path):
    """Save one dictionary to a json file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as file:
        json.dump(data, file, indent=4)
    print(f"Saved: {save_path}")


def save_checkpoint(model, optimizer, save_path, epoch, extra_info=None):
    """Save one model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if extra_info is not None:
        checkpoint.update(extra_info)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"Saved: {save_path}")


def build_dataloader(
    dataset,
    batch_size,
    shuffle,
    num_workers,
    persistent_workers=False,
    prefetch_factor=2,
    ):
    """Create one dataloader."""
    dataloader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = persistent_workers
        dataloader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(**dataloader_kwargs)



# --- Make sure the episodes are not spliting ---
def _compute_split_counts(total_count, train_ratio, val_ratio, test_ratio):
    '''Compute split counts while keeping the total exact.'''
    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)
    test_count = int(total_count * test_ratio)

    remaining = total_count - (train_count + val_count + test_count)

    for _ in range(remaining):
        if train_ratio >= max(val_ratio, test_ratio):
            train_count += 1
        elif val_ratio >= test_ratio:
            val_count += 1
        else:
            test_count += 1

    return train_count, val_count, test_count



def split_dataset_by_episode(dataset, train_ratio, val_ratio, test_ratio, seed=42):
    '''Split dataset by episode name to avoid train / val / test leakage.'''
    if train_ratio <= 0 or val_ratio < 0 or test_ratio < 0:
        raise ValueError("Split ratios must be non-negative and train_ratio must be > 0.")

    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    episode_names = sorted({episode_name for episode_name, _, _ in dataset.index})
    total_episodes = len(episode_names)

    if total_episodes == 0:
        raise ValueError("No episodes found in dataset.")

    shuffled_episodes = episode_names.copy()
    random.Random(seed).shuffle(shuffled_episodes)

    num_train, num_val, num_test = _compute_split_counts(
        total_count=total_episodes,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    if num_train <= 0:
        raise ValueError("Train split has no episodes. Please adjust the ratios.")

    train_episodes = set(shuffled_episodes[:num_train])
    val_episodes = set(shuffled_episodes[num_train:num_train + num_val])
    test_episodes = set(shuffled_episodes[num_train + num_val:num_train + num_val + num_test])

    train_indices = []
    val_indices = []
    test_indices = []

    for idx, (episode_name, _, _) in enumerate(dataset.index):
        if episode_name in train_episodes:
            train_indices.append(idx)
        elif episode_name in val_episodes:
            val_indices.append(idx)
        elif episode_name in test_episodes:
            test_indices.append(idx)

    split_info = {
        "episode": {
            "train": sorted(list(train_episodes)),
            "val": sorted(list(val_episodes)),
            "test": sorted(list(test_episodes)),
            "num_train": len(train_episodes),
            "num_val": len(val_episodes),
            "num_test": len(test_episodes),
        },
        "sample": {
            "num_train": len(train_indices),
            "num_val": len(val_indices),
            "num_test": len(test_indices),
        },
    }

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices),
        split_info,
    )


# --- Build up models, calculate losses, output metrics ---
def set_seed(seed: int = 42, reproducible_mode: bool = True):
    """Set random seeds and runtime mode."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Make cudnn none deterministic when tuning and deterministic when finalizing
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(reproducible_mode)
        torch.backends.cudnn.benchmark = not bool(reproducible_mode)


def move_batch_to_device(batch, device):
    """Move all tensor values in one batch to the target device."""
    moved_batch = {}
    for key, value in batch.items():
        moved_batch[key] = value.to(device) if torch.is_tensor(value) else value
    return moved_batch

def build_model(config, device):
    """Build BC model."""
    input_channels = config["cnn"]["input_channels"]
    bev_feature_dim = config["cnn"]["bev_feature_dim"]
    hidden_dim = config["cnn"]["hidden_dim"]
    direction_dim = config["cnn"]["direction_dim"]
    dropout = config["bc"]["params"]["dropout"]

    cnn_encoder = CNNEncoder(input_channels=input_channels, feature_dim=bev_feature_dim)
    model = BehaviorCloningPolicy(
        cnn_encoder=cnn_encoder,
        bev_feature_dim=bev_feature_dim,
        # scalar_feature_dim=7,
        hidden_dim=hidden_dim,
        direction_dim=direction_dim,
        dropout=dropout
    ).to(device)

    return model


def normalize_xy_direction(direction_xy, eps=1e-8):
    '''Normalize 2D direction vectors.'''
    norm = torch.norm(direction_xy, dim=-1, keepdim=True)
    norm = torch.clamp(norm, min=eps)
    return direction_xy / norm


def masked_direction_loss(pred_direction, gt_direction, direction_mask):
    '''
    Cosine-style direction loss on valid samples only.
    Invalid direction labels (for nearly stationary samples) are ignored.
    '''
    pred_direction = normalize_xy_direction(pred_direction)
    gt_direction = normalize_xy_direction(gt_direction)

    cosine = torch.sum(pred_direction * gt_direction, dim=-1)
    cosine = torch.clamp(cosine, min=-1.0, max=1.0)
    per_sample_loss = 1.0 - cosine

    valid_mask = direction_mask.bool()
    if valid_mask.any():
        return per_sample_loss[valid_mask].mean()

    return pred_direction.sum() * 0.0



def compute_bc_loss(outputs,
                    batch,
                    speed_loss_weight=1.0,
                    direction_loss_weight=1.0):
    '''Compute BC loss using local-frame targets.'''
    gt_speed = batch["target_speed"].unsqueeze(-1)
    gt_direction = batch["target_direction_local"]
    direction_mask = batch["target_direction_mask"]

    pred_speed = outputs["pred_speed"]
    pred_direction = outputs["pred_direction"]

    loss_speed = F.mse_loss(pred_speed, gt_speed)
    loss_direction = masked_direction_loss(
        pred_direction=pred_direction,
        gt_direction=gt_direction,
        direction_mask=direction_mask,
    )
    loss_total = speed_loss_weight * loss_speed + direction_loss_weight * loss_direction

    return {
        "loss_total": float(loss_total.item()),
        "loss_speed": float(loss_speed.item()),
        "loss_direction": float(loss_direction.item()),
    }, loss_total
    

def compute_bc_accuracy(outputs,
                        batch,
                        speed_tolerance=0.20,
                        direction_tolerance_deg=15.0):
    '''
    Compute regression-style accuracy on the new local-frame targets.

    speed_accuracy:
        abs(pred_speed - gt_speed) <= speed_tolerance

    direction_accuracy:
        angle(pred_direction_local, gt_direction_local) <= direction_tolerance_deg
        computed on valid direction samples only

    joint_accuracy:
        for valid direction samples -> both speed and direction must be correct
        for invalid direction samples -> only speed must be correct
    '''
    pred_speed = outputs["pred_speed"].detach().squeeze(-1)
    gt_speed = batch["target_speed"].detach()

    pred_direction = outputs["pred_direction"].detach()
    gt_direction = batch["target_direction_local"].detach()
    direction_mask = batch["target_direction_mask"].detach().bool()

    pred_direction = normalize_xy_direction(pred_direction)
    gt_direction = normalize_xy_direction(gt_direction)

    speed_abs_error = torch.abs(pred_speed - gt_speed)
    speed_correct = speed_abs_error <= speed_tolerance

    dot = torch.sum(pred_direction * gt_direction, dim=-1)
    dot = torch.clamp(dot, min=-1.0, max=1.0)
    direction_angle_deg_all = torch.rad2deg(torch.acos(dot))

    if direction_mask.any():
        direction_angle_deg = float(direction_angle_deg_all[direction_mask].mean().item())
        direction_correct = direction_angle_deg_all <= direction_tolerance_deg
        direction_accuracy = float(direction_correct[direction_mask].float().mean().item())
    else:
        direction_angle_deg = 0.0
        direction_correct = torch.ones_like(speed_correct, dtype=torch.bool)
        direction_accuracy = 0.0

    joint_correct = torch.where(direction_mask, speed_correct & direction_correct, speed_correct)

    return {
        "speed_mae": float(speed_abs_error.mean().item()),
        "direction_angle_deg": direction_angle_deg,
        "speed_accuracy": float(speed_correct.float().mean().item()),
        "direction_accuracy": direction_accuracy,
        "joint_accuracy": float(joint_correct.float().mean().item()),
    }



def create_history(seed, split_seed, speed_tolerance, direction_tolerance_deg):
    '''Create training history dictionary.'''
    epoch_keys = [
        "total",
        "speed",
        "direction",
        "speed_mae",
        "direction_angle_deg",
        "speed_accuracy",
        "direction_accuracy",
        "joint_accuracy",
    ]

    return {
        "seed": seed,
        "split_seed": split_seed,
        "epoch": {
            "train": {key: [] for key in epoch_keys},
            "val": {key: [] for key in epoch_keys},
        },
        "iteration": {
            "total": [],
            "speed": [],
            "direction": [],
        },
        "test": None,
        "split_info": {},
        "accuracy_definition": {
            "type": "tolerance_based_regression_accuracy_local_frame",
            "speed_tolerance": speed_tolerance,
            "direction_tolerance_deg": direction_tolerance_deg,
        },
    }


def append_epoch_metrics(history, stage, metrics):
    '''Append one epoch of metrics to history.'''
    history["epoch"][stage]["total"].append(metrics["loss_total"])
    history["epoch"][stage]["speed"].append(metrics["loss_speed"])
    history["epoch"][stage]["direction"].append(metrics["loss_direction"])
    history["epoch"][stage]["speed_mae"].append(metrics["speed_mae"])
    history["epoch"][stage]["direction_angle_deg"].append(metrics["direction_angle_deg"])
    history["epoch"][stage]["speed_accuracy"].append(metrics["speed_accuracy"])
    history["epoch"][stage]["direction_accuracy"].append(metrics["direction_accuracy"])
    history["epoch"][stage]["joint_accuracy"].append(metrics["joint_accuracy"])


def run_one_epoch(model,
                  loader,
                  device,
                  optimizer=None,
                  speed_tolerance=0.20,
                  direction_tolerance_deg=15.0,
                  speed_loss_weight=1.0,
                  direction_loss_weight=1.0,
                  grad_clip_norm=None,
                  desc=None,
                  history=None):
    '''Run one training or evaluation epoch.'''
    is_training = optimizer is not None

    metric_sums = {
        "loss_total": 0.0,
        "loss_speed": 0.0,
        "loss_direction": 0.0,
        "speed_mae": 0.0,
        "direction_angle_deg": 0.0,
        "speed_accuracy": 0.0,
        "direction_accuracy": 0.0,
        "joint_accuracy": 0.0,
    }
    num_samples = 0

    if is_training:
        model.train()
        loop = tqdm(loader, desc=desc)
    else:
        model.eval()
        loop = loader

    for batch in loop:
        batch = move_batch_to_device(batch, device)

        if is_training:
            outputs = model(batch)
            loss_info, loss = compute_bc_loss(
                outputs=outputs,
                batch=batch,
                speed_loss_weight=speed_loss_weight,
                direction_loss_weight=direction_loss_weight,
            )

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping prevent too large or too small
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            optimizer.step()
        else:
            with torch.no_grad():
                outputs = model(batch)
                loss_info, _ = compute_bc_loss(
                    outputs=outputs,
                    batch=batch,
                    speed_loss_weight=speed_loss_weight,
                    direction_loss_weight=direction_loss_weight,
                )

        with torch.no_grad():
            accuracy_info = compute_bc_accuracy(
                outputs=outputs,
                batch=batch,
                speed_tolerance=speed_tolerance,
                direction_tolerance_deg=direction_tolerance_deg,
            )

        batch_size = int(batch["target_speed"].shape[0])
        num_samples += batch_size

        for key in loss_info:
            metric_sums[key] += loss_info[key] * batch_size
        for key in accuracy_info:
            metric_sums[key] += accuracy_info[key] * batch_size

        if is_training and history is not None:
            history["iteration"]["total"].append(loss_info["loss_total"])
            history["iteration"]["speed"].append(loss_info["loss_speed"])
            history["iteration"]["direction"].append(loss_info["loss_direction"])

            loop.set_postfix({
                "loss": f"{loss_info['loss_total']:.4f}",
                "joint_acc": f"{accuracy_info['joint_accuracy']:.3f}",
            })

    num_samples = max(num_samples, 1)
    return {key: value / num_samples for key, value in metric_sums.items()}


def summarize_test_metrics(history_list, seed_list):
    '''Summarize test metrics across seeds.'''
    metric_names = [
        "loss_total",
        "loss_speed",
        "loss_direction",
        "speed_mae",
        "direction_angle_deg",
        "speed_accuracy",
        "direction_accuracy",
        "joint_accuracy",
    ]

    summary = {
        "seeds": list(seed_list),
        "per_seed": {},
        "mean": {},
        "std": {},
    }

    for seed, history in zip(seed_list, history_list):
        summary["per_seed"][f"seed_{seed}"] = history["test"]

    for metric_name in metric_names:
        values = np.asarray(
            [history["test"][metric_name] for history in history_list],
            dtype=np.float32,
        )
        summary["mean"][metric_name] = float(values.mean())
        summary["std"][metric_name] = float(values.std())

    return summary


def get_seed_dirs(media_root, checkpoint_root, seed):
    """Get output directories for one seed."""
    seed_name = f"seed_{seed}"
    return (
        os.path.join(media_root, seed_name),
        os.path.join(checkpoint_root, seed_name),
    )




# --- Plotting functions ---
EPOCH_PLOT_SPECS = [
    {
        "history_key": "total",
        "filename": "epoch_total_loss.png",
        "title": "Total loss",
        "ylabel": "Loss",
        "ylim": None,
    },
    {
        "history_key": "speed",
        "filename": "epoch_speed_loss.png",
        "title": "Speed loss",
        "ylabel": "Loss",
        "ylim": None,
    },
    {
        "history_key": "direction",
        "filename": "epoch_direction_loss.png",
        "title": "Direction loss",
        "ylabel": "Loss",
        "ylim": None,
    },
    {
        "history_key": "joint_accuracy",
        "filename": "epoch_joint_accuracy.png",
        "title": "Joint accuracy",
        "ylabel": "Accuracy",
        "ylim": (0.0, 1.0),
    },
    {
        "history_key": "speed_accuracy",
        "filename": "epoch_speed_accuracy.png",
        "title": "Speed accuracy",
        "ylabel": "Accuracy",
        "ylim": (0.0, 1.0),
    },
    {
        "history_key": "direction_accuracy",
        "filename": "epoch_direction_accuracy.png",
        "title": "Direction accuracy",
        "ylabel": "Accuracy",
        "ylim": (0.0, 1.0),
    },
    {
        "history_key": "speed_mae",
        "filename": "epoch_speed_mae.png",
        "title": "Speed absolute error",
        "ylabel": "MAE",
        "ylim": None,
    },
    {
        "history_key": "direction_angle_deg",
        "filename": "epoch_direction_angle_deg.png",
        "title": "Direction angle error",
        "ylabel": "Angle (deg)",
        "ylim": None,
    },
]

ITERATION_PLOT_SPECS = [
    {
        "history_key": "total",
        "filename": "iteration_total_loss.png",
        "title": "Training loss",
        "ylabel": "Loss",
    },
    # {
    #     "history_key": "speed",
    #     "filename": "iteration_speed_loss.png",
    #     "title": "Speed loss",
    #     "ylabel": "Loss",
    # },
    # {
    #     "history_key": "direction",
    #     "filename": "iteration_direction_loss.png",
    #     "title": "Direction loss",
    #     "ylabel": "Loss",
    # },
]


def set_dynamic_x_axis(ax, num_points, use_step_axis=False):
    """
    Dynamic x-axis:
    - if max <= 10  -> ticks every 1
    - if max > 10   -> ticks every 5
    """
    if num_points <= 0:
        return

    if use_step_axis:
        x_max = num_points / 1000.0
        xlabel = r"Step ($\times 10^3$)"
    else:
        x_max = float(num_points)
        xlabel = "Epoch"

    if x_max <= 10:
        tick_step = 1
        x_limit = np.ceil(x_max)
    else:
        tick_step = 5
        x_limit = np.ceil(x_max / tick_step) * tick_step

    ticks = np.arange(0, x_limit + 1e-6, tick_step)

    ax.set_xlim(0, x_limit)
    ax.set_xticks(ticks)
    ax.set_xlabel(xlabel)

def smooth_curve(values, window=1):
    """Smooth one curve with moving average without boundary shrinkage."""
    values = np.asarray(values, dtype=np.float32)

    if len(values) == 0 or window <= 1:
        return values

    window = min(window, len(values))

    if window % 2 == 0:
        window += 1
        window = min(window, len(values))
        if window % 2 == 0:
            window -= 1

    if window <= 1:
        return values

    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)

    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed


def set_plot_style():
    """Set a clean plotting style for paper-like figures."""
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 400,
        "ps.fonttype": 42,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 1.0,
        "axes.facecolor": "#f4f4f4",
        "figure.facecolor": "white",
        "grid.color": "#9a9a9a",
        "grid.alpha": 0.35,
        "grid.linewidth": 0.7,
        "lines.linewidth": 2.3,
        "lines.solid_capstyle": "round",
    })


def save_figure(fig, save_path):
    """Save one figure as png."""
    save_root, _ = os.path.splitext(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_root + ".png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_root}.png")


def plot_train_val_curves(train_values,
                          val_values,
                          title,
                          ylabel,
                          save_path,
                          smooth_window=1,
                          use_step_axis=False,
                          show_raw=True,
                          ylim=None):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(6.6, 4.1))

    train_values = np.asarray(train_values, dtype=np.float32)
    val_values = np.asarray(val_values, dtype=np.float32)

    x_train = np.arange(len(train_values), dtype=np.float32)
    x_val = np.arange(len(val_values), dtype=np.float32)

    if use_step_axis:
        x_train = x_train / 1000.0
        x_val = x_val / 1000.0

    train_curve = smooth_curve(train_values, smooth_window)
    val_curve = smooth_curve(val_values, smooth_window)

    train_color = "#1f77b4"
    val_color = "#2ca02c"

    if show_raw and smooth_window > 1 and len(train_values) > 1:
        ax.plot(x_train, train_values, color=train_color, alpha=0.18, linewidth=1.0)
    ax.plot(x_train, train_curve, color=train_color, label="Train")

    if len(val_values) > 0:
        if show_raw and smooth_window > 1 and len(val_values) > 1:
            ax.plot(x_val, val_values, color=val_color, alpha=0.16, linewidth=1.0)
        ax.plot(x_val, val_curve, color=val_color, label="Validation")

    ax.set_title(title, pad=6)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="best", frameon=False)
    # Set dynamic x-axis
    set_dynamic_x_axis(ax, len(train_values), use_step_axis=use_step_axis)

    if ylim is not None:
        ax.set_ylim(ylim)

    fig.tight_layout()
    save_figure(fig, save_path)


def plot_mean_std_curves(train_curves,
                         val_curves,
                         title,
                         ylabel,
                         save_path,
                         smooth_window=1,
                         use_step_axis=False,
                         ylim=None):
    """Plot mean ± std curves across multiple seeds."""
    if len(train_curves) == 0:
        raise ValueError("train_curves must contain at least one run.")

    set_plot_style()
    fig, ax = plt.subplots(figsize=(6.6, 4.1))

    train_min_length = min(len(curve) for curve in train_curves)
    train_stacked = np.stack(
        [np.asarray(curve[:train_min_length], dtype=np.float32) for curve in train_curves],
        axis=0
    )

    train_mean = smooth_curve(train_stacked.mean(axis=0), smooth_window)
    train_std = smooth_curve(train_stacked.std(axis=0), smooth_window)

    x_train = np.arange(len(train_mean), dtype=np.float32)

    if use_step_axis:
        x_train = x_train / 1000.0
        xlabel = r"Step ($\times 10^3$)"
    else:
        xlabel = "Epoch"

    train_color = "#1f77b4"
    val_color = "#2ca02c"

    ax.plot(x_train, train_mean, color=train_color, label="Train mean")
    ax.fill_between(
        x_train,
        train_mean - train_std,
        train_mean + train_std,
        color=train_color,
        alpha=0.22,
    )

    if len(val_curves) > 0:
        val_min_length = min(len(curve) for curve in val_curves)
        val_stacked = np.stack(
            [np.asarray(curve[:val_min_length], dtype=np.float32) for curve in val_curves],
            axis=0
        )

        val_mean = smooth_curve(val_stacked.mean(axis=0), smooth_window)
        val_std = smooth_curve(val_stacked.std(axis=0), smooth_window)
        x_val = np.arange(len(val_mean), dtype=np.float32)

        if use_step_axis:
            x_val = x_val / 1000.0

        ax.plot(x_val, val_mean, color=val_color, label="Validation mean")
        ax.fill_between(
            x_val,
            val_mean - val_std,
            val_mean + val_std,
            color=val_color,
            alpha=0.18,
        )

    ax.set_title(title, pad=6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="best", frameon=False)
    # Set dynamic x-axis
    set_dynamic_x_axis(ax, len(train_mean), use_step_axis=use_step_axis)

    if ylim is not None:
        ax.set_ylim(ylim)

    fig.tight_layout()
    save_figure(fig, save_path)


def save_test_summary_csv(summary, save_path):
    """Save test result table as csv."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    metric_names = [
        "loss_total",
        "loss_speed",
        "loss_direction",
        "speed_mae",
        "direction_angle_deg",
        "speed_accuracy",
        "direction_accuracy",
        "joint_accuracy",
    ]

    with open(save_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["seed"] + metric_names)

        for seed_name, metric_dict in summary["per_seed"].items():
            writer.writerow([seed_name] + [metric_dict[metric_name] for metric_name in metric_names])

        writer.writerow([])
        writer.writerow(["mean"] + [summary["mean"][metric_name] for metric_name in metric_names])
        writer.writerow(["std"] + [summary["std"][metric_name] for metric_name in metric_names])

    print(f"Saved: {save_path}")


def plot_single_seed_results(history, save_dir, epoch_smooth_window=1, iteration_smooth_window=101):
    '''Save plots for one seed.'''
    os.makedirs(save_dir, exist_ok=True)

    for spec in EPOCH_PLOT_SPECS:
        plot_train_val_curves(
            train_values=history["epoch"]["train"][spec["history_key"]],
            val_values=history["epoch"]["val"][spec["history_key"]],
            title=spec["title"],
            ylabel=spec["ylabel"],
            save_path=os.path.join(save_dir, spec["filename"]),
            smooth_window=epoch_smooth_window,
            use_step_axis=False,
            show_raw=True,
            ylim=spec["ylim"],
        )

    empty_values = np.array([])

    for spec in ITERATION_PLOT_SPECS:
        plot_train_val_curves(
            train_values=history["iteration"][spec["history_key"]],
            val_values=empty_values,
            title=spec["title"],
            ylabel=spec["ylabel"],
            save_path=os.path.join(save_dir, spec["filename"]),
            smooth_window=iteration_smooth_window,
            use_step_axis=True,
            show_raw=True,
            ylim=None,
        )


def plot_multi_seed_results(history_list, save_dir, epoch_smooth_window=1, iteration_smooth_window=101):
    '''Save aggregated plots across multiple seeds.'''
    os.makedirs(save_dir, exist_ok=True)

    for spec in EPOCH_PLOT_SPECS:
        plot_mean_std_curves(
            train_curves=[history["epoch"]["train"][spec["history_key"]] for history in history_list],
            val_curves=[history["epoch"]["val"][spec["history_key"]] for history in history_list],
            title=spec["title"],
            ylabel=spec["ylabel"],
            save_path=os.path.join(save_dir, spec["filename"]),
            smooth_window=epoch_smooth_window,
            use_step_axis=False,
            ylim=spec["ylim"],
        )

    for spec in ITERATION_PLOT_SPECS:
        plot_mean_std_curves(
            train_curves=[history["iteration"][spec["history_key"]] for history in history_list],
            val_curves=[],
            title=spec["title"],
            ylabel=spec["ylabel"],
            save_path=os.path.join(save_dir, spec["filename"]),
            smooth_window=iteration_smooth_window,
            use_step_axis=True,
            ylim=None,
        )