import csv
import json
import math
import os
import weakref
from collections import defaultdict

import carla
import matplotlib.pyplot as plt
import numpy as np

from ..utils.config_loader import load_config


class EpisodeEvaluator:
    '''
    1. Episode-level metric collection
        - EpisodeEvaluator
        - maybe a helper to clean up sensor safely
    2. Result storage / aggregation
        - save episode rows to CSV
        - load CSV back
        - summarize by controller / seed
        - maybe compute mean, std, median, collision rate
    Generic plotting
        - collision-rate bar plot
        - box/violin plots for:
        - steps_to_collision
        - episode_length
        - avg_speed
        - time_on_drivable
        - stall_steps
        - min_vehicle_distance
    '''
    training_config = load_config("training_config.json")
    td3_params = training_config["td3"]["params"]

    def __init__(
        self,
        world: carla.World,
        target_ped: carla.Actor,
        dt: float,
        stall_speed_threshold: float = 0.05,
    ):
        self.world = world
        self.world_map = world.get_map()
        self.target_ped = target_ped
        self.dt = float(dt)
        self.stall_speed_threshold = float(stall_speed_threshold)
        
        self.collision = False
        self.steps_to_collision = None
        self.physical_collision = False
        self.threshold_collision = False
        self.near_collision = False
        self.steps_to_threshold_collision = None
        self.steps_to_near_collision = None
        self.dist_to_veh = self.td3_params["dist_to_veh"]

        self.episode_steps = 0
        self.drivable_steps = 0
        self.stall_steps = 0

        self.speed_sum = 0.0
        self.min_vehicle_distance = float("inf")

        self.collision_sensor = None
        self.attach_collision_sensor()

    def attach_collision_sensor(self):
        bp = self.world.get_blueprint_library().find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(
            bp,
            carla.Transform(),
            attach_to=self.target_ped
        )

        weak_self = weakref.ref(self)
        self.collision_sensor.listen(
            lambda event: EpisodeEvaluator.on_collision(weak_self, event)
        )

    @staticmethod
    def on_collision(weak_self, event):
        self = weak_self()
        if self is None:
            return

        # only count collisions with vehicles
        if event.other_actor is not None and "vehicle." in event.other_actor.type_id:
            if not self.collision:
                self.physical_collision = True
                self.collision = True
                self.steps_to_collision = self.episode_steps

    def update(self):
        if self.target_ped is None or not self.target_ped.is_alive:
            return

        self.episode_steps += 1

        loc = self.target_ped.get_location()
        vel = self.target_ped.get_velocity()

        # use horizontal speed only
        speed = math.sqrt(vel.x ** 2 + vel.y ** 2)
        self.speed_sum += speed

        if speed < self.stall_speed_threshold:
            self.stall_steps += 1

        wp = self.world_map.get_waypoint(
            loc,
            project_to_road=False,
            lane_type=carla.LaneType.Any
        )
        if wp is not None and wp.lane_type == carla.LaneType.Driving:
            self.drivable_steps += 1

        vehicles = self.world.get_actors().filter("vehicle.*")
        alive_vehicles = [veh for veh in vehicles if veh.is_alive]

        if len(alive_vehicles) > 0:
            min_dist = min(
                veh.get_location().distance(loc)
                for veh in alive_vehicles
            )
            self.min_vehicle_distance = min(self.min_vehicle_distance, min_dist)
        

        # Keep track on near_collision and (threshold) collision
        if len(alive_vehicles) > 0:
            min_dist = min(veh.get_location().distance(loc) for veh in alive_vehicles)
            self.min_vehicle_distance = min(self.min_vehicle_distance, min_dist)

            if min_dist < self.dist_to_veh:
                if not self.threshold_collision:
                    self.threshold_collision = True
                    if self.steps_to_threshold_collision is None:
                        self.steps_to_threshold_collision = self.episode_steps

            elif self.dist_to_veh <= min_dist < (self.dist_to_veh * 4):
                if not self.near_collision:
                    self.near_collision = True
                    if self.steps_to_near_collision is None:
                        self.steps_to_near_collision = self.episode_steps
        

    def get_metrics(self, controller_name: str, seed: int, episode_id: int, ped_id):
        avg_speed = self.speed_sum / self.episode_steps if self.episode_steps > 0 else 0.0

        return {
            "ped_id": ped_id,
            "controller": controller_name,
            "seed": seed,
            "episode_id": episode_id,
            "collision": self.physical_collision or self.threshold_collision,
            "physical_collision": self.physical_collision,
            "threshold_collision": self.threshold_collision,
            "near_collision": self.near_collision,
            "steps_to_collision": self.steps_to_collision,
            "steps_to_threshold_collision": self.steps_to_threshold_collision,
            "steps_to_near_collision": self.steps_to_near_collision,
            "episode_length": self.episode_steps,
            "avg_speed": avg_speed,
            "drivable_steps": self.drivable_steps,
            "time_on_drivable": self.drivable_steps * self.dt,
            "stall_steps": self.stall_steps,
            "min_vehicle_distance": None
            if self.min_vehicle_distance == float("inf")
            else self.min_vehicle_distance,
        }

    def destroy(self):
        if self.collision_sensor is not None:
            try:
                if self.collision_sensor.is_alive:
                    self.collision_sensor.stop()
                    self.collision_sensor.destroy()
            except RuntimeError:
                pass
            finally:
                self.collision_sensor = None


# --- metric saving functions ---
def save_json(data, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as file:
        json.dump(data, file, indent=4)
    print(f"Saved: {save_path}")


def save_episode_results_csv(rows, save_path):
    '''Save episode rows to CSV.''' 
    if len(rows) == 0:
        print("[save_episode_results_csv] No rows to save.")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fieldnames = [
        "controller",
        "seed",
        "episode_id",
        "ped_id",
        "collision",
        "physical_collision",
        "threshold_collision",
        "near_collision",
        "steps_to_collision",
        "steps_to_threshold_collision",
        "steps_to_near_collision",
        "episode_length",
        "avg_speed",
        "drivable_steps",
        "time_on_drivable",
        "stall_steps",
        "min_vehicle_distance",
    ]

    with open(save_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore",)
        writer.writeheader()
        for row in rows:
            clean_row = {key: row.get(key, None) for key in fieldnames}
            writer.writerow(clean_row)

    print(f"Saved: {save_path}")


# --- aggregation + summarization ---
def _finite_values(rows, key):
    values = []
    for row in rows:
        value = row.get(key, None)
        if value is None:
            continue
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            continue
        values.append(float(value))
    return np.asarray(values, dtype=np.float32)


def aggregate_rows_to_episode_level(rows):
    '''
    Convert per-ped evaluator rows into per-episode rows.

    Episode-level interpretation:
        - collision / near-collision style metrics use OR across tracked peds
        - min vehicle distance uses MIN across tracked peds
        - avg speed uses MEAN across tracked peds
        - episode length uses MAX across tracked peds
        - steps-to-event uses MIN across valid tracked peds
    '''
    grouped = defaultdict(list)
    for row in rows:
        key = (row["controller"], row["seed"], row["episode_id"])
        grouped[key].append(row)

    episode_rows = []
    for (controller, seed, episode_id), group_rows in grouped.items():
        episode_length_values = _finite_values(group_rows, "episode_length")
        avg_speed_values = _finite_values(group_rows, "avg_speed")
        drivable_steps_values = _finite_values(group_rows, "drivable_steps")
        time_on_drivable_values = _finite_values(group_rows, "time_on_drivable")
        stall_steps_values = _finite_values(group_rows, "stall_steps")
        min_vehicle_distance_values = _finite_values(group_rows, "min_vehicle_distance")
        steps_to_collision_values = _finite_values(group_rows, "steps_to_collision")
        steps_to_threshold_collision_values = _finite_values(group_rows, "steps_to_threshold_collision")
        steps_to_near_collision_values = _finite_values(group_rows, "steps_to_near_collision")

        episode_rows.append({
            "controller": controller,
            "seed": seed,
            "episode_id": episode_id,
            "ped_id": "episode",
            "collision": any(bool(row.get("collision", False)) for row in group_rows),
            "physical_collision": any(bool(row.get("physical_collision", False)) for row in group_rows),
            "threshold_collision": any(bool(row.get("threshold_collision", False)) for row in group_rows),
            "near_collision": any(bool(row.get("near_collision", False)) for row in group_rows),
            "steps_to_collision": float(np.min(steps_to_collision_values)) if len(steps_to_collision_values) > 0 else None,
            "steps_to_threshold_collision": float(np.min(steps_to_threshold_collision_values)) if len(steps_to_threshold_collision_values) > 0 else None,
            "steps_to_near_collision": float(np.min(steps_to_near_collision_values)) if len(steps_to_near_collision_values) > 0 else None,
            "episode_length": int(np.max(episode_length_values)) if len(episode_length_values) > 0 else None,
            "avg_speed": float(np.mean(avg_speed_values)) if len(avg_speed_values) > 0 else None,
            "drivable_steps": float(np.mean(drivable_steps_values)) if len(drivable_steps_values) > 0 else None,
            "time_on_drivable": float(np.mean(time_on_drivable_values)) if len(time_on_drivable_values) > 0 else None,
            "stall_steps": float(np.mean(stall_steps_values)) if len(stall_steps_values) > 0 else None,
            "min_vehicle_distance": float(np.min(min_vehicle_distance_values)) if len(min_vehicle_distance_values) > 0 else None,
        })

    episode_rows.sort(key=lambda row: (row["controller"], row["seed"], row["episode_id"]))
    return episode_rows


def summarize_episode_results(rows):
    '''Summarize per-controller evaluation metrics from episode-level rows.'''
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["controller"]].append(row)

    summary = {
        "num_rows": int(len(rows)),
        "controllers": {},
    }

    for controller_name, controller_rows in grouped.items():
        collision_values = np.asarray([float(bool(row["collision"])) for row in controller_rows], dtype=np.float32)
        physical_collision_values = np.asarray([float(bool(row.get("physical_collision", False))) for row in controller_rows], dtype=np.float32)
        threshold_collision_values = np.asarray([float(bool(row.get("threshold_collision", False))) for row in controller_rows], dtype=np.float32)
        near_collision_values = np.asarray([float(bool(row.get("near_collision", False))) for row in controller_rows], dtype=np.float32)

        episode_length_values = _finite_values(controller_rows, "episode_length")
        avg_speed_values = _finite_values(controller_rows, "avg_speed")
        time_on_drivable_values = _finite_values(controller_rows, "time_on_drivable")
        stall_steps_values = _finite_values(controller_rows, "stall_steps")
        min_vehicle_distance_values = _finite_values(controller_rows, "min_vehicle_distance")
        steps_to_collision_values = _finite_values(controller_rows, "steps_to_collision")
        steps_to_threshold_collision_values = _finite_values(controller_rows, "steps_to_threshold_collision")
        steps_to_near_collision_values = _finite_values(controller_rows, "steps_to_near_collision")

        summary["controllers"][controller_name] = {
            "num_episodes": int(len(controller_rows)),
            "collision_rate": float(collision_values.mean()) if len(collision_values) > 0 else 0.0,
            "physical_collision_rate": float(physical_collision_values.mean()) if len(physical_collision_values) > 0 else 0.0,
            "threshold_collision_rate": float(threshold_collision_values.mean()) if len(threshold_collision_values) > 0 else 0.0,
            "near_collision_rate": float(near_collision_values.mean()) if len(near_collision_values) > 0 else 0.0,
            "episode_length_mean": float(episode_length_values.mean()) if len(episode_length_values) > 0 else None,
            "episode_length_std": float(episode_length_values.std()) if len(episode_length_values) > 0 else None,
            "avg_speed_mean": float(avg_speed_values.mean()) if len(avg_speed_values) > 0 else None,
            "avg_speed_std": float(avg_speed_values.std()) if len(avg_speed_values) > 0 else None,
            "time_on_drivable_mean": float(time_on_drivable_values.mean()) if len(time_on_drivable_values) > 0 else None,
            "time_on_drivable_std": float(time_on_drivable_values.std()) if len(time_on_drivable_values) > 0 else None,
            "stall_steps_mean": float(stall_steps_values.mean()) if len(stall_steps_values) > 0 else None,
            "stall_steps_std": float(stall_steps_values.std()) if len(stall_steps_values) > 0 else None,
            "min_vehicle_distance_mean": float(min_vehicle_distance_values.mean()) if len(min_vehicle_distance_values) > 0 else None,
            "min_vehicle_distance_std": float(min_vehicle_distance_values.std()) if len(min_vehicle_distance_values) > 0 else None,
            "steps_to_collision_mean": float(steps_to_collision_values.mean()) if len(steps_to_collision_values) > 0 else None,
            "steps_to_collision_std": float(steps_to_collision_values.std()) if len(steps_to_collision_values) > 0 else None,
            "steps_to_threshold_collision_mean": float(steps_to_threshold_collision_values.mean()) if len(steps_to_threshold_collision_values) > 0 else None,
            "steps_to_threshold_collision_std": float(steps_to_threshold_collision_values.std()) if len(steps_to_threshold_collision_values) > 0 else None,
            "steps_to_near_collision_mean": float(steps_to_near_collision_values.mean()) if len(steps_to_near_collision_values) > 0 else None,
            "steps_to_near_collision_std": float(steps_to_near_collision_values.std()) if len(steps_to_near_collision_values) > 0 else None,
        }

    return summary


# --- plotting ---
def set_plot_style():
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
    save_root, _ = os.path.splitext(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_root + ".png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_root}.png")


def _group_metric_values(rows, metric_name):
    grouped = defaultdict(list)
    for row in rows:
        value = row.get(metric_name, None)
        if value is None:
            continue
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            continue
        grouped[row["controller"]].append(float(value))
    return grouped


def plot_boolean_rate(rows, metric_name, title, save_path):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["controller"]].append(float(bool(row.get(metric_name, False))))

    if len(grouped) == 0:
        return

    set_plot_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    labels = list(grouped.keys())
    values = [float(np.mean(grouped[label])) for label in labels]

    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)

    fig.tight_layout()
    save_figure(fig, save_path)


def plot_metric_boxplot(rows, metric_name, title, ylabel, save_path):
    grouped = _group_metric_values(rows, metric_name)
    if len(grouped) == 0:
        return

    labels = [label for label, values in grouped.items() if len(values) > 0]
    data = [grouped[label] for label in labels]

    if len(data) == 0:
        return

    set_plot_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.boxplot(data, tick_labels=labels, patch_artist=False)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)

    fig.tight_layout()
    save_figure(fig, save_path)


def plot_evaluation_results(rows, save_dir):
    '''
    Create publication-friendly controller comparison plots.
    Expected input: episode-level rows from aggregate_rows_to_episode_level().
    '''
    if len(rows) == 0:
        print("[plot_evaluation_results] No rows to plot.")
        return

    os.makedirs(save_dir, exist_ok=True)

    plot_boolean_rate(
        rows=rows,
        metric_name="collision",
        title="Collision rate by controller",
        save_path=os.path.join(save_dir, "collision_rate.png"),
    )
    plot_boolean_rate(
        rows=rows,
        metric_name="near_collision",
        title="Near-collision rate by controller",
        save_path=os.path.join(save_dir, "near_collision_rate.png"),
    )
    plot_metric_boxplot(
        rows=rows,
        metric_name="min_vehicle_distance",
        title="Minimum vehicle distance by controller",
        ylabel="Distance (m)",
        save_path=os.path.join(save_dir, "min_vehicle_distance.png"),
    )
    plot_metric_boxplot(
        rows=rows,
        metric_name="avg_speed",
        title="Average speed by controller",
        ylabel="Speed (m/s)",
        save_path=os.path.join(save_dir, "avg_speed.png"),
    )
    plot_metric_boxplot(
        rows=rows,
        metric_name="episode_length",
        title="Episode length by controller",
        ylabel="Steps",
        save_path=os.path.join(save_dir, "episode_length.png"),
    )