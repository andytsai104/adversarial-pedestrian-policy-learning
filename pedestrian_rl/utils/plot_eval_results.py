import csv
import os

from pedestrian_rl.utils.eval_utils import plot_evaluation_results

def _to_bool(value):
    if value is None or value == "":
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}

def _to_int_or_none(value):
    if value is None or value == "":
        return None
    return int(float(value))

def _to_float_or_none(value):
    if value is None or value == "":
        return None
    return float(value)

def load_episode_rows_from_csv(csv_path):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "controller": row["controller"],
                "seed": _to_int_or_none(row["seed"]),
                "episode_id": _to_int_or_none(row["episode_id"]),
                "ped_id": row["ped_id"],
                "collision": _to_bool(row["collision"]),
                "physical_collision": _to_bool(row.get("physical_collision")),
                "threshold_collision": _to_bool(row.get("threshold_collision")),
                "near_collision": _to_bool(row["near_collision"]),
                "steps_to_collision": _to_float_or_none(row["steps_to_collision"]),
                "steps_to_threshold_collision": _to_float_or_none(row.get("steps_to_threshold_collision")),
                "steps_to_near_collision": _to_float_or_none(row.get("steps_to_near_collision")),
                "episode_length": _to_int_or_none(row["episode_length"]),
                "avg_speed": _to_float_or_none(row["avg_speed"]),
                "drivable_steps": _to_float_or_none(row["drivable_steps"]),
                "time_on_drivable": _to_float_or_none(row["time_on_drivable"]),
                "stall_steps": _to_float_or_none(row["stall_steps"]),
                "min_vehicle_distance": _to_float_or_none(row["min_vehicle_distance"]),
            })
    return rows

if __name__ == "__main__":
    csv_path = "checkpoints/evaluation/controller_comparison/controller_evaluation_per_episode.csv"
    save_dir = "media/evaluation/model4"

    episode_rows = load_episode_rows_from_csv(csv_path)
    os.makedirs(save_dir, exist_ok=True)
    plot_evaluation_results(rows=episode_rows, save_dir=save_dir)

    print(f"Loaded {len(episode_rows)} episode rows")
    print(f"Plots saved to: {save_dir}")