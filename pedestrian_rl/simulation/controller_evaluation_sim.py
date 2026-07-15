import argparse
import json
import os
import random
from typing import Dict, List, Optional

import numpy as np
import torch

from ..models.bc_model import BehaviorCloningPolicy
from ..utils.config_loader import load_config
from ..utils.eval_utils import (
    EpisodeEvaluator,
    aggregate_rows_to_episode_level,
    load_episode_results_csv,
    plot_evaluation_results,
    save_episode_results_csv,
    save_json,
    summarize_episode_results,
)
from ..utils.policy_runners.bc_policy_runner import PolicyRunner
from ..utils.td3_utils import PedestrianRLEnv, build_td3_agent


DEFAULT_NUM_EPISODES = 50
DEFAULT_EVALUATION_SEED = 42
DEFAULT_NUM_MODEL_PEDS = 1
DEFAULT_OUTPUT_ROOT = os.path.join("checkpoints", "evaluation", "bc_vs_td3")
DEFAULT_PLOT_ROOT = os.path.join("media", "evaluation", "bc_vs_td3")


CONTROLLER_LABELS = {
    "bc": "BC",
    "pure_td3": "Pure TD3",
    "bc_td3": "BC + TD3",
}


def load_json(json_path: str) -> dict:
    with open(json_path, "r") as file:
        return json.load(file)


def set_evaluation_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_best_bc_checkpoint(config: dict) -> tuple[str, int, float]:
    checkpoint_root_dir = config["bc"]["checkpoint_dir"]
    summary_path = os.path.join(checkpoint_root_dir, "multi_seed_summary.json")

    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"BC multi-seed summary not found: {summary_path}\n"
            "Pass --bc-checkpoint explicitly or update bc.checkpoint_dir."
        )

    multi_seed_summary = load_json(summary_path)
    best_seed_name = None
    best_joint_acc = float("-inf")

    for seed in multi_seed_summary.get("seeds", []):
        seed_name = f"seed_{seed}"
        seed_info = multi_seed_summary.get("per_seed", {}).get(seed_name)
        if seed_info is None:
            continue
        joint_acc = float(seed_info.get("joint_accuracy", float("-inf")))
        if joint_acc > best_joint_acc:
            best_joint_acc = joint_acc
            best_seed_name = seed_name

    if best_seed_name is None:
        raise ValueError(f"Could not determine the best BC checkpoint from {summary_path}")

    checkpoint_path = os.path.join(checkpoint_root_dir, best_seed_name, "best_model.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"BC checkpoint not found: {checkpoint_path}")

    return checkpoint_path, int(best_seed_name.split("_")[-1]), best_joint_acc


def resolve_td3_checkpoint(path_or_dir: str) -> str:
    """Resolve a TD3 checkpoint from either a .pt file or a checkpoint directory."""
    if not path_or_dir:
        raise ValueError("TD3 checkpoint path cannot be empty.")

    if os.path.isfile(path_or_dir):
        return path_or_dir

    if not os.path.isdir(path_or_dir):
        raise FileNotFoundError(f"TD3 checkpoint file/directory not found: {path_or_dir}")

    preferred_names = ("td3_last.pt", "best_model.pt", "td3_best.pt")
    for filename in preferred_names:
        candidate = os.path.join(path_or_dir, filename)
        if os.path.isfile(candidate):
            return candidate

    candidates = []
    for filename in os.listdir(path_or_dir):
        if filename.startswith("td3_episode_") and filename.endswith(".pt"):
            try:
                episode_num = int(filename.removeprefix("td3_episode_").removesuffix(".pt"))
            except ValueError:
                continue
            candidates.append((episode_num, os.path.join(path_or_dir, filename)))

    if not candidates:
        raise FileNotFoundError(
            f"No TD3 checkpoint found in {path_or_dir}. Expected td3_last.pt, "
            "td3_best.pt, best_model.pt, or td3_episode_<N>.pt."
        )

    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def _collect_evaluator_rows(
    evaluators: Dict[int, EpisodeEvaluator],
    controller_name: str,
    seed: int,
    episode_id: int,
    termination_reason: str,
) -> List[dict]:
    rows = []
    for ped_id, evaluator in evaluators.items():
        row = evaluator.get_metrics(
            controller_name=controller_name,
            seed=seed,
            episode_id=episode_id,
            ped_id=ped_id,
        )
        row["termination_reason"] = termination_reason
        rows.append(row)
    return rows


def run_bc_evaluation(
    checkpoint_path: str,
    num_episodes: int,
    num_model_peds: int,
    evaluation_seed: int,
    no_rendering_mode: bool,
    device: str,
) -> List[dict]:
    set_evaluation_seed(evaluation_seed)

    runner = PolicyRunner(
        model_class=BehaviorCloningPolicy,
        model_name=CONTROLLER_LABELS["bc"],
        checkpoint_path=checkpoint_path,
        training_config_name="training_config.json",
        sim_config_name="sim_config.json",
        no_rendering_mode=no_rendering_mode,
        device=device,
        num_model_peds=num_model_peds,
        evaluation=True,
        evaluation_seed=evaluation_seed,
    )

    try:
        runner.reset_episode()

        while runner.current_episode_id < num_episodes:
            runner.step_once(render_bev=True)

    finally:
        runner.close_target_wrappers()
        runner.close_evaluators()

    return runner.get_completed_rows()

def run_td3_evaluation(
    checkpoint_path: str,
    controller_name: str,
    num_episodes: int,
    evaluation_seed: int,
    no_rendering_mode: bool,
    device: str,
) -> List[dict]:
    set_evaluation_seed(evaluation_seed)
    training_config = load_config("training_config.json")
    env = PedestrianRLEnv(
        sim_config_name="sim_config.json",
        training_config_name="training_config.json",
        no_rendering_mode=no_rendering_mode,
        render_bev=False,
        device=device,
    )
    agent = build_td3_agent(
        training_config=training_config,
        max_speed=env.max_ped_speed,
        device=env.device,
    )
    agent.load(checkpoint_path=checkpoint_path, load_optimizers=False)
    print(f"[{controller_name}] Loaded checkpoint: {checkpoint_path}")

    completed_rows = []
    try:
        for episode_id in range(num_episodes):
            obs_dict, _ = env.reset()
            evaluators = {
                ped_id: EpisodeEvaluator(
                    world=env.world,
                    target_ped=ped,
                    dt=env.fixed_delta_seconds,
                    stall_speed_threshold=env.stall_speed_threshold,
                )
                for ped_id, ped in env.target_peds.items()
            }
            termination_reason = "unknown"

            try:
                while True:
                    if not obs_dict:
                        termination_reason = "no_active_pedestrians"
                        break

                    action_dict = {
                        ped_id: agent.select_action(obs, add_noise=False)
                        for ped_id, obs in obs_dict.items()
                    }
                    next_obs_dict, _, _, _, info = env.step(action_dict)

                    for evaluator in evaluators.values():
                        evaluator.update()

                    obs_dict = next_obs_dict
                    if bool(info.get("episode_done", False)):
                        termination_reason = info.get("episode_reason") or "episode_done"
                        break

                completed_rows.extend(
                    _collect_evaluator_rows(
                        evaluators=evaluators,
                        controller_name=controller_name,
                        seed=evaluation_seed,
                        episode_id=episode_id,
                        termination_reason=termination_reason,
                    )
                )
            finally:
                for evaluator in evaluators.values():
                    evaluator.destroy()

            print(
                f"[{controller_name}] episode {episode_id + 1}/{num_episodes} "
                f"finished: {termination_reason}"
            )
    finally:
        env.close()

    return completed_rows


def evaluate_models(
    bc_checkpoint: str,
    pure_td3_checkpoint: str,
    bc_td3_checkpoint: str,
    num_episodes: int = DEFAULT_NUM_EPISODES,
    num_model_peds: int = DEFAULT_NUM_MODEL_PEDS,
    evaluation_seed: int = DEFAULT_EVALUATION_SEED,
    no_rendering_mode: bool = False,
    device: str = "cuda",
    output_root: str = DEFAULT_OUTPUT_ROOT,
    plot_root: str = DEFAULT_PLOT_ROOT,
) -> tuple[List[dict], dict]:
    checkpoints = {
        CONTROLLER_LABELS["bc"]: bc_checkpoint,
        CONTROLLER_LABELS["pure_td3"]: resolve_td3_checkpoint(pure_td3_checkpoint),
        CONTROLLER_LABELS["bc_td3"]: resolve_td3_checkpoint(bc_td3_checkpoint),
    }

    print("=" * 72)
    print("Controller evaluation setup")
    for controller_name, checkpoint_path in checkpoints.items():
        print(f"{controller_name:>10}: {checkpoint_path}")
    print(f"Episodes/controller: {num_episodes}")
    print(f"Controlled pedestrians/episode: {num_model_peds}")
    print(f"Evaluation seed: {evaluation_seed}")
    print("=" * 72)

    all_rows = []
    all_rows.extend(
        run_bc_evaluation(
            checkpoint_path=checkpoints[CONTROLLER_LABELS["bc"]],
            num_episodes=num_episodes,
            num_model_peds=num_model_peds,
            evaluation_seed=evaluation_seed,
            no_rendering_mode=no_rendering_mode,
            device=device,
        )
    )
    all_rows.extend(
        run_td3_evaluation(
            checkpoint_path=checkpoints[CONTROLLER_LABELS["pure_td3"]],
            controller_name=CONTROLLER_LABELS["pure_td3"],
            num_episodes=num_episodes,
            evaluation_seed=evaluation_seed,
            no_rendering_mode=no_rendering_mode,
            device=device,
        )
    )
    all_rows.extend(
        run_td3_evaluation(
            checkpoint_path=checkpoints[CONTROLLER_LABELS["bc_td3"]],
            controller_name=CONTROLLER_LABELS["bc_td3"],
            num_episodes=num_episodes,
            evaluation_seed=evaluation_seed,
            no_rendering_mode=no_rendering_mode,
            device=device,
        )
    )

    os.makedirs(output_root, exist_ok=True)
    os.makedirs(plot_root, exist_ok=True)

    per_ped_path = os.path.join(output_root, "controller_evaluation_per_ped.csv")
    per_episode_path = os.path.join(output_root, "controller_evaluation_per_episode.csv")
    summary_path = os.path.join(output_root, "controller_evaluation_summary.json")

    episode_rows = aggregate_rows_to_episode_level(all_rows)
    summary = summarize_episode_results(episode_rows)
    summary["metadata"] = {
        "num_episodes_per_controller": int(num_episodes),
        "num_model_peds": int(num_model_peds),
        "evaluation_seed": int(evaluation_seed),
        "checkpoints": checkpoints,
    }

    save_episode_results_csv(all_rows, per_ped_path)
    save_episode_results_csv(episode_rows, per_episode_path)
    save_json(summary, summary_path)
    plot_evaluation_results(episode_rows, plot_root)

    print("\nEvaluation finished.")
    print(f"Per-ped CSV    : {per_ped_path}")
    print(f"Per-episode CSV: {per_episode_path}")
    print(f"Summary        : {summary_path}")
    print(f"Plots          : {plot_root}")
    return all_rows, summary


def plot_saved_results(csv_path: str, plot_root: str, show_titles: bool = True,) -> None:
    rows = load_episode_results_csv(csv_path)
    plot_evaluation_results(rows, plot_root, show_titles=show_titles)
    print(f"Re-plotted {len(rows)} episode rows from {csv_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare BC, pure TD3, and BC-initialized TD3 pedestrian policies."
    )
    parser.add_argument("--bc-checkpoint", default=None)
    parser.add_argument("--pure-td3-checkpoint", default=None)
    parser.add_argument("--bc-td3-checkpoint", default=None)
    parser.add_argument("--num-episodes", type=int, default=DEFAULT_NUM_EPISODES)
    parser.add_argument("--num-model-peds", type=int, default=DEFAULT_NUM_MODEL_PEDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_EVALUATION_SEED)
    parser.add_argument("--device", default="cuda")
    # parser.add_argument("--render", action="store_true", help="Enable CARLA rendering.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--plot-root", default=DEFAULT_PLOT_ROOT)
    parser.add_argument(
        "--plot-only",
        default=None,
        metavar="EPISODE_CSV",
        help="Skip CARLA and regenerate plots from a saved per-episode CSV.",
    )
    parser.add_argument(
    "--no-plot-titles",
    action="store_true",
    help="Generate plots without titles.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.plot_only:
        # save plots without titles
        plot_saved_results(
            csv_path=args.plot_only,
            plot_root=args.plot_root,
            show_titles=not args.no_plot_titles,
        )
        return

    config = load_config("training_config.json")
    if args.bc_checkpoint is None:
        bc_checkpoint, bc_seed, bc_joint_acc = get_best_bc_checkpoint(config)
        print(f"Auto-selected BC seed {bc_seed} (joint accuracy={bc_joint_acc:.4f})")
    else:
        bc_checkpoint = args.bc_checkpoint

    if args.pure_td3_checkpoint is None:
        raise ValueError(
            "Please provide --pure-td3-checkpoint with the pure-RL TD3 checkpoint "
            "file or directory."
        )

    bc_td3_checkpoint = args.bc_td3_checkpoint or config["td3"]["checkpoint_dir"]

    evaluate_models(
        bc_checkpoint=bc_checkpoint,
        pure_td3_checkpoint=args.pure_td3_checkpoint,
        bc_td3_checkpoint=bc_td3_checkpoint,
        num_episodes=args.num_episodes,
        num_model_peds=args.num_model_peds,
        evaluation_seed=args.seed,
        # no_rendering_mode=not args.render,
        device=args.device,
        output_root=args.output_root,
        plot_root=args.plot_root,
    )


if __name__ == "__main__":
    main()
