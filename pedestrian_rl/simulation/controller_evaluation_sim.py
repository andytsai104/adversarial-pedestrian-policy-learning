import json
import os

from ..models.bc_model import BehaviorCloningPolicy
from ..utils.config_loader import load_config
from ..utils.eval_utils import (
    plot_evaluation_results,
    save_episode_results_csv,
    save_json,
    summarize_episode_results,
)
from ..utils.policy_runners.ai_policy_runner import AIControllerRunner
from ..utils.policy_runners.bc_policy_runner import PolicyRunner
from ..utils.policy_runners.td3_policy_runner import TD3PolicyRunner
from ..utils.td3_utils import PedestrianRLEnv


DEFAULT_NUM_EPISODES = 20
DEFAULT_AI_SEED = 0
DEFAULT_TD3_SEED = 0


def load_json(json_path):
    with open(json_path, "r") as file:
        return json.load(file)


# ---------- checkpoint helpers ----------
def get_best_bc_checkpoint(config):
    checkpoint_root_dir = config["bc"]["checkpoint_dir"]
    summary_path = os.path.join(checkpoint_root_dir, "multi_seed_summary.json")

    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"BC multi-seed summary not found: {summary_path}\n"
            f"Please train BC first or update checkpoint_dir in training_config.json."
        )

    multi_seed_summary = load_json(summary_path)
    seeds = multi_seed_summary.get("seeds", [])
    per_seed = multi_seed_summary.get("per_seed", {})

    if len(seeds) == 0 or len(per_seed) == 0:
        raise ValueError(f"Invalid BC summary file: {summary_path}")

    best_seed_name = None
    best_joint_acc = float("-inf")

    for seed in seeds:
        seed_name = f"seed_{seed}"
        seed_info = per_seed.get(seed_name, None)
        if seed_info is None:
            continue

        joint_acc = float(seed_info.get("joint_accuracy", float("-inf")))
        if joint_acc > best_joint_acc:
            best_joint_acc = joint_acc
            best_seed_name = seed_name

    if best_seed_name is None:
        raise ValueError(f"Could not determine best BC seed from: {summary_path}")

    checkpoint_path = os.path.join(checkpoint_root_dir, best_seed_name, "best_model.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"BC checkpoint not found: {checkpoint_path}")

    best_seed = int(best_seed_name.split("_")[-1])
    return checkpoint_path, best_seed, best_joint_acc


def get_td3_checkpoint(config):
    checkpoint_root_dir = config["td3"]["checkpoint_dir"]

    preferred_paths = [
        os.path.join(checkpoint_root_dir, "td3_last.pt"),
    ]

    for checkpoint_path in preferred_paths:
        if os.path.exists(checkpoint_path):
            return checkpoint_path

    episode_checkpoints = []
    if os.path.isdir(checkpoint_root_dir):
        for filename in os.listdir(checkpoint_root_dir):
            if filename.startswith("td3_episode_") and filename.endswith(".pt"):
                episode_checkpoints.append(os.path.join(checkpoint_root_dir, filename))

    if len(episode_checkpoints) == 0:
        raise FileNotFoundError(
            f"No TD3 checkpoint found in: {checkpoint_root_dir}\n"
            f"Expected td3_last.pt or td3_episode_XXX.pt"
        )

    episode_checkpoints.sort()
    return episode_checkpoints[-1]


# ---------- runners ----------
def run_ai_evaluation(num_episodes, num_model_peds, no_rendering_mode=False, evaluation_seed=DEFAULT_AI_SEED):
    runner = AIControllerRunner(
        sim_config_name="sim_config.json",
        no_rendering_mode=no_rendering_mode,
        num_model_peds=num_model_peds,
        evaluation_seed=evaluation_seed,
    )

    try:
        runner.reset_episode()
        while runner.current_episode_id < num_episodes:
            runner.step_once()
    finally:
        runner.close_evaluators()

    return runner.get_completed_rows()



def run_bc_evaluation(checkpoint_path, bc_seed, num_episodes, num_model_peds, no_rendering_mode=False):
    runner = PolicyRunner(
        model_class=BehaviorCloningPolicy,
        model_name="bc_model",
        checkpoint_path=checkpoint_path,
        training_config_name="training_config.json",
        sim_config_name="sim_config.json",
        no_rendering_mode=no_rendering_mode,
        device="cuda",
        num_model_peds=num_model_peds,
        evaluation=True,
        evaluation_seed=bc_seed,
    )

    try:
        runner.reset_episode()
        while runner.current_episode_id < num_episodes:
            runner.step_once(render_bev=False)
    finally:
        runner.close_target_wrappers()
        runner.close_evaluators()

    return runner.get_completed_rows()



def run_td3_evaluation(checkpoint_path, num_episodes, evaluation_seed=DEFAULT_TD3_SEED, no_rendering_mode=False):
    training_config = load_config("training_config.json")

    env = PedestrianRLEnv(
        sim_config_name="sim_config.json",
        training_config_name="training_config.json",
        no_rendering_mode=no_rendering_mode,
        render_bev=False,
        device="cuda",
    )
    runner = TD3PolicyRunner(
        env=env,
        checkpoint_path=checkpoint_path,
        training_config=training_config,
        evaluation_seed=evaluation_seed,
    )

    try:
        while runner.current_episode_id < num_episodes:
            obs_dict, _ = env.reset()
            evaluators = runner._build_evaluators()

            try:
                while True:
                    if len(obs_dict) == 0:
                        runner._finalize_episode_metrics(evaluators, "no_active_pedestrians")
                        break

                    action_dict = {}
                    for ped_id, obs in obs_dict.items():
                        action_dict[ped_id] = runner.agent.select_action(obs, add_noise=False)

                    next_obs_dict, reward_dict, terminated_dict, truncated_dict, info = env.step(action_dict)

                    for evaluator in evaluators.values():
                        evaluator.update()

                    obs_dict = next_obs_dict

                    if bool(info.get("episode_done", False)):
                        reason = info.get("episode_reason", "episode_done")
                        runner._finalize_episode_metrics(evaluators, reason)
                        break
            finally:
                runner._destroy_evaluators(evaluators)
    finally:
        env.close()

    return runner.get_completed_rows()


# ---------- main orchestration ----------
def evaluate_all_controllers(num_episodes=DEFAULT_NUM_EPISODES, no_rendering_mode=False):
    config = load_config("training_config.json")

    td3_num_model_peds = int(config["td3"]["params"].get("num_model_peds", 1))
    eval_root = os.path.join("checkpoints", "evaluation", "controller_comparison")
    os.makedirs(eval_root, exist_ok=True)

    bc_checkpoint_path, bc_seed, bc_joint_acc = get_best_bc_checkpoint(config)
    td3_checkpoint_path = get_td3_checkpoint(config)

    print("=" * 80)
    print("Controller evaluation setup")
    print(f"BC checkpoint : {bc_checkpoint_path}")
    print(f"BC best seed  : {bc_seed}")
    print(f"BC joint acc  : {bc_joint_acc:.4f}")
    print(f"TD3 checkpoint: {td3_checkpoint_path}")
    print(f"Episodes/controller: {num_episodes}")
    print(f"Tracked pedestrians/controller episode: {td3_num_model_peds}")
    print("=" * 80)

    all_rows = []

    print("\n[1/3] Evaluating CARLA AI controller...")
    ai_rows = run_ai_evaluation(
        num_episodes=num_episodes,
        num_model_peds=td3_num_model_peds,
        no_rendering_mode=no_rendering_mode,
        evaluation_seed=DEFAULT_AI_SEED,
    )
    all_rows.extend(ai_rows)
    print(f"[AI] Collected {len(ai_rows)} rows.")

    print("\n[2/3] Evaluating BC controller...")
    bc_rows = run_bc_evaluation(
        checkpoint_path=bc_checkpoint_path,
        bc_seed=bc_seed,
        num_episodes=num_episodes,
        num_model_peds=td3_num_model_peds,
        no_rendering_mode=no_rendering_mode,
    )
    all_rows.extend(bc_rows)
    print(f"[BC] Collected {len(bc_rows)} rows.")

    print("\n[3/3] Evaluating TD3 controller...")
    td3_rows = run_td3_evaluation(
        checkpoint_path=td3_checkpoint_path,
        num_episodes=num_episodes,
        evaluation_seed=DEFAULT_TD3_SEED,
        no_rendering_mode=no_rendering_mode,
    )
    all_rows.extend(td3_rows)
    print(f"[TD3] Collected {len(td3_rows)} rows.")

    csv_path = os.path.join(eval_root, "controller_evaluation_rows.csv")
    summary_path = os.path.join(eval_root, "controller_evaluation_summary.json")
    plots_dir = os.path.join(eval_root, "plots")

    save_episode_results_csv(all_rows, csv_path)
    summary = summarize_episode_results(all_rows)
    summary["metadata"] = {
        "num_episodes_per_controller": int(num_episodes),
        "tracked_pedestrians_per_episode": int(td3_num_model_peds),
        "bc_checkpoint_path": bc_checkpoint_path,
        "bc_seed": int(bc_seed),
        "td3_checkpoint_path": td3_checkpoint_path,
    }
    save_json(summary, summary_path)
    plot_evaluation_results(all_rows, plots_dir)

    print("\nEvaluation finished.")
    print(f"Rows CSV : {csv_path}")
    print(f"Summary  : {summary_path}")
    print(f"Plots    : {plots_dir}")

    return all_rows, summary


if __name__ == "__main__":
    evaluate_all_controllers(
        num_episodes=DEFAULT_NUM_EPISODES,
        no_rendering_mode=False,
    )
