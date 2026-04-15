import os
import cv2
from ..utils.config_loader import load_config
from ..utils.policy_runners.bc_policy_runner import PolicyRunner
from ..models.bc_model import BehaviorCloningPolicy


def main():
    config_name = "training_config.json"
    config = load_config(config_name=config_name)
    checkpoint_root_dir = config["bc"]["checkpoint_dir"]

    summary_name = "multi_seed_summary.json"
    multi_seed_summary = load_config(config_name=summary_name, config_path=checkpoint_root_dir)
    

    # --- Get best checkpoint from multi_seed_summary.json ---
    seeds_names = [f"seed_{seed_i}" for seed_i in multi_seed_summary["seeds"]]
    per_seed = multi_seed_summary["per_seed"]
    best_seed = seeds_names[0]
    for seed in seeds_names:
        if per_seed[seed]["joint_accuracy"] > per_seed[best_seed]["joint_accuracy"]:
            best_seed = seed


    checkpoint_name = "best_model.pt"
    checkpoint_seed_dir = os.path.join(best_seed)
    checkpoint_path = os.path.join(checkpoint_root_dir, checkpoint_seed_dir, checkpoint_name)

    model_name = "BC"
    runner = PolicyRunner(
        model_class=BehaviorCloningPolicy,
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        num_model_peds=1,
    )

    try:
        runner.run(render_bev=True)
    except KeyboardInterrupt:
        print(f"\n[{model_name} PolicyRunner] Stopped by user.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
