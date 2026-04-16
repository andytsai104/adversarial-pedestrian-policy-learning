from ..config_loader import load_config
from ..eval_utils import EpisodeEvaluator
from ..td3_utils import PedestrianRLEnv, build_td3_agent


# --- TD3 Policy runner ---
class TD3PolicyRunner:
    '''Run and evaluate a trained TD3 policy with the current multi-ped env API.''' 

    def __init__(self, env, checkpoint_path, training_config, evaluation_seed=0):
        self.env = env
        self.training_config = training_config
        self.evaluation_seed = int(evaluation_seed)
        self.current_episode_id = 0
        self.completed_episode_rows = []

        self.agent = build_td3_agent(
            training_config=training_config,
            max_speed=env.max_ped_speed,
            device=env.device,
        )
        self.agent.load(checkpoint_path=checkpoint_path, load_optimizers=False)
        print(f"[TD3PolicyRunner] Loaded checkpoint: {checkpoint_path}")

    def _build_evaluators(self):
        evaluators = {}
        for ped_id, ped in self.env.target_peds.items():
            evaluators[ped_id] = EpisodeEvaluator(
                world=self.env.world,
                target_ped=ped,
                dt=self.env.fixed_delta_seconds,
            )
        return evaluators

    @staticmethod
    def _destroy_evaluators(evaluators):
        for evaluator in evaluators.values():
            evaluator.destroy()

    def _finalize_episode_metrics(self, evaluators, reason):
        rows = []
        for ped_id, evaluator in evaluators.items():
            row = evaluator.get_metrics(
                controller_name="rl_model",
                seed=self.evaluation_seed,
                episode_id=self.current_episode_id,
                ped_id=ped_id,
            )
            # row["term_reason"] = reason
            rows.append(row)

        self.completed_episode_rows.extend(rows)
        print(f"[TD3PolicyRunner] Finalized episode {self.current_episode_id} with {len(rows)} rows. reason={reason}")
        self.current_episode_id += 1
        return rows

    def get_completed_rows(self):
        return list(self.completed_episode_rows)

    def run(self):
        while True:
            obs_dict, _ = self.env.reset()
            evaluators = self._build_evaluators()

            try:
                while True:
                    if len(obs_dict) == 0:
                        reason = "no_active_pedestrians"
                        self._finalize_episode_metrics(evaluators, reason)
                        break

                    action_dict = {}
                    for ped_id, obs in obs_dict.items():
                        action_dict[ped_id] = self.agent.select_action(obs, add_noise=False)

                    next_obs_dict, reward_dict, terminated_dict, truncated_dict, info = self.env.step(action_dict)

                    for evaluator in evaluators.values():
                        evaluator.update()

                    first_ped_id = next(iter(info["ped_info"].keys())) if len(info.get("ped_info", {})) > 0 else None
                    if first_ped_id is not None:
                        ped_step_info = info["ped_info"][first_ped_id]
                        print(
                            f"[TD3 Run] step={info['episode_step']} "
                            f"reward={reward_dict.get(first_ped_id, 0.0):.4f} "
                            f"goal_distance={ped_step_info.get('goal_distance', None)} "
                            f"min_vehicle_distance={ped_step_info.get('min_vehicle_distance', None)}"
                        )

                    obs_dict = next_obs_dict

                    if bool(info.get("episode_done", False)):
                        reason = info.get("episode_reason", "episode_done")
                        self._finalize_episode_metrics(evaluators, reason)
                        break
            finally:
                self._destroy_evaluators(evaluators)


def run_td3_policy(checkpoint_path):
    '''Run trained TD3 policy in CARLA.''' 
    training_config = load_config("training_config.json")

    env = PedestrianRLEnv(
        sim_config_name="sim_config.json",
        training_config_name="training_config.json",
        no_rendering_mode=False,
        render_bev=True,
        device="cuda",
    )
    runner = TD3PolicyRunner(
        env=env,
        checkpoint_path=checkpoint_path,
        training_config=training_config,
    )

    try:
        runner.run()
    except KeyboardInterrupt:
        print("\n[TD3 Run] Stopped by user.")
    finally:
        env.close()