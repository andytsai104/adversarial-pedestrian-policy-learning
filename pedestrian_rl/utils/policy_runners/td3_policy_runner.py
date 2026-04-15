from ..config_loader import load_config
from ..td3_utils import PedestrianRLEnv, build_td3_agent


# --- TD3 Policy runner ---
class TD3PolicyRunner:
    '''Run one trained TD3 policy in CARLA.''' 

    def __init__(self, env, checkpoint_path, training_config):
        self.env = env
        self.training_config = training_config
        self.agent = build_td3_agent(training_config=training_config, max_speed=env.max_ped_speed, device=env.device)
        self.agent.load(checkpoint_path=checkpoint_path, load_optimizers=False)
        print(f"[TD3PolicyRunner] Loaded checkpoint: {checkpoint_path}")

    def run(self):
        '''Run trained TD3 policy without exploration noise.''' 
        obs, _ = self.env.reset()

        while True:
            action = self.agent.select_action(obs, add_noise=False)
            obs, reward, terminated, truncated, info = self.env.step(action)

            print(
                f"[TD3 Run] step={info['episode_step']} "
                f"reward={reward:.4f} "
                f"goal_distance={info['goal_distance']} "
                f"min_vehicle_distance={info['min_vehicle_distance']}"
            )

            if terminated or truncated:
                print(f"[TD3 Run] Episode ended: {info['term_reason']}")
                obs, _ = self.env.reset()


def run_td3_policy(checkpoint_path):
    '''Run trained TD3 policy in CARLA.''' 
    training_config = load_config('training_config.json')

    env = PedestrianRLEnv(
        sim_config_name='sim_config.json',
        training_config_name='training_config.json',
        no_rendering_mode=False,
        render_bev=True,
        device='cuda',
    )
    runner = TD3PolicyRunner(
        env=env,
        checkpoint_path=checkpoint_path,
        training_config=training_config,
    )

    try:
        runner.run()
    except KeyboardInterrupt:
        print('\n[TD3 Run] Stopped by user.')
    finally:
        env.close()