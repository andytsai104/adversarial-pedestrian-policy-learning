from pedestrian_rl.utils.td3_utils import PedestrianRLEnv, TD3Trainer, build_td3_agent, load_bc_weights_into_td3_actor
from ..utils.config_loader import load_config


def train_td3():
    '''Train TD3 pedestrian policy.''' 
    training_config = load_config('training_config.json')

    env = PedestrianRLEnv(
        sim_config_name='sim_config.json',
        training_config_name='training_config.json',
        no_rendering_mode=False,
        render_bev=True,
        device='cuda',
    )
    agent = build_td3_agent(
        training_config=training_config,
        max_speed=env.max_ped_speed,
        device=env.device,
    )

    td3_params = training_config["td3"]
    if td3_params["initialize_actor_from_bc"]:
        load_bc_weights_into_td3_actor(
            td3_agent=agent,
            bc_checkpoint_path=td3_params["bc_init_checkpoint"],
            device=env.device,
        )

    trainer = TD3Trainer(
        env=env,
        agent=agent,
        training_config=training_config,
    )

    try:
        trainer.train()
    except KeyboardInterrupt:
        print('\n[TD3 Train] Stopped by user.')
    finally:
        env.close()


if __name__ == '__main__':
    train_td3()
