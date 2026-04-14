import carla
import os
import time
import cv2
from ..utils.config_loader import load_config
from ..utils.sim_utils import Spector, CrossroadPedestrians, AggressiveVehicles, refresh_sim, spawn_actors
from ..data_collection.bev.bev_sample import BEVWrapper
from ..data_collection.bev.bev_seg_sample import SemanticBEVWrapper, SemanticBEVSample
from ..utils.data_utils import DataSampler, convert_to_dataset

'''TODO: Increase dataset quality
Create a logic that sample pedestrians in the certain area only or every n steps (in the intersection)
'''

def data_sampling_sim(output_file=True, no_rendering_mode=True, show_bev=False, print_out_data=False):
    # ----- connect to CARLA -----
    config = load_config("sim_config.json")
    num_episode = config["dataset"]["num_episode"]
    num_ped_per_episode = config["dataset"]["num_ped_per_episode"]
    sample_every_n_steps = config["dataset"]["sample_every_n_steps"]
    min_samples_per_episode = config["dataset"]["min_samples_per_episode"]
    sim_config = config["simulation"]
    fixed_delta_time = sim_config["fixed_delta_seconds"]
    warmup_ticks = sim_config["warmup_ticks"]


    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = fixed_delta_time
    settings.no_rendering_mode = no_rendering_mode
    world.apply_settings(settings)

    # ----- setup scenario -----
    intersection_position = carla.Location(
        x=sim_config["intersection"]["x"],
        y=sim_config["intersection"]["y"],
        z=sim_config["intersection"]["z"],
    )
    distance = sim_config["intersection"]["dist"]

    spector = Spector(world, location=intersection_position + carla.Location(z=50), dist=distance)
    aggressive_vehicles = AggressiveVehicles(client, world, location=intersection_position)
    crossroad_pedestrians = CrossroadPedestrians(world, location=intersection_position)
    # bev_wrapper = BEVWrapper(cfg=None, world=world)
    bev_wrapper = SemanticBEVWrapper(cfg=None, world=world)

    sampler = DataSampler(
        world=world,
        bev_wrapper=bev_wrapper,
        crossroad_pedestrians=crossroad_pedestrians,
        config=config,
        bev_sample_class=SemanticBEVSample
    )

    # ----- refresh conditions -----
    stuck_detection_config = config["simulation"]["stuck_detection"]
    refresh_conditions = {
        "time_out": stuck_detection_config["time_out"],
        "start time": world.get_snapshot().timestamp.elapsed_seconds,
        "vehicle": {
            "velocity_threshold": stuck_detection_config["vehicle"]["velocity_threshold"],
            "stuck_tracker": {},
            "stuck_time_limit": stuck_detection_config["vehicle"]["stuck_time_limit"],
            "stuck_count_limit": stuck_detection_config["vehicle"]["stuck_count_limit"],
        },
        "pedestrian": {
            "dist": stuck_detection_config["pedestrian"]["dist"],
            "min_peds": stuck_detection_config["pedestrian"]["min_pedestrians"],
        },
    }

    # ----- output path -----
    output_path = os.path.join(
        config["dataset"]["save_path"],
        config["dataset"]["file_name"]
    )

    episode_idx = 0

    # ----- initial spawn -----
    spawn_actors(
        world=world,
        spector=spector,
        aggressive_vehicles=aggressive_vehicles,
        crossroad_pedestrians=crossroad_pedestrians,
    )
    sampler.select_target_ped_ids()
    sampler.attach_target_wrappers()

    for _ in range(warmup_ticks):
        world.tick()

    def respawn_same_episode(reason):
        print(f"--- RESAMPLE SAME EPISODE: {reason} ---")
        print("Discarding current episode buffer and respawning actors...")

        refresh_conditions["start time"] = world.get_snapshot().timestamp.elapsed_seconds
        refresh_conditions["vehicle"]["stuck_tracker"] = {}

        sampler.close_target_wrappers()
        sampler.reset_episode_tracking()

        spawn_actors(
            world=world,
            spector=spector,
            aggressive_vehicles=aggressive_vehicles,
            crossroad_pedestrians=crossroad_pedestrians,
        )
        sampler.select_target_ped_ids()
        sampler.attach_target_wrappers()

        for _ in range(warmup_ticks):
            world.tick()

    try:
        sim_step = 0
        while True:
            world.tick()
            sim_step+=1

            if episode_idx == num_episode:
                print(f"\nSampling finished: collected {episode_idx} episodes.")
                print(f"Dataset saved to: {output_path}")
                break

            # ----- refresh episode if needed -----
            sim_state, should_refresh = refresh_sim(
                world=world,
                refresh_conditions=refresh_conditions,
                intersection_position=intersection_position
            )

            if should_refresh:
                print(f"--- {sim_state} ---")
                print("Refreshing simulation...")

                if output_file:
                    episode_data = sampler.get_episode_buffer()
                    sample_counts = len(episode_data)
                    if sample_counts < min_samples_per_episode:
                        respawn_same_episode(reason=f" only {sample_counts} samples (< {min_samples_per_episode})")
                        continue

                    convert_to_dataset(
                        episode_idx=episode_idx,
                        episode_data=episode_data,
                        output_path=output_path
                    )

                episode_idx += 1

                refresh_conditions["start time"] = world.get_snapshot().timestamp.elapsed_seconds
                refresh_conditions["vehicle"]["stuck_tracker"] = {}

                sampler.close_target_wrappers()
                sampler.reset_episode_tracking()

                spawn_actors(
                    world=world,
                    spector=spector,
                    aggressive_vehicles=aggressive_vehicles,
                    crossroad_pedestrians=crossroad_pedestrians,
                )

                sampler.select_target_ped_ids()
                sampler.attach_target_wrappers()
                for _ in range(warmup_ticks):
                    world.tick()
                continue
            
            if sim_step % sample_every_n_steps != 0:
                continue

            # ----- sample current frame -----

            sample_peds = sampler.get_sample_pedestrians()
            num_sample_peds = len(sample_peds)

            # Check if the peds in sample_peds meet the required number
            if num_sample_peds < num_ped_per_episode:
                respawn_same_episode(reason=f"target pedestrian missing ({num_sample_peds}/{num_ped_per_episode} alive)")
                continue
            

            # Try to sample the bev if the pedestrian is alive; or resample the current episode
            try:
                for ped in sample_peds:
                    ped_info, bev_sample = sampler.sample_single_pedestrian(ped=ped)
                    sampler.append_sample(ped_info)
            except RuntimeError as exc:
                respawn_same_episode(reason=exc)
                continue
            
            frame_id = ped_info.state_action_pair["frame_id"]

            # Visualize the last pedestrian
            if (print_out_data):
                print(
                    f"\n[Ped Sample] "
                    f"ped_id={ped_info.ped_id} | frame={frame_id}\n"
                    f"  location         : {ped_info.state['current_location']}\n"
                    f"  velocity         : {ped_info.state['velocity']}\n"
                    # f"  speed            : {ped_info.state['speed']}\n"
                    # f"  motion_heading   : {ped_info.state['motion_heading']}\n"
                    f"  yaw_heading   : {ped_info.state['yaw_heading']}\n"
                    # f"  target_speed     : {ped_info.action['target_speed']}\n"
                    f"  target_direction : {ped_info.action['target_direction']}\n"
                )

            if (show_bev):
                try: 
                    image = bev_sample.visualize_bev()
                except RuntimeError as exc:
                    respawn_same_episode(reason=exc)
                    continue
                cv2.imshow("BEV Debug Tool", image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    return
    finally:
        sampler.close_target_wrappers()
        cv2.destroyAllWindows()
        

def visualize_sampled_data():
    data_sampling_sim(
        output_file=False,
        no_rendering_mode=False,
        show_bev=True,
        print_out_data=True    
    )

if __name__ == "__main__":

    try:
        # Test the functionality of data sampling
        # visualize_sampled_data()

        # Output dataset
        data_sampling_sim(
            output_file=True,
            no_rendering_mode=False,
            show_bev=True,
            print_out_data=False   
        )
    except KeyboardInterrupt:
        print("Sampling stopped by user.")
    finally:
        cv2.destroyAllWindows()
