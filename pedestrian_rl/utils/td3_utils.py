import json
import math
import os
import random
import weakref
from matplotlib import pyplot as plt
import carla
import cv2
import numpy as np

from ..data_collection.bev.bev_sample import BEVSample, BEVWrapper
from .data_utils import rotate_local_to_world_2d, rotate_world_to_local_2d
from ..models.td3_model import TD3Agent
from .config_loader import load_config
from .sim_utils import (
    AggressiveVehicles,
    CrossroadPedestrians,
    Spector,
    cleanup_simulation,
    refresh_sim,
)

# --- Environment for TD3 to train in ---
class PedestrianRLEnv:
    '''
    RL environment for one TD3-controlled pedestrian in CARLA.

    Observation matches the BC policy input style:
        bev_data
        velocity_local
        speed
        yaw_sin
        yaw_cos
        goal_rel_local

    Action format:
        [target_speed, dir_right, dir_forward]
    '''

    def __init__(
        self,
        sim_config_name="sim_config.json",
        training_config_name="training_config.json",
        no_rendering_mode=True,
        render_bev=False,
        device="cuda",
    ):
        # ----- load config -----
        self.sim_config = load_config(sim_config_name)
        self.training_config = load_config(training_config_name)

        sim_cfg = self.sim_config["simulation"]
        td3_cfg = self.training_config["td3"]
        td3_params = td3_cfg["params"]

        self.fixed_delta_seconds = sim_cfg["fixed_delta_seconds"]
        self.max_episode_steps = sim_cfg["max_episode_steps"]
        self.warmup_ticks = sim_cfg["warmup_ticks"]
        self.max_ped_speed = sim_cfg["pedestrian"]["speed_range"][1]

        self.goal_scale = float(td3_params["goal_scale"])
        self.num_background_pedestrians = int(td3_params["num_background_pedestrians"])
        self.stall_speed_threshold = float(td3_params["stall_speed_threshold"])
        self.goal_reached_threshold = float(td3_params["goal_reached_threshold"])
        self.clip_bound = float(td3_params["clip_bound"])
        self.render_bev = bool(render_bev)
        self.device = device

        self.reward_weight = td3_cfg["reward"]

        # ----- connect to CARLA -----
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.world_map = self.world.get_map()

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.fixed_delta_seconds
        settings.no_rendering_mode = no_rendering_mode
        self.world.apply_settings(settings)

        # ----- setup scenario -----
        self.intersection_position = carla.Location(
            x=sim_cfg["intersection"]["x"],
            y=sim_cfg["intersection"]["y"],
            z=sim_cfg["intersection"]["z"],
        )
        self.distance = sim_cfg["intersection"]["dist"]

        self.spector = Spector(
            self.world,
            location=self.intersection_position + carla.Location(z=50),
            dist=self.distance,
        )
        self.aggressive_vehicles = AggressiveVehicles(
            self.client,
            self.world,
            location=self.intersection_position,
        )
        self.crossroad_pedestrians = CrossroadPedestrians(
            self.world,
            location=self.intersection_position,
        )
        self.bev_wrapper = BEVWrapper(cfg=None, world=self.world)

        stuck_detection_cfg = sim_cfg["stuck_detection"]
        self.refresh_conditions = {
            "time_out": stuck_detection_cfg["time_out"],
            "start time": self.world.get_snapshot().timestamp.elapsed_seconds,
            "vehicle": {
                "velocity_threshold": stuck_detection_cfg["vehicle"]["velocity_threshold"],
                "stuck_tracker": {},
                "stuck_time_limit": stuck_detection_cfg["vehicle"]["stuck_time_limit"],
                "stuck_count_limit": stuck_detection_cfg["vehicle"]["stuck_count_limit"],
            },
            "pedestrian": {
                "dist": stuck_detection_cfg["pedestrian"]["dist"],
                "min_peds": stuck_detection_cfg["pedestrian"]["min_pedestrians"],
            },
        }

        # ----- reset-able buffers -----
        self.target_ped = None
        self.target_goal = None
        self.prev_location = None
        self.prev_frame_id = None
        self.prev_goal_distance = None
        self.prev_min_vehicle_distance = None
        self.episode_step = 0
        self.last_collision = False
        self.last_bev_sample = None
        self.last_obs = None
        self.collision_sensor = None

    def attach_collision_sensor(self):
        '''Attach one collision sensor to the target pedestrian.'''
        blueprint = self.world.get_blueprint_library().find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(
            blueprint,
            carla.Transform(),
            attach_to=self.target_ped,
        )

        weak_self = weakref.ref(self)
        self.collision_sensor.listen(
            lambda event: PedestrianRLEnv.on_collision(weak_self, event)
        )

    @staticmethod
    def on_collision(weak_self, event):
        '''Collision callback: only count vehicle collisions.'''
        self = weak_self()
        if self is None:
            return

        if event.other_actor is not None and "vehicle." in event.other_actor.type_id:
            self.last_collision = True

    def destroy_collision_sensor(self):
        '''Destroy collision sensor safely.'''
        if self.collision_sensor is not None and self.collision_sensor.is_alive:
            self.collision_sensor.destroy()
        self.collision_sensor = None

    def close(self):
        '''Close environment resources.'''
        self.destroy_collision_sensor()
        cv2.destroyAllWindows()

    @staticmethod
    def normalize_direction(direction_local, eps=1e-6):
        '''Normalize local-frame 2D direction vectors.'''
        norm = float(np.linalg.norm(direction_local))
        if norm < eps:
            return np.array([0.0, 1.0], dtype=np.float32)
        return (direction_local / norm).astype(np.float32)

    def compute_velocity_speed(self, ped, frame_id):
        '''Compute current location, velocity, and speed.'''
        loc = ped.get_location()
        current_location = np.array([loc.x, loc.y, loc.z], dtype=np.float32)

        if self.prev_location is None or self.prev_frame_id is None:
            velocity = np.zeros(3, dtype=np.float32)
        else:
            passed_frames = frame_id - self.prev_frame_id
            dt = passed_frames * self.fixed_delta_seconds
            if dt > 0:
                velocity = (current_location - self.prev_location) / dt
            else:
                velocity = np.zeros(3, dtype=np.float32)

        speed = float(np.linalg.norm(velocity[:2]))
        return current_location, velocity, speed

    def get_goal_distance(self, current_location):
        '''Get target pedestrian distance to the goal.'''
        if self.target_goal is None:
            return 0.0
        return float(np.linalg.norm(self.target_goal[:2] - current_location[:2]))

    def get_min_vehicle_distance(self, ped_location):
        '''Get minimum distance between target pedestrian and all vehicles.'''
        vehicles = [veh for veh in self.world.get_actors().filter("vehicle.*") if veh.is_alive]
        if len(vehicles) == 0:
            return 999.0

        return float(min(veh.get_location().distance(ped_location) for veh in vehicles))

    def is_on_driving_lane(self, ped_location):
        '''Check if pedestrian is on drivable lane.'''
        wp = self.world_map.get_waypoint(
            ped_location,
            project_to_road=False,
            lane_type=carla.LaneType.Any,
        )
        return bool(wp is not None and wp.lane_type == carla.LaneType.Driving)

    def build_observation(self):
        '''Build one RL observation from current CARLA state.'''
        frame_id = self.world.get_snapshot().timestamp.frame
        self.last_bev_sample = BEVSample(actor=self.target_ped, bev_wrapper=self.bev_wrapper)
        bev_data = self.last_bev_sample.get_bev().astype(np.float32)

        current_location, velocity, speed = self.compute_velocity_speed(self.target_ped, frame_id)
        yaw_heading = math.radians(self.target_ped.get_transform().rotation.yaw)

        goal_rel_world = self.target_goal - current_location
        velocity_local = rotate_world_to_local_2d(velocity[:2], yaw_heading).astype(np.float32)
        goal_rel_local = rotate_world_to_local_2d(goal_rel_world[:2], yaw_heading)
        goal_rel_local = goal_rel_local / max(self.goal_scale, 1e-6)
        goal_rel_local = np.clip(goal_rel_local, -self.clip_bound, self.clip_bound).astype(np.float32)

        obs = {
            "bev_data": bev_data,
            "velocity_local": velocity_local,
            "goal_rel_local": goal_rel_local,
            "yaw_sin": np.float32(math.sin(yaw_heading)),
            "yaw_cos": np.float32(math.cos(yaw_heading)),
            "speed": np.float32(speed),
        }

        debug_state = {
            "frame_id": frame_id,
            "current_location": current_location,
            "velocity": velocity,
            "speed": speed,
            "yaw_heading": yaw_heading,
            "goal_distance": self.get_goal_distance(current_location),
            "min_vehicle_distance": self.get_min_vehicle_distance(self.target_ped.get_location()),
        }
        return obs, debug_state

    def apply_action(self, action):
        '''Apply one RL action to the target pedestrian.'''
        action = np.asarray(action, dtype=np.float32)
        target_speed = float(np.clip(action[0], 0.0, self.max_ped_speed))
        direction_local = self.normalize_direction(action[1:3])

        yaw_heading = math.radians(self.target_ped.get_transform().rotation.yaw)
        direction_world_xy = rotate_local_to_world_2d(direction_local, yaw_heading)

        control = carla.WalkerControl()
        control.speed = target_speed
        control.jump = False
        control.direction = carla.Vector3D(
            x=float(direction_world_xy[0]),
            y=float(direction_world_xy[1]),
            z=0.0,
        )
        self.target_ped.apply_control(control)

        return np.array([target_speed, direction_local[0], direction_local[1]], dtype=np.float32)

    def compute_reward(self, current_location, speed, ped_location):
        '''Compute reward and reward terms for one step.'''
        reward = 0.0
        reward_terms = {}

        goal_distance = self.get_goal_distance(current_location)
        min_vehicle_distance = self.get_min_vehicle_distance(ped_location)
        on_driving_lane = self.is_on_driving_lane(ped_location)

        # Collision reward
        if self.last_collision:
            reward_terms["collision"] = self.reward_weight["collision"]
            reward += reward_terms["collision"]
        else:
            reward_terms["collision"] = 0.0

        # Reward when approaching vehicles
        if self.prev_min_vehicle_distance is None:
            approach_delta = 0.0
        else:
            approach_delta = self.prev_min_vehicle_distance - min_vehicle_distance
        reward_terms["approach_vehicle"] = self.reward_weight["approach_vehicle"] * float(np.clip(approach_delta, -1.0, 1.0))
        reward += reward_terms["approach_vehicle"]

        # Small goal progress reward to prevent meaningless wandering
        if self.prev_goal_distance is None:
            goal_progress = 0.0
        else:
            goal_progress = self.prev_goal_distance - goal_distance
        reward_terms["goal_progress"] = self.reward_weight["goal_progress"] * float(np.clip(goal_progress, -1.0, 1.0))
        reward += reward_terms["goal_progress"]

        # Stall penalty
        if speed < self.stall_speed_threshold:
            reward_terms["stall"] = self.reward_weight["stall"]
            reward += reward_terms["stall"]
        else:
            reward_terms["stall"] = 0.0

        # Small per-step reward / penalty
        reward_terms["living"] = self.reward_weight["living"]
        reward += reward_terms["living"]

        # Drivable-lane related term
        if on_driving_lane:
            reward_terms["lane"] = self.reward_weight["on_driving"]
            reward += reward_terms["lane"]
        else:
            reward_terms["lane"] = self.reward_weight["off_road"]
            reward += reward_terms["lane"]

        # Goal reached bonus
        goal_reached = goal_distance < self.goal_reached_threshold
        if goal_reached:
            reward_terms["goal_reached"] = self.reward_weight["goal_reached"]
            reward += reward_terms["goal_reached"]
        else:
            reward_terms["goal_reached"] = 0.0

        self.prev_goal_distance = goal_distance
        self.prev_min_vehicle_distance = min_vehicle_distance

        extra_state = {
            "goal_distance": goal_distance,
            "min_vehicle_distance": min_vehicle_distance,
            "on_driving_lane": on_driving_lane,
            "goal_reached": goal_reached,
        }
        return reward, reward_terms, extra_state

    def spawn_target_and_background(self):
        '''Spawn one RL-controlled pedestrian and background AI pedestrians.'''
        self.target_ped = None
        self.target_goal = None

        spawn_points = self.crossroad_pedestrians.get_ped_spawn_points(
            self.crossroad_pedestrians.ped_num,
            self.crossroad_pedestrians.in_intersection,
        )
        random.shuffle(spawn_points)

        # ----- spawn target pedestrian first -----
        while len(spawn_points) > 0 and self.target_ped is None:
            spawn_location = spawn_points.pop()
            destination = self.world.get_random_location_from_navigation()

            if destination is None:
                continue
            destination.z += 1.0

            ped = self.crossroad_pedestrians.spawn_single_walker(
                spawn_location=spawn_location,
                destination=destination,
                controller="manual",
                max_speed=self.max_ped_speed,
            )

            if ped is None:
                continue

            self.target_ped = ped
            self.target_goal = np.asarray(
                self.crossroad_pedestrians.ped_goal_loc[ped.id],
                dtype=np.float32,
            )

        if self.target_ped is None:
            raise RuntimeError("Failed to spawn RL-controlled pedestrian.")

        # ----- spawn background pedestrians -----
        max_background = max(self.crossroad_pedestrians.ped_num - 1, 0)
        background_target = min(self.num_background_pedestrians, max_background)

        background_spawned = 0
        while background_spawned < background_target and len(spawn_points) > 0:
            spawn_location = spawn_points.pop()
            destination = self.world.get_random_location_from_navigation()

            if destination is None:
                continue
            destination.z += 1.0

            ped = self.crossroad_pedestrians.spawn_single_walker(
                spawn_location=spawn_location,
                destination=destination,
                controller="ai",
            )

            if ped is not None:
                background_spawned += 1

    def reset(self):
        '''Reset one RL episode.'''
        self.destroy_collision_sensor()
        cleanup_simulation(self.world)
        self.crossroad_pedestrians.reset_pedestrians()

        self.spector.set_spector()
        self.aggressive_vehicles.aggressive_vehicles_spawn()
        self.spawn_target_and_background()

        if self.warmup_ticks > 0:
            for _ in range(self.warmup_ticks):
                self.world.tick()

        self.attach_collision_sensor()
        self.refresh_conditions["start time"] = self.world.get_snapshot().timestamp.elapsed_seconds
        self.refresh_conditions["vehicle"]["stuck_tracker"] = {}

        self.prev_location = None
        self.prev_frame_id = None
        self.prev_goal_distance = None
        self.prev_min_vehicle_distance = None
        self.episode_step = 0
        self.last_collision = False

        obs, debug_state = self.build_observation()
        self.last_obs = obs

        self.prev_location = debug_state["current_location"].copy()
        self.prev_frame_id = debug_state["frame_id"]
        self.prev_goal_distance = debug_state["goal_distance"]
        self.prev_min_vehicle_distance = debug_state["min_vehicle_distance"]

        info = {
            "episode_step": self.episode_step,
            "reset": True,
            "ped_id": self.target_ped.id,
            "goal_distance": debug_state["goal_distance"],
            "min_vehicle_distance": debug_state["min_vehicle_distance"],
        }
        return obs, info

    def step(self, action):
        '''Run one environment step.'''
        applied_action = self.apply_action(action)
        self.world.tick()
        self.episode_step += 1

        sim_state, should_refresh = refresh_sim(
            world=self.world,
            refresh_conditions=self.refresh_conditions,
            intersection_position=self.intersection_position,
        )

        terminated = False
        truncated = False
        term_reason = None

        if self.target_ped is None or not self.target_ped.is_alive:
            truncated = True
            term_reason = "pedestrian_missing"
            info = {
                "episode_step": self.episode_step,
                "ped_id": None,
                "action": applied_action,
                "reward_terms": {},
                "collision": self.last_collision,
                "goal_distance": None,
                "min_vehicle_distance": None,
                "on_driving_lane": None,
                "terminated": terminated,
                "truncated": truncated,
                "term_reason": term_reason,
            }
            return self.last_obs, 0.0, terminated, truncated, info

        obs, debug_state = self.build_observation()
        self.last_obs = obs

        current_location = debug_state["current_location"]
        speed = debug_state["speed"]
        ped_location = self.target_ped.get_location()

        reward, reward_terms, extra_state = self.compute_reward(current_location, speed, ped_location)

        if self.last_collision:
            terminated = True
            term_reason = "vehicle_collision"
        elif extra_state["goal_reached"]:
            terminated = True
            term_reason = "goal_reached"
        elif should_refresh:
            truncated = True
            term_reason = sim_state
        elif self.episode_step >= self.max_episode_steps:
            truncated = True
            term_reason = "max_episode_steps"

        self.prev_location = current_location.copy()
        self.prev_frame_id = debug_state["frame_id"]

        info = {
            "episode_step": self.episode_step,
            "ped_id": self.target_ped.id,
            "action": applied_action,
            "reward_terms": reward_terms,
            "collision": self.last_collision,
            "goal_distance": extra_state["goal_distance"],
            "min_vehicle_distance": extra_state["min_vehicle_distance"],
            "on_driving_lane": extra_state["on_driving_lane"],
            "terminated": terminated,
            "truncated": truncated,
            "term_reason": term_reason,
        }

        if self.render_bev and self.last_bev_sample is not None:
            image = self.last_bev_sample.visualize_bev()
            cv2.imshow("TD3 RL BEV", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

        return obs, float(reward), terminated, truncated, info

    def sample_random_action(self):
        '''Sample one random continuous action.'''
        speed = np.float32(np.random.uniform(0.0, self.max_ped_speed))
        theta = np.random.uniform(-math.pi, math.pi)
        direction = np.array([math.sin(theta), math.cos(theta)], dtype=np.float32)
        return np.array([speed, direction[0], direction[1]], dtype=np.float32)


# --- TD3 Trainer ---
class TD3Trainer:
    '''Train TD3 online in CARLA.''' 

    def __init__(self, env, agent, training_config):
        self.env = env
        self.agent = agent
        self.training_config = training_config

        td3_cfg = training_config['td3']
        params = td3_cfg['params']

        self.checkpoint_dir = td3_cfg['checkpoint_dir']
        self.media_dir = td3_cfg['media_dir']

        self.num_episodes = int(params['num_episodes'])
        self.batch_size = int(params['batch_size'])
        self.start_steps = int(params['start_steps'])
        self.updates_per_step = int(params['updates_per_step'])
        self.save_every = int(params['save_every'])
        self.episode_smooth_window = int(params.get('episode_smooth_window', 10))
        self.loss_smooth_window = int(params.get('loss_smooth_window', 10))
        self.rate_smooth_window = int(params.get('rate_smooth_window', 10))

        self.total_env_steps = 0
        self.history = []

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.media_dir, exist_ok=True)

    def save_history(self, save_path):
        '''Save TD3 training history to json.''' 
        save_json(self.history, save_path)

    def save_outputs(self):
        '''Save history json and TD3 learning plots.''' 
        history_path = os.path.join(self.checkpoint_dir, 'td3_training_history.json')
        self.save_history(history_path)

        plot_td3_training_results(
            history=self.history,
            save_dir=self.media_dir,
            episode_smooth_window=self.episode_smooth_window,
            loss_smooth_window=self.loss_smooth_window,
            rate_smooth_window=self.rate_smooth_window,
        )

    def train(self):
        '''Run TD3 training loop.''' 
        reward_term_names = [
            'collision',
            'approach_vehicle',
            'goal_progress',
            'stall',
            'living',
            'lane',
            'goal_reached',
        ]

        for episode_idx in range(1, self.num_episodes + 1):
            obs, reset_info = self.env.reset()
            episode_reward = 0.0
            episode_steps = 0
            terminated = False
            truncated = False
            last_info = reset_info
            last_update_info = None

            collision_flag = False
            drivable_steps = 0
            stall_steps = 0
            final_goal_distance = None
            episode_min_vehicle_distance = float('inf')
            reward_terms_sum = {term_name: 0.0 for term_name in reward_term_names}

            while not (terminated or truncated):
                if self.total_env_steps < self.start_steps:
                    action = self.env.sample_random_action()
                else:
                    action = self.agent.select_action(obs, add_noise=True)

                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = bool(terminated or truncated)

                self.agent.store_transition(obs, action, reward, next_obs, done)

                if len(self.agent.replay_buffer) >= self.batch_size:
                    for _ in range(self.updates_per_step):
                        update_info = self.agent.train_step(batch_size=self.batch_size)
                        if update_info is not None:
                            last_update_info = update_info

                obs = next_obs
                episode_reward += reward
                episode_steps += 1
                self.total_env_steps += 1
                last_info = info

                if bool(info.get('collision', False)):
                    collision_flag = True

                if bool(info.get('on_driving_lane', False)):
                    drivable_steps += 1

                reward_terms = info.get('reward_terms', {})
                for term_name in reward_term_names:
                    reward_terms_sum[term_name] += float(reward_terms.get(term_name, 0.0))

                if float(reward_terms.get('stall', 0.0)) != 0.0:
                    stall_steps += 1

                if info.get('goal_distance', None) is not None:
                    final_goal_distance = float(info['goal_distance'])

                if info.get('min_vehicle_distance', None) is not None:
                    episode_min_vehicle_distance = min(
                        episode_min_vehicle_distance,
                        float(info['min_vehicle_distance'])
                    )

            episode_result = {
                'episode': episode_idx,
                'reward': float(episode_reward),
                'steps': int(episode_steps),
                'termination': last_info.get('term_reason', None),
                'terminated': bool(terminated),
                'truncated': bool(truncated),
                'goal_reached': bool(last_info.get('term_reason', None) == 'goal_reached'),
                'collision': bool(collision_flag),
                'final_goal_distance': final_goal_distance,
                'min_vehicle_distance': None if episode_min_vehicle_distance == float('inf') else float(episode_min_vehicle_distance),
                'drivable_ratio': float(drivable_steps / max(episode_steps, 1)),
                'stall_ratio': float(stall_steps / max(episode_steps, 1)),
                'buffer_size': len(self.agent.replay_buffer),
                'total_env_steps': self.total_env_steps,
                'reward_terms': reward_terms_sum,
            }

            if last_update_info is not None:
                episode_result['critic_loss'] = last_update_info['critic_loss']
                episode_result['actor_loss'] = last_update_info['actor_loss']
                episode_result['total_updates'] = last_update_info['total_updates']

            self.history.append(episode_result)

            print(
                f"[TD3 Train] Episode [{episode_idx}/{self.num_episodes}] "
                f"reward={episode_reward:.4f}, "
                f"steps={episode_steps}, "
                f"reason={last_info.get('term_reason', None)}, "
                f"collision={collision_flag}, "
                f"goal_reached={episode_result['goal_reached']}, "
                f"buffer={len(self.agent.replay_buffer)}"
            )

            if episode_idx % self.save_every == 0:
                checkpoint_path = os.path.join(self.checkpoint_dir, f'td3_episode_{episode_idx:03d}.pt')
                self.agent.save(checkpoint_path)
                print(f"Saved: {checkpoint_path}")
                self.save_outputs()

        final_checkpoint_path = os.path.join(self.checkpoint_dir, 'td3_last.pt')
        self.agent.save(final_checkpoint_path)
        print(f"Saved: {final_checkpoint_path}")

        self.save_outputs()

# --- helper to build td3 agent ---
def build_td3_agent(training_config, max_speed, device='cuda'):
    '''Build TD3 agent from training_config.json.''' 
    cnn_cfg = training_config['cnn']
    td3_params = training_config['td3']['params']

    agent = TD3Agent(
        input_channels=cnn_cfg['input_channels'],
        bev_feature_dim=cnn_cfg['bev_feature_dim'],
        scalar_feature_dim=cnn_cfg['scalar_feature_dim'],
        hidden_dim=cnn_cfg['hidden_dim'],
        action_dim=3,
        max_speed=max_speed,
        actor_learning_rate=td3_params['actor_learning_rate'],
        critic_learning_rate=td3_params['critic_learning_rate'],
        actor_weight_decay=td3_params['actor_weight_decay'],
        critic_weight_decay=td3_params['critic_weight_decay'],
        gamma=td3_params['gamma'],
        tau=td3_params['tau'],
        policy_noise=td3_params['policy_noise'],
        noise_clip=td3_params['noise_clip'],
        policy_delay=td3_params['policy_delay'],
        exploration_speed_noise=td3_params['exploration_speed_noise'],
        exploration_direction_noise=td3_params['exploration_direction_noise'],
        replay_capacity=td3_params['replay_capacity'],
        dropout=td3_params['dropout'],
        device=device,
    )

    return agent


# ----- plotting helpers -----
def save_json(data, save_path):
    '''Save one dictionary or list to json.'''
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as file:
        json.dump(data, file, indent=4)
    print(f"Saved: {save_path}")


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
        xlabel = "Episode"

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

    window = int(max(1, round(window)))
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


def fill_missing_curve_values(values):
    """
    Fill NaN / inf values with the median of valid values.
    Return empty array if there are no valid values.
    """
    values = np.asarray(values, dtype=np.float32)
    valid_mask = np.isfinite(values)

    if not valid_mask.any():
        return np.asarray([], dtype=np.float32)

    fill_value = float(np.nanmedian(values[valid_mask]))
    return np.where(valid_mask, values, fill_value).astype(np.float32)


def plot_single_curve(values,
                      title,
                      ylabel,
                      save_path,
                      smooth_window=1,
                      use_step_axis=False,
                      show_raw=True,
                      ylim=None,
                      label="Curve",
                      color="#1f77b4"):
    """Plot one TD3 curve with BC-style formatting."""
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return

    set_plot_style()
    fig, ax = plt.subplots(figsize=(6.6, 4.1))

    x = np.arange(len(values), dtype=np.float32)
    if use_step_axis:
        x = x / 1000.0

    smooth_values = smooth_curve(values, smooth_window)

    if show_raw and smooth_window > 1 and len(values) > 1:
        ax.plot(x, values, color=color, alpha=0.18, linewidth=1.0)
    ax.plot(x, smooth_values, color=color, label=label)

    ax.set_title(title, pad=6)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="best", frameon=False)

    set_dynamic_x_axis(ax, len(values), use_step_axis=use_step_axis)

    if ylim is not None:
        ax.set_ylim(ylim)

    fig.tight_layout()
    save_figure(fig, save_path)


def plot_multi_curves(curves,
                      title,
                      ylabel,
                      save_path,
                      smooth_window=1,
                      use_step_axis=False,
                      show_raw=False,
                      ylim=None,
                      legend_ncol=1):
    """
    Plot multiple curves in one figure using the same BC-style logic.
    curves: dict[label] = values
    """
    valid_curves = {}
    for label, values in curves.items():
        values = np.asarray(values, dtype=np.float32)
        if len(values) > 0:
            valid_curves[label] = values

    if len(valid_curves) == 0:
        return

    set_plot_style()
    fig, ax = plt.subplots(figsize=(6.6, 4.1))

    colors = [
        "#1f77b4",  # blue
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#ff7f0e",  # orange
        "#8c564b",  # brown
        "#17becf",  # cyan
    ]

    max_points = 0

    for idx, (label, values) in enumerate(valid_curves.items()):
        x = np.arange(len(values), dtype=np.float32)
        if use_step_axis:
            x = x / 1000.0

        curve = smooth_curve(values, smooth_window)
        color = colors[idx % len(colors)]
        max_points = max(max_points, len(values))

        if show_raw and smooth_window > 1 and len(values) > 1:
            ax.plot(x, values, color=color, alpha=0.14, linewidth=1.0)
        ax.plot(x, curve, color=color, label=label)

    ax.set_title(title, pad=6)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="best", frameon=False, ncol=legend_ncol)

    set_dynamic_x_axis(ax, max_points, use_step_axis=use_step_axis)

    if ylim is not None:
        ax.set_ylim(ylim)

    fig.tight_layout()
    save_figure(fig, save_path)


def plot_td3_training_results(history,
                              save_dir,
                              episode_smooth_window=10,
                              loss_smooth_window=10,
                              rate_smooth_window=10):
    """Save all TD3 training plots using BC-style plotting logic."""
    os.makedirs(save_dir, exist_ok=True)

    if len(history) == 0:
        return

    rewards = [episode.get("reward", 0.0) for episode in history]
    steps = [episode.get("steps", 0) for episode in history]

    critic_losses = fill_missing_curve_values(
        [episode.get("critic_loss", np.nan) for episode in history]
    )
    actor_losses = fill_missing_curve_values(
        [episode.get("actor_loss", np.nan) for episode in history]
    )
    final_goal_distance = fill_missing_curve_values(
        [episode.get("final_goal_distance", np.nan) for episode in history]
    )
    min_vehicle_distance = fill_missing_curve_values(
        [episode.get("min_vehicle_distance", np.nan) for episode in history]
    )

    drivable_ratio = np.asarray(
        [episode.get("drivable_ratio", 0.0) for episode in history],
        dtype=np.float32,
    )
    stall_ratio = np.asarray(
        [episode.get("stall_ratio", 0.0) for episode in history],
        dtype=np.float32,
    )
    goal_reached = np.asarray(
        [float(bool(episode.get("goal_reached", False))) for episode in history],
        dtype=np.float32,
    )
    collision = np.asarray(
        [float(bool(episode.get("collision", False))) for episode in history],
        dtype=np.float32,
    )

    plot_single_curve(
        values=rewards,
        title="Episode reward",
        ylabel="Reward",
        save_path=os.path.join(save_dir, "episode_reward.png"),
        smooth_window=episode_smooth_window,
        show_raw=True,
        label="Reward",
        color="#1f77b4",
    )

    plot_single_curve(
        values=steps,
        title="Episode length",
        ylabel="Steps",
        save_path=os.path.join(save_dir, "episode_length.png"),
        smooth_window=episode_smooth_window,
        show_raw=True,
        label="Episode length",
        color="#1f77b4",
    )

    if len(critic_losses) > 0:
        plot_single_curve(
            values=critic_losses,
            title="Critic loss",
            ylabel="Loss",
            save_path=os.path.join(save_dir, "critic_loss.png"),
            smooth_window=loss_smooth_window,
            show_raw=True,
            label="Critic loss",
            color="#1f77b4",
        )

    if len(actor_losses) > 0:
        plot_single_curve(
            values=actor_losses,
            title="Actor loss",
            ylabel="Loss",
            save_path=os.path.join(save_dir, "actor_loss.png"),
            smooth_window=loss_smooth_window,
            show_raw=True,
            label="Actor loss",
            color="#1f77b4",
        )

    if len(final_goal_distance) > 0:
        plot_single_curve(
            values=final_goal_distance,
            title="Final goal distance",
            ylabel="Distance (m)",
            save_path=os.path.join(save_dir, "final_goal_distance.png"),
            smooth_window=episode_smooth_window,
            show_raw=True,
            label="Final goal distance",
            color="#1f77b4",
        )

    if len(min_vehicle_distance) > 0:
        plot_single_curve(
            values=min_vehicle_distance,
            title="Minimum vehicle distance",
            ylabel="Distance (m)",
            save_path=os.path.join(save_dir, "min_vehicle_distance.png"),
            smooth_window=episode_smooth_window,
            show_raw=True,
            label="Minimum vehicle distance",
            color="#1f77b4",
        )

    plot_multi_curves(
        curves={
            "Drivable ratio": drivable_ratio,
            "Stall ratio": stall_ratio,
        },
        title="Episode behavior ratio",
        ylabel="Ratio",
        save_path=os.path.join(save_dir, "behavior_ratio.png"),
        smooth_window=episode_smooth_window,
        show_raw=False,
        ylim=(0.0, 1.0),
    )

    plot_multi_curves(
        curves={
            "Goal reached rate": goal_reached,
            "Collision rate": collision,
        },
        title="Episode outcome rate",
        ylabel="Rate",
        save_path=os.path.join(save_dir, "outcome_rate.png"),
        smooth_window=rate_smooth_window,
        show_raw=False,
        ylim=(0.0, 1.0),
    )

    reward_term_map = {
        "collision": "Collision",
        "approach_vehicle": "Approach vehicle",
        "goal_progress": "Goal progress",
        "stall": "Stall",
        "living": "Living",
        "lane": "Lane",
        "goal_reached": "Goal reached",
    }

    reward_curves = {}
    for key, label in reward_term_map.items():
        values = np.asarray(
            [episode.get("reward_terms", {}).get(key, 0.0) for episode in history],
            dtype=np.float32,
        )
        if np.any(np.abs(values) > 1e-8):
            reward_curves[label] = values

    plot_multi_curves(
        curves=reward_curves,
        title="Episode reward-term breakdown",
        ylabel="Reward contribution",
        save_path=os.path.join(save_dir, "reward_terms.png"),
        smooth_window=episode_smooth_window,
        show_raw=False,
        legend_ncol=2,
    )