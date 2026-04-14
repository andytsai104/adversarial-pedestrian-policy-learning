import weakref
import math
import random
import carla
import cv2
import numpy as np
import torch
from ..utils.config_loader import load_config
from ..utils.sim_utils import (
    Spector,
    CrossroadPedestrians,
    AggressiveVehicles,
    refresh_sim,
    cleanup_simulation,
)
from ..data_collection.bev.bev_sample import BEVWrapper, BEVSample
from ..data_collection.bev.bev_seg_sample import SemanticBEVWrapper, SemanticBEVSample
from .data_utils import rotate_world_to_local_2d, rotate_local_to_world_2d
from ..utils.td3_utils import PedestrianRLEnv, build_td3_agent
from ..models.cnn_encoder import CNNEncoder
from collections import defaultdict



class PolicyRunner:
    """
    Run a trained Behavior Cloning pedestrian policy in CARLA for multiple pedestrians.

    Pipeline per tick:
        CARLA world -> build observation -> model -> WalkerControl -> apply_control
    """

    def __init__(
        self,
        model_class,
        model_name: str,
        checkpoint_path,
        training_config_name="training_config.json",
        sim_config_name="sim_config.json",
        no_rendering_mode=False,
        device="cuda",
        num_model_peds=5,
        bev_wrapper=SemanticBEVWrapper,
        bev_sampler=SemanticBEVSample
    ):
        # --- load config ---
        self.sim_config = load_config(sim_config_name)
        self.training_config = load_config(training_config_name)

        sim_cfg = self.sim_config["simulation"]
        self.fixed_delta_seconds = sim_cfg["fixed_delta_seconds"]
        self.max_episode_steps = sim_cfg["max_episode_steps"]
        self.warmup_ticks = sim_cfg["warmup_ticks"]
        self.max_ped_speed = sim_cfg["pedestrian"]["speed_range"][1]
        self.goal_scale = float(self.training_config["bc"]["params"]["goal_scale"])
        self.clip_bound = float(self.training_config["bc"]["params"]["clip_bound"])
        self.num_model_peds = int(num_model_peds)

        if device is None or (device == "cuda" and not torch.cuda.is_available()):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # --- connecting CARLA ---
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.fixed_delta_seconds
        settings.no_rendering_mode = no_rendering_mode
        self.world.apply_settings(settings)

        self.intersection_position = carla.Location(
            x=sim_cfg["intersection"]["x"],
            y=sim_cfg["intersection"]["y"],
            z=sim_cfg["intersection"]["z"],
        )
        self.distance = sim_cfg["intersection"]["dist"]

        # --- intersection sim setup ---
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
        self.bev_wrapper = bev_wrapper(cfg=None, world=self.world)
        self.bev_sampler = bev_sampler

        # --- define prediction model ---
        self.model_class = model_class
        self.model = None
        self.model_name = model_name
        self._build_model(checkpoint_path=checkpoint_path)

        # --- initialize buffers ---
        self.target_peds = {}
        self.target_goals = {}
        self.prev_locations = {}
        self.prev_frames = {}
        self.peds_step = {}
        self.episode_step = 0

        # --- get stuck conditions from config ---
        stuck_detection_config = sim_cfg["stuck_detection"]
        self.refresh_conditions = {
            "time_out": stuck_detection_config["time_out"],
            "start time": self.world.get_snapshot().timestamp.elapsed_seconds,
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

    def _build_model(self, checkpoint_path: str):
        """Build up prediction model from optimal checkpoint."""
        bev_feature_dim = self.training_config["cnn"]["bev_feature_dim"]
        hidden_dim = self.training_config["cnn"]["hidden_dim"]
        direction_dim = self.training_config["cnn"]["direction_dim"]
        dropout = 0.0

        if self.training_config is not None:
            input_channels = self.training_config["cnn"]["input_channels"]
            bev_feature_dim = self.training_config["cnn"]["bev_feature_dim"]
            hidden_dim = self.training_config["cnn"]["hidden_dim"]
            direction_dim = self.training_config["cnn"]["direction_dim"]
            dropout = self.training_config["bc"]["params"]["dropout"]

        cnn_encoder = CNNEncoder(input_channels=input_channels, feature_dim=bev_feature_dim)
        self.model = self.model_class(
            cnn_encoder=cnn_encoder,
            bev_feature_dim=bev_feature_dim,
            hidden_dim=hidden_dim,
            direction_dim=direction_dim,
            dropout=dropout,
        ).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        print(f"[{self.model_name} PolicyRunner] Loaded checkpoint: {checkpoint_path}")

    def _reset_target_tracking(self):
        self.prev_locations = {}
        self.prev_frames = {}
        self.peds_step = {}
        self.episode_step = 0

    def reset_episode(self):
        cleanup_simulation(self.world)
        self.crossroad_pedestrians.reset_pedestrians()

        self.spector.set_spector()
        self.aggressive_vehicles.aggressive_vehicles_spawn()

        self.target_peds = {}
        self.target_goals = {}
        self._reset_target_tracking()

        spawned = 0
        model_spawned = 0
        spawn_points = self.crossroad_pedestrians.get_ped_spawn_points(
            self.crossroad_pedestrians.ped_num,
            self.crossroad_pedestrians.in_intersection,
        )
        random.shuffle(spawn_points)

        while spawned < self.crossroad_pedestrians.ped_num and len(spawn_points) > 0:
            spawn_location = spawn_points.pop()
            destination = self.world.get_random_location_from_navigation()

            if destination is None:
                continue
            destination.z += 1.0

            controller = self if model_spawned < self.num_model_peds else "ai"

            ped = self.crossroad_pedestrians.spawn_single_walker(
                spawn_location=spawn_location,
                destination=destination,
                controller=controller,
            )

            if ped is None:
                continue

            spawned += 1

            if controller is self:
                goal = self.crossroad_pedestrians.ped_goal_loc.get(ped.id, None)
                if goal is None:
                    loc = ped.get_location()
                    goal = np.array([loc.x, loc.y, loc.z], dtype=np.float32)
                else:
                    goal = np.asarray(goal, dtype=np.float32)

                self.target_peds[ped.id] = ped
                self.target_goals[ped.id] = goal
                self.peds_step[ped.id] = 0
                model_spawned += 1

        if len(self.target_peds) == 0:
            raise RuntimeError("Failed to spawn any model-controlled pedestrians.")

        if self.warmup_ticks > 0:
            for _ in range(self.warmup_ticks):
                self.world.tick()

        self.refresh_conditions["start time"] = self.world.get_snapshot().timestamp.elapsed_seconds
        self.refresh_conditions["vehicle"]["stuck_tracker"] = {}

        print(
            f"[{self.model_name} PolicyRunner] New episode with "
            f"{len(self.target_peds)} model-controlled pedestrians: {list(self.target_peds.keys())}"
        )

    def _compute_velocity_speed(self, ped: carla.Actor, frame_id: int):
        current_loc = ped.get_location()
        current_location = np.array([current_loc.x, current_loc.y, current_loc.z], dtype=np.float32)

        prev_location = self.prev_locations.get(ped.id, None)
        prev_frame = self.prev_frames.get(ped.id, None)

        if prev_location is None or prev_frame is None:
            velocity = np.zeros(3, dtype=np.float32)
        else:
            passed_frames = frame_id - prev_frame
            dt = passed_frames * self.fixed_delta_seconds
            if dt > 0:
                velocity = (current_location - prev_location) / dt
            else:
                velocity = np.zeros(3, dtype=np.float32)

        speed = float(np.linalg.norm(velocity[:2]))
        self.prev_locations[ped.id] = current_location.copy()
        self.prev_frames[ped.id] = frame_id
        return current_location, velocity, speed

    def _build_observation(self, ped: carla.Actor, frame_id: int):
        """Build up observation for one target pedestrian."""
        bev_sample = self.bev_sampler(actor=ped, bev_wrapper=self.bev_wrapper)
        bev_data = bev_sample.get_bev()

        current_location, velocity, speed = self._compute_velocity_speed(ped, frame_id)
        yaw_heading = math.radians(ped.get_transform().rotation.yaw)

        goal = self.target_goals.get(ped.id, current_location.copy())
        goal_rel_world = goal - current_location
        velocity_local = rotate_world_to_local_2d(velocity[:2], yaw_heading)
        goal_rel_local = rotate_world_to_local_2d(goal_rel_world[:2], yaw_heading)
        goal_rel_local = goal_rel_local / max(self.goal_scale, 1e-6)
        goal_rel_local = np.clip(goal_rel_local, -self.clip_bound, self.clip_bound).astype(np.float32)

        batch = {
            "bev_data": torch.from_numpy(np.asarray(bev_data, dtype=np.float32)).unsqueeze(0).to(self.device),
            "velocity_local": torch.from_numpy(velocity_local).unsqueeze(0).to(self.device),
            "goal_rel_local": torch.from_numpy(goal_rel_local).unsqueeze(0).to(self.device),
            "yaw_sin": torch.tensor([math.sin(yaw_heading)], dtype=torch.float32, device=self.device),
            "yaw_cos": torch.tensor([math.cos(yaw_heading)], dtype=torch.float32, device=self.device),
            "speed": torch.tensor([speed], dtype=torch.float32, device=self.device),
        }

        debug_state = {
            "bev_sample": bev_sample,
            "current_location": current_location,
            "velocity": velocity,
            "velocity_local": velocity_local,
            "speed": speed,
            "yaw_heading": yaw_heading,
            "goal_rel_local": goal_rel_local,
        }
        return batch, debug_state

    def _postprocess_action(self, outputs, ped: carla.Actor):
        """Process predicted actions to executable controller actions."""
        pred_speed = float(outputs["pred_speed"][0, 0].item())
        pred_direction_local = outputs["pred_direction"][0].detach().cpu().numpy().astype(np.float32)

        direction_norm = float(np.linalg.norm(pred_direction_local))
        if direction_norm < 1e-6:
            pred_direction_local = np.array([0.0, 1.0], dtype=np.float32)
        else:
            pred_direction_local = pred_direction_local / direction_norm

        yaw_heading = math.radians(ped.get_transform().rotation.yaw)
        pred_direction_world_xy = rotate_local_to_world_2d(pred_direction_local, yaw_heading)

        direction_world = np.array(
            [pred_direction_world_xy[0], pred_direction_world_xy[1], 0.0],
            dtype=np.float32,
        )

        pred_speed = max(0.0, min(pred_speed, float(self.max_ped_speed)))
        return pred_speed, pred_direction_local, direction_world

    def _apply_control(self, ped: carla.Actor, speed: float, direction: np.ndarray, jump: bool = False):
        control = carla.WalkerControl()
        control.speed = float(speed)
        control.jump = bool(jump)
        control.direction = carla.Vector3D(
            x=float(direction[0]),
            y=float(direction[1]),
            z=float(direction[2]),
        )
        ped.apply_control(control)

    def _target_reached(self, ped: carla.Actor, threshold: float = 1.5):
        goal = self.target_goals.get(ped.id, None)
        if goal is None:
            return False

        current_loc = ped.get_location()
        goal_loc = carla.Location(
            x=float(goal[0]),
            y=float(goal[1]),
            z=float(goal[2]),
        )
        return current_loc.distance(goal_loc) < threshold

    def step_once(self, render_bev=True):
        self.world.tick()
        self.episode_step += 1

        # Refresh condition
        sim_state, should_refresh = refresh_sim(
            world=self.world,
            refresh_conditions=self.refresh_conditions,
            intersection_position=self.intersection_position,
        )

        if should_refresh:
            print(f"[{self.model_name} PolicyRunner] Refresh triggered: {sim_state}")
            self.reset_episode()
            return

        if len(self.target_peds) == 0:
            print(f"[{self.model_name} PolicyRunner] No target pedestrians found. Resetting episode.")
            self.reset_episode()
            return

        dead_peds = [ped_id for ped_id, ped in self.target_peds.items() if ped is None or not ped.is_alive]
        if len(dead_peds) > 0:
            print(
                f"[{self.model_name} PolicyRunner] Lost model-controlled pedestrians: "
                f"{dead_peds}. Resetting episode."
            )
            self.reset_episode()
            return

        snapshot = self.world.get_snapshot()
        frame_id = snapshot.timestamp.frame

        peds_debug = defaultdict(dict)
        reached_any_goal = False

        # Spawn model-controlled pedestrians
        for ped_id, ped in self.target_peds.items():
            batch, debug_state = self._build_observation(ped, frame_id)

            with torch.no_grad():
                outputs = self.model(batch)

            pred_speed, pred_direction_local, pred_direction_world = self._postprocess_action(outputs, ped)
            self._apply_control(ped, pred_speed, pred_direction_world, jump=False)
            self.peds_step[ped_id] = self.peds_step.get(ped_id, 0) + 1

            
            peds_debug[ped_id] = {
                    "debug_state": debug_state,
                    "pred_speed": pred_speed,
                    "pred_direction_local": pred_direction_local,
                }

            if self._target_reached(ped):
                reached_any_goal = True

        if peds_debug[ped_id] is not None:
            # debug_state = peds_debug[ped_id]["debug_state"]
            # pred_speed = peds_debug[ped_id]["pred_speed"]
            # pred_direction_local = peds_debug[ped_id]["pred_direction_local"]

            print(
                f"[{self.model_name}] ped_id={ped_id} | step={self.peds_step[ped_id]} "
                f"| episode_step={self.episode_step} | frame={frame_id}\n"
                f"  speed(state)         : {debug_state['speed']:.3f}\n"
                f"  velocity_local       : {np.round(debug_state['velocity_local'], 3)}\n"
                f"  pred_speed           : {pred_speed:.3f}\n"
                f"  pred_dir_local       : {np.round(pred_direction_local, 3)}\n"
                f"  pred_dir_world       : {np.round(pred_direction_world, 3)}\n"
                f"  goal_rel_local       : {np.round(debug_state['goal_rel_local'], 3)}\n"
            )

            if render_bev:
                image = debug_state["bev_sample"].visualize_bev()
                cv2.imshow(f"{self.model_name} Policy Runner BEV", image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    raise KeyboardInterrupt

        if reached_any_goal:
            print(f"[{self.model_name} PolicyRunner] At least one target reached its goal. Resetting episode.")
            self.reset_episode()
            return

        if self.episode_step >= self.max_episode_steps:
            print(f"[{self.model_name} PolicyRunner] Max episode steps reached. Resetting episode.")
            self.reset_episode()
            return

    def run(self, render_bev=True):
        self.reset_episode()
        while True:
            for _ in range(2):
                self.world.tick()
            self.step_once(render_bev=render_bev)



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

    def get_metrics(self, controller_name: str, seed: int, episode_id: int):
        avg_speed = self.speed_sum / self.episode_steps if self.episode_steps > 0 else 0.0

        return {
            "controller": controller_name,
            "seed": seed,
            "episode_id": episode_id,
            "collision": self.collision,
            "steps_to_collision": self.steps_to_collision,
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
        if self.collision_sensor is not None and self.collision_sensor.is_alive:
            self.collision_sensor.destroy()
            self.collision_sensor = None


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