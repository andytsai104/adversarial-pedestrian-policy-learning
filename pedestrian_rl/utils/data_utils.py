import carla
import math
import random
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from .sim_utils import CrossroadPedestrians
from ..data_collection.bev.bev_sample import BEVSample, BEVWrapper
from ..data_collection.bev.bev_seg_sample import SemanticBEVSample, SemanticBEVWrapper
from ..data_collection.state_action_pair import PedestrianStateAction



'''
Funtions help converting local-to-world or world-to-local vectors/coordinates
'''
def rotate_world_to_local_2d(vector_xy: np.ndarray, yaw_rad: float) -> np.ndarray:
    '''
    Convert a world-frame 2D vector [x, y] to the pedestrian local frame.
    Output convention: [right, forward].
    '''
    vx = float(vector_xy[0])
    vy = float(vector_xy[1])

    forward_x = math.cos(yaw_rad)
    forward_y = math.sin(yaw_rad)
    right_x = -math.sin(yaw_rad)
    right_y = math.cos(yaw_rad)

    local_forward = vx * forward_x + vy * forward_y
    local_right = vx * right_x + vy * right_y

    return np.array([local_right, local_forward], dtype=np.float32)


def rotate_local_to_world_2d(local_vector_rf: np.ndarray, yaw_rad: float) -> np.ndarray:
    '''
    Convert a pedestrian local-frame vector [right, forward] back to world frame [x, y].
    '''
    local_right = float(local_vector_rf[0])
    local_forward = float(local_vector_rf[1])

    forward_x = math.cos(yaw_rad)
    forward_y = math.sin(yaw_rad)
    right_x = -math.sin(yaw_rad)
    right_y = math.cos(yaw_rad)

    world_x = local_right * right_x + local_forward * forward_x
    world_y = local_right * right_y + local_forward * forward_y

    return np.array([world_x, world_y], dtype=np.float32)


def normalize_direction_2d(direction_xy: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    norm = float(np.linalg.norm(direction_xy))
    if norm < eps:
        return np.zeros(2, dtype=np.float32)
    return (direction_xy / norm).astype(np.float32)



class DataSampler:
    '''
    Class for sampling pedestrian state-action data during one simulation episode.

    This class manages the pedestrian data collection process for dataset generation.
    It keeps a fixed set of target pedestrian IDs for each episode, tracks their previous
    locations and frames, samples their current state and action, and stores the sampled
    results in an episode buffer.

    Attributes:
        world: CARLA world object.
        bev_wrapper: BEV wrapper used to generate bird's-eye-view observations.
        crossroad_pedestrians: CrossroadPedestrians object that stores spawned pedestrians and their goal locations.
        config: Configuration dictionary loaded from the config file.
        fixed_delta_time: Fixed simulation time step.
        sample_ped_num: Number of pedestrians to sample in each episode.
        prev_ped_location: Dictionary storing the previous location of each sampled pedestrian.
        prev_ped_frame: Dictionary storing the previous frame ID of each sampled pedestrian.
        episode_buffer: List storing sampled PedestrianStateAction objects for the current episode.
        target_ped_ids: List of sampled pedestrian IDs that remain fixed during the episode.

    Methods:
        append_sample(ped_info):
            Append one sampled PedestrianStateAction object to the episode buffer.

        get_episode_buffer():
            Return the sampled data buffer for the current episode.

        reset_episode_tracking():
            Reset previous pedestrian tracking info, episode buffer, and target pedestrian IDs
            when a new episode starts.

        select_target_ped_ids():
            Randomly select a fixed set of pedestrian IDs for the current episode.

        get_sample_pedestrians():
            Get the live pedestrian actor objects that correspond to the stored target IDs.

        sample_single_pedestrian(ped, frame_id, timestamp):
            Sample one pedestrian's state and action at the current frame, and return the
            PedestrianStateAction object together with its BEV sample.
    '''

    def __init__(self, world: carla.World, 
                 bev_wrapper, 
                 crossroad_pedestrians: CrossroadPedestrians, 
                 config,
                 bev_sample_class=SemanticBEVSample,
        ):

        self.world = world
        self.bev_wrapper = bev_wrapper
        self.ped_wrappers = {}
        self.crossroad_pedestrians = crossroad_pedestrians
        self.config = config
        self.bev_sample_class = bev_sample_class

        self.fixed_delta_time = config["simulation"]["fixed_delta_seconds"]
        self.sample_ped_num = config["dataset"]["num_ped_per_episode"]

        self.prev_ped_location = {}
        self.prev_ped_frame = {}
        self.episode_buffer = []
        self.target_ped_ids = []

    def append_sample(self, ped_info):
        self.episode_buffer.append(ped_info)

    def get_episode_buffer(self):
        return self.episode_buffer
    
    def reset_episode_tracking(self):
        self.prev_ped_location = {}
        self.prev_ped_frame = {}
        self.episode_buffer = []
        self.target_ped_ids = []

    def select_target_ped_ids(self):
        all_peds = list(self.world.get_actors().filter("walker.*"))
        k = min(self.sample_ped_num, len(all_peds))

        if k == 0:
            self.target_ped_ids = []
        else:
            self.target_ped_ids = [ped.id for ped in random.sample(all_peds, k)]

        return self.target_ped_ids
    
    def get_sample_pedestrians(self):
        all_peds = list(self.world.get_actors().filter("walker.*"))
        ped_dict = {ped.id: ped for ped in all_peds}
        return [ped_dict[pid] for pid in self.target_ped_ids if pid in ped_dict]
    
    def sample_single_pedestrian(self, ped: carla.Walker):
        if ped.id not in self.ped_wrappers:
            raise RuntimeError(f"No semantic wrapper found for ped {ped.id}")

        bev_sample = self.bev_sample_class(
            actor=ped,
            bev_wrapper=self.ped_wrappers[ped.id]
        )
        controller = ped.get_control()
        
        # ----- states -----
        # get_bev() would can tick() once -> sample time relavent information later
        bev_data = bev_sample.get_bev()

        snapshot = self.world.get_snapshot()
        frame_id = snapshot.timestamp.frame
        timestamp = snapshot.timestamp

        ped_info = PedestrianStateAction(
            target_ped=ped,
            frame_id=frame_id,
            timestamp=timestamp
        )

        # current_carla_loc = ped.get_location()
        current_carla_loc = ped.get_transform().location
        current_location = np.array(
            [current_carla_loc.x, current_carla_loc.y, current_carla_loc.z],
            dtype=np.float32
        )

        if ped.id not in self.prev_ped_location:
            velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        else:
            previous_location = self.prev_ped_location[ped.id]
            previous_frame = self.prev_ped_frame[ped.id]
            passed_frame = frame_id - previous_frame
            dt = passed_frame * self.fixed_delta_time

            if dt > 0:
                velocity = (current_location - previous_location) / dt
            else:
                velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        vx = float(velocity[0])
        vy = float(velocity[1])
        speed = float(math.sqrt(vx ** 2 + vy ** 2))

        yaw_heading = math.radians(ped.get_transform().rotation.yaw)
        # if speed >= 0.1:
        #     motion_heading = float(math.atan2(vy, vx))
        # else:
        #     motion_heading = float(yaw_heading)

        self.prev_ped_location[ped.id] = current_location
        self.prev_ped_frame[ped.id] = frame_id

        ped_goal_locations = self.crossroad_pedestrians.ped_goal_loc
        goal_location = ped_goal_locations.get(ped.id)

        ped_info.set_states(
            bev_data=bev_data,
            current_location=current_location,
            velocity=velocity,
            speed=speed,
            # motion_heading=motion_heading,
            yaw_heading=yaw_heading,
            goal_location=goal_location,
        )

        # ----- action -----
        target_speed = float(controller.speed)
        direction = controller.direction
        target_direction = np.array(
            [direction.x, direction.y, direction.z],
            dtype=np.float32
        )

        ped_info.set_actions(
            target_speed=target_speed,
            target_direction=target_direction
        )

        return ped_info, bev_sample
    
    def attach_target_wrappers(self):
        self.close_target_wrappers()

        sample_peds = self.get_sample_pedestrians()

        # Save individual bev_wrapper to each targeted sampling pedestrian
        for ped in sample_peds:
            wrapper = self.bev_wrapper.__class__(cfg=None, world=self.world)
            wrapper.attach_to_actor(ped)
            self.ped_wrappers[ped.id] = wrapper

    def close_target_wrappers(self):
        for wrapper in self.ped_wrappers.values():
            wrapper.close()
        self.ped_wrappers = {}


    

def convert_to_dataset(episode_data: list, output_path, episode_idx=None):
    """
    Save one episode of pedestrian samples to an HDF5 file.

    Args:
        episode_data (list[PedestrianStateAction]):
            List of sampled pedestrian state-action objects from one episode.
        output_path (str):
            Path to output .h5 file.
        episode_idx (int | None):
            Optional episode index. If provided, data will be saved under
            group name f"episode_{episode_idx:05d}".
            If None, data is saved under "episode".

    HDF5 structure:
        /episode_xxxxx/
            /ped_<id>/
                frame_id                    (N,)
                timestamp                   (N,)
                state/bev_data              (N, H, W, C)
                state/current_location      (N, 3)
                state/velocity              (N, 3)
                state/speed                 (N,)
                state/motion_heading        (N,) -> Optional, had removed
                state/yaw_heading           (N,)
                state/goal_location         (N, 3)
                action/target_speed         (N,)
                action/target_direction     (N, 3)
    """
    if len(episode_data) == 0:
        print("[convert_to_dataset] No data to save.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    group_name = f"episode_{episode_idx:03d}" if episode_idx is not None else "episode"

    # Organize the sampled data to {ped_id: [PedestrainStateAction_1, PedestrainStateAction_2, ...]}
    ped_groups = defaultdict(list)
    for sample in episode_data:
        ped_groups[sample.ped_id].append(sample)

    with h5py.File(output_path, "a") as file:
        # overwrite same episode if exists
        if group_name in file:
            del file[group_name]

        episode_grp = file.create_group(group_name)

        for ped_id, samples in ped_groups.items():
            # sort each pedestrian's samples by frame_id
            samples = sorted(samples, key=lambda s: s.state_action_pair["frame_id"])

            # --- Info Buffer Initialization ---
            # Time
            frame_ids = []
            timestamps = []
            
            # States
            bev_data_list = []
            current_locations = []
            velocities = []
            speeds = []
            # motion_headings = []
            yaw_headings = []
            goal_locations = []

            # Actions
            target_speeds = []
            target_directions = []

            # --- Store Data ---
            for sample in samples:
                # Time
                frame_ids.append(sample.state_action_pair["frame_id"])

                ts = sample.state_action_pair["timestamp"]
                if hasattr(ts, "elapsed_seconds"):
                    timestamps.append(float(ts.elapsed_seconds))
                else:
                    timestamps.append(float(ts))

                # States
                bev_data_list.append(np.asarray(sample.state["bev_data"], dtype=np.uint8))
                current_locations.append(np.asarray(sample.state["current_location"], dtype=np.float32))
                velocities.append(np.asarray(sample.state["velocity"], dtype=np.float32))
                speeds.append(np.float32(sample.state["speed"]))
                # motion_headings.append(np.float32(sample.state["motion_heading"]))

                yaw_heading = sample.state.get("yaw_heading")
                yaw_headings.append(np.float32(yaw_heading))

                goal_loc = sample.state["goal_location"]
                if hasattr(goal_loc, "x"):   # carla.Location
                    goal_loc = np.array([goal_loc.x, goal_loc.y, goal_loc.z], dtype=np.float32)
                else:
                    goal_loc = np.asarray(goal_loc, dtype=np.float32)
                goal_locations.append(goal_loc)

                # Actions
                target_speeds.append(np.float32(sample.action["target_speed"]))
                target_directions.append(np.asarray(sample.action["target_direction"], dtype=np.float32))

            # Convert data to numpy array/matrix
            frame_ids = np.asarray(frame_ids, dtype=np.int32)
            timestamps = np.asarray(timestamps, dtype=np.float64)

            bev_data = np.stack(bev_data_list, axis=0)
            current_locations = np.stack(current_locations, axis=0)
            velocities = np.stack(velocities, axis=0)
            speeds = np.asarray(speeds, dtype=np.float32)
            # motion_headings = np.asarray(motion_headings, dtype=np.float32)
            yaw_headings = np.asarray(yaw_headings, dtype=np.float32)
            goal_locations = np.stack(goal_locations, axis=0)

            target_speeds = np.asarray(target_speeds, dtype=np.float32)
            target_directions = np.stack(target_directions, axis=0)

            # --- Create HDF5 dataset ---
            # Create groups
            ped_grp = episode_grp.create_group(f"ped_{ped_id}")
            state_grp = ped_grp.create_group("state")
            action_grp = ped_grp.create_group("action")

            # ped_grp.create_dataset("ped_id", data=np.asarray([ped_id], dtype=np.int32))
            ped_grp.create_dataset("frame_id", data=frame_ids)
            ped_grp.create_dataset("timestamp", data=timestamps)

            state_grp.create_dataset("bev_data", data=bev_data, compression="gzip")
            state_grp.create_dataset("current_location", data=current_locations)
            state_grp.create_dataset("velocity", data=velocities)
            state_grp.create_dataset("speed", data=speeds)
            # state_grp.create_dataset("motion_heading", data=motion_headings)
            state_grp.create_dataset("yaw_heading", data=yaw_headings)
            state_grp.create_dataset("goal_location", data=goal_locations)

            action_grp.create_dataset("target_speed", data=target_speeds)
            action_grp.create_dataset("target_direction", data=target_directions)

    print(f"[convert_to_dataset] Saved {len(episode_data)} samples to {output_path}::{group_name}")


def normalize_direction_2d(direction_xy: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    norm = float(np.linalg.norm(direction_xy))
    if norm < eps:
        return np.zeros(2, dtype=np.float32)
    return (direction_xy / norm).astype(np.float32)


def compute_future_direction_and_speed(
        current_location: np.ndarray,
        future_location: np.ndarray,
        current_yaw_heading: float,
        current_timestamp: float,
        future_timestamp: float,
        eps: float = 1e-6,
    ):
    """
    Compute future realized motion direction from t -> t+k.

    Returns:
        future_direction_local: local-frame unit vector [right, forward]
        future_speed: average realized speed magnitude over the horizon
    """
    future_disp_world = np.asarray(
        future_location[:2] - current_location[:2],
        dtype=np.float32,
    )
    future_dt = float(future_timestamp - current_timestamp)

    if future_dt <= eps:
        return np.zeros(2, dtype=np.float32), 0.0

    future_speed = float(np.linalg.norm(future_disp_world) / future_dt)

    future_direction_local = rotate_world_to_local_2d(
        future_disp_world,
        current_yaw_heading,
    )
    future_direction_local = normalize_direction_2d(future_direction_local, eps=eps)

    return future_direction_local, future_speed

class PedestrianStepDataset(Dataset):
    '''
    One sample = one pedestrian at one timestep.

    HDF5 structure:
        episode_xxx/
            ped_xxx/
                action/
                    target_direction
                    target_speed
                frame_id
                state/
                    bev_data
                    current_location
                    goal_location
                    motion_heading
                    speed
                    velocity
                timestamp
    '''

    def __init__(
        self,
        h5_path,
        use_goal_relative=True,
        goal_scale=16.0,
        clip_bound=3.0,
        speed_eps=0.05,
        future_steps=1,
    ):
        self.h5_path = h5_path
        self.use_goal_relative = use_goal_relative
        self.goal_scale = float(goal_scale)
        self.clip_bound = float(clip_bound)
        self.index = []
        self.speed_eps = float(speed_eps)
        self.future_steps = int(max(1, future_steps))
        self._h5_file = None

        self._build_index()

    def _build_index(self):
        with h5py.File(self.h5_path, "r") as f:
            for episode_name in f.keys():
                episode_group = f[episode_name]
                for ped_name in episode_group.keys():
                    ped_group = episode_group[ped_name]
                    n_steps = ped_group["state"]["bev_data"].shape[0]
                    for t in range(n_steps):
                        self.index.append((episode_name, ped_name, t))

    def _get_h5(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")
        return self._h5_file

    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        f = self._get_h5()
        episode_name, ped_name, t = self.index[idx]
        ped_group = f[episode_name][ped_name]

        # ----- state -----
        bev_data = np.asarray(ped_group["state"]["bev_data"][t], dtype=np.float32)                      # (H, W, C)
        current_location = np.asarray(ped_group["state"]["current_location"][t], dtype=np.float32)      # (2,) or (3,)
        goal_location = np.asarray(ped_group["state"]["goal_location"][t], dtype=np.float32)            # (2,) or (3,)
        velocity = np.asarray(ped_group["state"]["velocity"][t], dtype=np.float32)                      # (2,) or (3,)
        speed = np.float32(ped_group["state"]["speed"][t])                                              # scalar
        # motion_heading = np.float32(ped_group["state"]["motion_heading"][t])                            # scalar

        # if "yaw_heading" in ped_group["state"]:
        yaw_heading = np.float32(ped_group["state"]["yaw_heading"][t])                              # scalar
        # else:
        #     yaw_heading = np.float32(motion_heading)

        # ----- metadata -----
        frame_id = np.int32(ped_group["frame_id"][t])
        timestamp = np.float32(ped_group["timestamp"][t])
        timestep = np.int32(t)

        if self.use_goal_relative:
            goal_world = goal_location - current_location
        else:
            goal_world = goal_location.copy()

        velocity_local = rotate_world_to_local_2d(velocity[:2], yaw_heading)
        goal_rel_local = rotate_world_to_local_2d(goal_world[:2], yaw_heading)
        goal_rel_local = goal_rel_local / max(self.goal_scale, 1e-6)
        goal_rel_local = np.clip(goal_rel_local, -self.clip_bound, self.clip_bound).astype(np.float32)

        # ----- target speed -----
        target_speed = np.float32(ped_group["action"]["target_speed"][t])

        # ----- target direction from future realized motion -----
        n_steps = ped_group["state"]["current_location"].shape[0]
        future_t = min(t + self.future_steps, n_steps - 1)

        future_location = np.asarray(
            ped_group["state"]["current_location"][future_t],
            dtype=np.float32,
        )
        future_timestamp = np.float32(ped_group["timestamp"][future_t])

        target_direction_local, future_motion_speed = compute_future_direction_and_speed(
            current_location=current_location,
            future_location=future_location,
            current_yaw_heading=float(yaw_heading),
            current_timestamp=float(timestamp),
            future_timestamp=float(future_timestamp),
        )

        direction_valid = bool(future_motion_speed > self.speed_eps)
        yaw_sin = np.float32(math.sin(float(yaw_heading)))
        yaw_cos = np.float32(math.cos(float(yaw_heading)))

        sample = {
            # inputs
            "bev_data": torch.from_numpy(bev_data),
            "velocity_local": torch.from_numpy(velocity_local),
            "goal_rel_local": torch.from_numpy(goal_rel_local),
            "yaw_sin": torch.tensor(yaw_sin, dtype=torch.float32),
            "yaw_cos": torch.tensor(yaw_cos, dtype=torch.float32),
            "speed": torch.tensor(speed, dtype=torch.float32),

            # keep raw values for debugging
            "current_location": torch.from_numpy(current_location),
            "goal_location": torch.from_numpy(goal_location),
            "velocity": torch.from_numpy(velocity),
            # "motion_heading": torch.tensor(motion_heading, dtype=torch.float32),
            "yaw_heading": torch.tensor(yaw_heading, dtype=torch.float32),

            # targets
            "target_speed": torch.tensor(target_speed, dtype=torch.float32),
            "target_direction_local": torch.from_numpy(target_direction_local),
            "target_direction_mask": torch.tensor(direction_valid, dtype=torch.bool),

            # debug
            "future_timestep": torch.tensor(future_t),
            "future_motion_speed": torch.tensor(future_motion_speed, dtype=torch.float32),

            # metadata
            "episode": episode_name,
            "ped_id": ped_name,
            "timestep": timestep,
            "frame_id": torch.tensor(frame_id),
            "timestamp": torch.tensor(timestamp),
        }

        return sample

    def close(self):
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None

    def __del__(self):
        self.close()

def test_dataloader():
    dataset_pth = "datasets/pedestrian/pedestrian_dataset.h5"
    dataset = PedestrianStepDataset(
        h5_path=dataset_pth,
        use_goal_relative=True,
        goal_scale=48.0,
        clip_bound=3.0,
        speed_eps=0.05,
    )

    print(f"Total samples: {len(dataset)}")

    sample = dataset[0]
    print("\n--- single sample ---")
    for k, v in sample.items():
        if hasattr(v, "shape"):
            print(f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}")
        else:
            print(f"{k}: {v}")

    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))

    print("\n--- one batch ---")
    for k, v in batch.items():
        if hasattr(v, "shape"):
            print(f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}")
        else:
            print(f"{k}: {v}")

    dataset.close()


def inspect_samples(dataset, indices=(0, 10, 100, 500, 1000)):
    for idx in indices:
        if idx >= len(dataset):
            continue

        s = dataset[idx]
        print(f"\n===== idx={idx} =====")
        print(f"episode               : {s['episode']}")
        print(f"ped_id                : {s['ped_id']}")
        print(f"timestep              : {s['timestep']}")
        print(f"speed                 : {s['speed'].item():.4f}")
        print(f"velocity_local        : {s['velocity_local'].numpy()}")
        print(f"goal_rel_local        : {s['goal_rel_local'].numpy()}")
        print(f"target_speed          : {s['target_speed'].item():.4f}")
        print(f"target_direction_local: {s['target_direction_local'].numpy()}")
        print(f"target_direction_mask : {s['target_direction_mask'].item()}")


if __name__ == "__main__":
    # test_dataloader()

    dataset_pth = "datasets/pedestrian/pedestrian_dataset.h5"
    dataset = PedestrianStepDataset(
        h5_path=dataset_pth,
        use_goal_relative=True,
        goal_scale=48.0,
        clip_bound=3.0,
        speed_eps=0.05,
    )
    inspect_samples(dataset=dataset)