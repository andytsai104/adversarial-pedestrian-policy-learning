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
            "ped_id": self.target_ped.id,
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

