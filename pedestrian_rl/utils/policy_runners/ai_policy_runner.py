import random

import carla

from ...utils.config_loader import load_config
from ...utils.eval_utils import EpisodeEvaluator
from ...utils.sim_utils import (
    AggressiveVehicles,
    CrossroadPedestrians,
    Spector,
    cleanup_simulation,
    refresh_sim,
)


class AIControllerRunner:
    '''Evaluate CARLA built-in AI walker controller under the same scenario.''' 

    def __init__(
        self,
        sim_config_name="sim_config.json",
        no_rendering_mode=False,
        num_model_peds=5,
        evaluation_seed=0,
    ):
        self.sim_config = load_config(sim_config_name)
        sim_cfg = self.sim_config["simulation"]

        self.fixed_delta_seconds = sim_cfg["fixed_delta_seconds"]
        self.max_episode_steps = sim_cfg["max_episode_steps"]
        self.warmup_ticks = sim_cfg["warmup_ticks"]
        self.num_model_peds = int(num_model_peds)
        self.evaluation_seed = int(evaluation_seed)
        self.model_name = "carla.ai.controller"

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

        self.target_peds = {}
        self.evaluators = {}
        self.episode_step = 0
        self.current_episode_id = 0
        self.completed_episode_rows = []

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

    def _finalize_episode_metrics(self, reason: str):
        rows = []
        for ped_id, evaluator in self.evaluators.items():
            row = evaluator.get_metrics(
                controller_name=self.model_name,
                seed=self.evaluation_seed,
                episode_id=self.current_episode_id,
                ped_id=ped_id,
            )
            row["term_reason"] = reason
            rows.append(row)
        self.completed_episode_rows.extend(rows)
        print(f"[{self.model_name}] Finalized episode {self.current_episode_id} with {len(rows)} rows. reason={reason}")
        self.current_episode_id += 1
        return rows

    def get_completed_rows(self):
        return list(self.completed_episode_rows)

    def close_evaluators(self):
        for evaluator in self.evaluators.values():
            evaluator.destroy()
        self.evaluators = {}

    def reset_episode(self, finalize_reason=None):
        if finalize_reason is not None:
            self._finalize_episode_metrics(finalize_reason)

        self.close_evaluators()
        cleanup_simulation(self.world)
        self.crossroad_pedestrians.reset_pedestrians()
        self.spector.set_spector()
        self.aggressive_vehicles.aggressive_vehicles_spawn()

        self.target_peds = {}
        self.episode_step = 0

        spawned = 0
        tracked_spawned = 0
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

            ped = self.crossroad_pedestrians.spawn_single_walker(
                spawn_location=spawn_location,
                destination=destination,
                controller="ai",
            )
            if ped is None:
                continue

            spawned += 1
            if tracked_spawned < self.num_model_peds:
                self.target_peds[ped.id] = ped
                tracked_spawned += 1

        if len(self.target_peds) == 0:
            raise RuntimeError("Failed to spawn any AI-controlled target pedestrians.")

        for ped_id, ped in self.target_peds.items():
            self.evaluators[ped_id] = EpisodeEvaluator(
                world=self.world,
                target_ped=ped,
                dt=self.fixed_delta_seconds,
            )

        for _ in range(self.warmup_ticks):
            self.world.tick()

        self.refresh_conditions["start time"] = self.world.get_snapshot().timestamp.elapsed_seconds
        self.refresh_conditions["vehicle"]["stuck_tracker"] = {}

        print(f"[{self.model_name}] New episode with target pedestrians: {list(self.target_peds.keys())}")

    def step_once(self):
        self.world.tick()
        self.episode_step += 1

        sim_state, should_refresh = refresh_sim(
            world=self.world,
            refresh_conditions=self.refresh_conditions,
            intersection_position=self.intersection_position,
        )

        if should_refresh:
            self.reset_episode(finalize_reason=sim_state)
            return

        if len(self.target_peds) == 0:
            self.reset_episode(finalize_reason="no_target_pedestrians")
            return

        dead_peds = [ped_id for ped_id, ped in self.target_peds.items() if ped is None or not ped.is_alive]
        if len(dead_peds) > 0:
            self.reset_episode(finalize_reason="pedestrian_missing")
            return

        for evaluator in self.evaluators.values():
            evaluator.update()

        if self.episode_step >= self.max_episode_steps:
            self.reset_episode(finalize_reason="max_episode_steps")
            return

    def run(self):
        self.reset_episode()
        while True:
            self.step_once()
