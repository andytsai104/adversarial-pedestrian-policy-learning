import math
import queue
import time
from typing import Dict, Optional

import carla
import cv2
import numpy as np

from ...utils.config_loader import load_config


class SemanticBEVWrapper:
    '''
    Semantic top-down BEV wrapper using CARLA semantic segmentation camera.

    This wrapper attaches one semantic segmentation camera above the target pedestrian,
    points it downward, reads the raw semantic image, and converts the CARLA semantic
    tags into a compact multi-channel BEV tensor.

    Notes:
        - This is a perspective top-down semantic view, not a true orthographic map BEV.
        - We use the raw semantic tag IDs, not the CityScapes palette image.
        - CARLA documents state that the semantic tag is encoded in the red channel of
          the raw image. Since CARLA image raw_data is BGRA, the red channel is index 2
          after reshaping to (H, W, 4).
    '''

    config = load_config("sim_config.json")["bev"]

    # CARLA 0.9.16 semantic tag IDs from the official sensor docs.
    TAGS = {
        "unlabeled": 0,
        "road": 1,
        "sidewalk": 2,
        "building": 3,
        "wall": 4,
        "fence": 5,
        "pole": 6,
        "traffic_light": 7,
        "traffic_sign": 8,
        "vegetation": 9,
        "terrain": 10,
        "sky": 11,
        "pedestrian": 12,
        "rider": 13,
        "car": 14,
        "truck": 15,
        "bus": 16,
        "train": 17,
        "motorcycle": 18,
        "bicycle": 19,
        "static": 20,
        "dynamic": 21,
        "other": 22,
        "water": 23,
        "road_line": 24,
        "ground": 25,
        "bridge": 26,
        "rail_track": 27,
        "guard_rail": 28,
    }

    def __init__(
        self,
        cfg,
        world: carla.World,
        camera_height: float = config["camera_height"],
        image_size_x: Optional[int] = config["size"][0],
        image_size_y: Optional[int] = config["size"][1],
        fov: float = config["fov"],
        sensor_tick: float = 0.0,
        hybrid: bool = True
    ):
        self.world = world
        self.hero_actor = None
        self.actor_list = None
        self.sensor = None

        if cfg is not None:
            self.config = cfg
        else:
            self.config = SemanticBEVWrapper.config

        self.hero_ped_size = self.config["hero_ped_size"]
        self.other_ped_size = self.config["other_ped_size"]

        self.width = int(image_size_x)
        self.height = int(image_size_y)
        self.bev_range = float(self.config["range"])
        self.hybrid = hybrid

        self.camera_height = float(camera_height)
        self.fov = float(fov)
        self.sensor_tick = float(sensor_tick)

        self.last_used_image = None
        self.last_tag_map = None
        self.latest_raw_image = None
        self.latest_frame = -1

    def _build_sensor_bp(self):
        blueprint_library = self.world.get_blueprint_library()
        sensor_bp = blueprint_library.find("sensor.camera.semantic_segmentation")
        sensor_bp.set_attribute("image_size_x", str(self.width))
        sensor_bp.set_attribute("image_size_y", str(self.height))
        sensor_bp.set_attribute("fov", str(self.fov))
        sensor_bp.set_attribute("sensor_tick", str(self.sensor_tick))
        return sensor_bp

    def _camera_transform(self) -> carla.Transform:
        '''
        Camera above pedestrian, looking straight downward.

        Since the sensor is attached to the pedestrian actor, this transform is
        relative to the pedestrian, not a fixed world location.
        '''
        return carla.Transform(
            carla.Location(x=0.0, y=0.0, z=self.camera_height),
            carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
        )

    def attach_to_actor(self, actor: carla.Actor):
        if actor is None:
            raise ValueError("actor cannot be None")

        self.destroy_sensor()
        self.hero_actor = actor

        sensor_bp = self._build_sensor_bp()
        self.sensor = self.world.spawn_actor(
            sensor_bp,
            self._camera_transform(),
            attach_to=actor,
        )

        def _on_image(image: carla.Image):
            self.latest_raw_image = image
            self.latest_frame = image.frame

        self.sensor.listen(_on_image)

    def destroy_sensor(self):
        if self.sensor is not None and self.sensor.is_alive:
            self.sensor.stop()
            self.sensor.destroy()

        self.sensor = None
        self.latest_raw_image = None
        self.last_used_image = None
        self.latest_frame = -1
        self.last_tag_map = None

    def set_hero_actor(self, actor: carla.Actor):
        '''
        Set the current hero actor and reattach the camera when needed.
        '''
        if actor is None:
            raise ValueError("actor cannot be None")

        need_reattach = False

        if self.hero_actor is None:
            need_reattach = True
        elif not self.hero_actor.is_alive:
            need_reattach = True
        elif self.hero_actor.id != actor.id:
            need_reattach = True
        elif self.sensor is None:
            need_reattach = True
        elif not self.sensor.is_alive:
            need_reattach = True

        if need_reattach:
            self.attach_to_actor(actor)
        else:
            self.hero_actor = actor

    def _ensure_sensor(self):
        if self.hero_actor is None:
            raise RuntimeError("hero_actor is not set")

        if not self.hero_actor.is_alive:
            raise RuntimeError("hero_actor is no longer alive")

        if self.sensor is None or (not self.sensor.is_alive):
            self.attach_to_actor(self.hero_actor)

    @staticmethod
    def image_to_tag_map(image: carla.Image):
        '''
        Convert CARLA semantic camera raw image to tag map.
        '''
        bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        tag_map = bgra[:, :, 2].copy()
        return tag_map

    def _mask_from_ids(self, tag_map: np.ndarray, tag_ids):
        if isinstance(tag_ids, int):
            mask = (tag_map == tag_ids)
        else:
            mask = np.isin(tag_map, np.asarray(list(tag_ids), dtype=np.uint8))
        return (mask.astype(np.uint8) * 255)

    def tag_map_to_layers(self, tag_map: np.ndarray):
        tags = self.TAGS
        
        sidewalk_ids = {tags["sidewalk"]}
        road_line_ids = {tags["road_line"]}

        road_ids = {
            tags["road"],
            tags["ground"],
            tags["bridge"],
        }

        vehicle_ids = {
            tags["car"],
            tags["truck"],
            tags["bus"],
            tags["train"],
            tags["motorcycle"],
            tags["bicycle"],
            tags["rider"],
            tags["dynamic"],
        }

        blocked_ids = {
            tags["building"],
            tags["wall"],
            tags["fence"],
            tags["pole"],
            tags["traffic_light"],
            tags["traffic_sign"],
            tags["guard_rail"],
            tags["rail_track"],
            tags["static"],
            tags["water"],
            tags["vegetation"],
            tags["terrain"],
            tags["other"],
            tags["unlabeled"]
        }

        # soft_obstacle_ids = {
        #     tags["vegetation"],
        #     tags["terrain"],
        #     tags["other"],
        # }

        layers = {
            "road": self._mask_from_ids(tag_map, road_ids),
            "sidewalk": self._mask_from_ids(tag_map, sidewalk_ids),
            "road_line": self._mask_from_ids(tag_map, road_line_ids),
            "vehicle": self._mask_from_ids(tag_map, vehicle_ids),
            "pedestrian": self._mask_from_ids(tag_map, tags["pedestrian"]),
            "obstacles": self._mask_from_ids(tag_map, blocked_ids),
            # "soft_obstacle": self._mask_from_ids(tag_map, soft_obstacle_ids),
            # "unknown": self._mask_from_ids(tag_map, tags["unlabeled"]),
        }
        return layers


    def get_bev_data(self):
        self._ensure_sensor()

        if self.latest_raw_image is None:
            raise RuntimeError("No semantic image received yet for this pedestrian")

        self.actor_list = self.world.get_actors()
        image = self.latest_raw_image
        tag_map = self.image_to_tag_map(image)
        self.last_used_image = image
        self.last_tag_map = tag_map

        layers = self.tag_map_to_layers(tag_map)
        if self.hybrid:
            layers["pedestrian"] = self.draw_actor_layers("walker")

        return layers

    def get_raw_tag_map(self):
        # self.world.tick()
        if self.last_used_image is None:
            return None
        image = self.last_used_image
        tag_map = self.image_to_tag_map(image)
        self.last_tag_map = tag_map
        return tag_map

    def get_cityscapes_visualization(self):
        '''
        Convert the last semantic image to CARLA CityScapes palette for visualization only.

        IMPORTANT:
            - carla.Image cannot be instantiated from Python.
            - We therefore convert self.last_used_image in place.
            - This is only for display. The next sensor callback will provide a fresh raw image.
        '''
        if self.last_used_image is None:
            return None

        # Some CARLA versions use cityScapesPalette, some use CityScapesPalette.
        if hasattr(carla.ColorConverter, "cityScapesPalette"):
            self.last_used_image.convert(carla.ColorConverter.cityScapesPalette)
        else:
            self.last_used_image.convert(carla.ColorConverter.CityScapesPalette)

        bgra = np.frombuffer(
            self.last_used_image.raw_data,
            dtype=np.uint8,
        ).reshape((self.last_used_image.height, self.last_used_image.width, 4))

        return bgra[:, :, :3][:, :, ::-1].copy()
    
    def draw_actor_layers(self, actor_type:str = "walker"):
        '''
        Draw layers for actors (walkers and vehicles)
        '''
        canvas = np.zeros((self.height, self.width), dtype=np.uint8)
        actors = self.actor_list.filter(actor_type + ".*")
        
        for actor in actors:            
            # Pedestrians (Walkers) as Circles
            if actor_type == "walker":
                # hero pedestrin (larger mark and lighter feature)
                if actor.id == self.hero_actor.id:
                    color = 100
                    radius = self.hero_ped_size
                else:    
                    color = 255
                    radius = self.other_ped_size   
                px, py = self.world_to_pixel(actor.get_location())
                if 0 <= px < self.width and 0 <= py < self.height:
                    cv2.circle(canvas, (px, py), radius=radius, color=color, thickness=-1)

        return canvas
    
    def world_to_pixel(self, target_location: carla.Location):
        '''
        Project world location to semantic camera image pixels
        using top-down pinhole geometry.
        '''

        hero_transform = self.hero_actor.get_transform()
        hero_location = hero_transform.location
        yaw_rad = np.radians(hero_transform.rotation.yaw)

        dx = target_location.x - hero_location.x
        dy = target_location.y - hero_location.y

        # hero local frame
        forward_x = np.cos(yaw_rad)
        forward_y = np.sin(yaw_rad)
        right_x = -np.sin(yaw_rad)
        right_y = np.cos(yaw_rad)

        local_forward = dx * forward_x + dy * forward_y
        local_right = dx * right_x + dy * right_y

        # camera footprint on ground from pinhole geometry
        half_fov_x = np.radians(self.fov * 0.5)
        half_width_m = self.camera_height * np.tan(half_fov_x)

        aspect_ratio = float(self.height) / float(self.width)
        half_height_m = half_width_m * aspect_ratio

        # normalize to [-1, 1]
        u = local_right / half_width_m
        v = local_forward / half_height_m

        # image coords
        px = int(round((u + 1.0) * 0.5 * (self.width - 1)))
        py = int(round((1.0 - (v + 1.0) * 0.5) * (self.height - 1)))


        return px, py
    
    def show_target_pedestrian(self):
        '''
        Visualize target pedestrian in CARLA
        '''
        if self.hero_actor:
            location = self.hero_actor.get_location()
            location.z = 1
            self.world.debug.draw_point(location, size=0.1, color=carla.Color(0, 255, 0), life_time=2.0)

    def close(self):
        self.destroy_sensor()


class SemanticBEVSample:
    '''
    Helper class to fetch semantic top-down BEV for one actor.
    '''

    def __init__(self, actor: carla.Walker, bev_wrapper: SemanticBEVWrapper):
        self.wrapper = bev_wrapper
        self.actor = actor
        self.feature_tensor = None

    def get_bev(self):
        if self.wrapper.hero_actor is None:
            self.wrapper.attach_to_actor(self.actor)
        elif self.wrapper.hero_actor.id != self.actor.id:
            raise RuntimeError(
                f"Wrapper attached to ped {self.wrapper.hero_actor.id}, "
                f"but asked to sample ped {self.actor.id}")
        
        layers = self.wrapper.get_bev_data()

        self.feature_tensor = np.stack([
            layers["road"],
            layers["sidewalk"],
            layers["road_line"],
            layers["vehicle"],
            layers["pedestrian"],
            layers["obstacles"],
        ], axis=-1)
        return self.feature_tensor

    def visualize_bev(self):
        self.wrapper.set_hero_actor(self.actor)
        self.wrapper.show_target_pedestrian()
        layers = self.wrapper.get_bev_data()

        image = np.zeros((self.wrapper.height, self.wrapper.width, 3), dtype=np.uint8)
        # pedestrian_circle = self.wrapper.pedestrian_mask_to_circles(layers["pedestrian"])
        # Draw colors
        # image[layers["road"] > 0] = (128, 64, 128)
        # image[layers["sidewalk"] > 0] = (232, 35, 244)
        # image[layers["obstacles"] > 0] = (156, 102, 102)
        # image[layers["vehicle"] > 0] = (142, 0, 0)
        # image[layers["pedestrian"] > 0] = (60, 20, 220)
        # image[layers["soft_obstacle"] > 0] = (35, 142, 107)
        # image[layers["unknown"] > 0] = (0, 0, 0)
        image[layers["road"] > 0] = (70, 70, 70)
        image[layers["sidewalk"] > 0] = (200, 200, 200)
        image[layers["road_line"] > 0] = (176, 211, 242)
        image[layers["obstacles"] > 0] = (0, 0, 0)
        image[layers["vehicle"] > 0] = (142, 0, 0)
        image[layers["pedestrian"] == 255] = (255, 150, 100)
        image[layers["pedestrian"] == 100] = (150, 255, 100)

        return image

    def visualize_cityscapes(self):
        rgb = self.wrapper.get_cityscapes_visualization()
        if rgb is None:
            return None
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def find_pedestrian(world, test_ped):
    '''
    Find one live pedestrian without advancing the world.
    '''
    if test_ped is not None and test_ped.is_alive:
        return test_ped

    ped_list = world.get_actors().filter("walker.*")

    if len(ped_list) > 0:
        test_ped = ped_list[0]
        print(f"Find target pedestrian!! ID: {test_ped.id}")
        return test_ped

    time.sleep(0.1)
    return None


def semantic_bev_test(world):
    seg_wrapper = SemanticBEVWrapper(cfg=None, world=world)
    test_ped = None
    last_ped_id = None
    view_every_n_step = 5
    step = 0
    try:
        while True:
            world.tick()
            step+=1
            test_ped = find_pedestrian(world, test_ped)

            if (step % view_every_n_step) != 0:
                continue
            if test_ped is None:
                continue

            if not test_ped.is_alive:
                test_ped = None
                last_ped_id = None
                continue

            # Attach only when needed
            if last_ped_id != test_ped.id:
                seg_wrapper.attach_to_actor(test_ped)
                last_ped_id = test_ped.id

                # warm up the newly attached sensor
                # for _ in range(2):
                #     world.tick()

            if seg_wrapper.latest_raw_image is None:
                continue

            seg_sample = SemanticBEVSample(actor=test_ped, bev_wrapper=seg_wrapper)

            bev_image = seg_sample.visualize_bev()
            cv2.imshow("Semantic BEV Debug Tool", bev_image)

            cityscapes_image = seg_sample.visualize_cityscapes()
            if cityscapes_image is not None:
                cv2.imshow("Semantic Camera CityScapesPalette", cityscapes_image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        seg_wrapper.close()
        world.tick()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    client = carla.Client("localhost", 2000)
    world = client.get_world()

    try:
        semantic_bev_test(world)
    except KeyboardInterrupt:
        print("Semantic BEV test stopped by user.")
