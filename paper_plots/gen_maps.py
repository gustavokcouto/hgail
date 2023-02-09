import queue
from pathlib import Path

from PIL import Image, ImageDraw, ImageColor
import numpy as np

import carla
from carla import ColorConverter as cc

from carla_gym.utils import config_utils
from carla_gym.core.task_actor.common.navigation.global_route_planner import GlobalRoutePlanner
from carla_gym.core.task_actor.common.navigation.route_manipulation import downsample_route


class Camera(object):
    def __init__(self, world, w, h, fov, x, y, z, pitch, yaw):
        bp_library = world.get_blueprint_library()
        camera_bp = bp_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(w))
        camera_bp.set_attribute('image_size_y', str(h))
        camera_bp.set_attribute('fov', str(fov))

        loc = carla.Location(x=x, y=y, z=z)
        rot = carla.Rotation(pitch=pitch, yaw=yaw)
        transform = carla.Transform(loc, rot)

        self.queue = queue.Queue()

        self.camera = world.spawn_actor(camera_bp, transform)
        self.camera.listen(self.queue.put)

    def get(self):
        image = None

        while image is None or self.queue.qsize() > 0:
            image = self.queue.get()

        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        return array

    def __del__(self):
        pass
        # self.camera.destroy()

        # with self.queue.mutex:
        #     self.queue.queue.clear()


def process_img(image):
    image.convert(cc.Raw)
    image.save_to_disk('_out/%08d' % image.frame_number)


def set_sync_mode(client, sync):
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = sync
    settings.fixed_delta_seconds = 1.0 / 10.0

    world.apply_settings(settings)


host = 'localhost'
port = 2000
client = carla.Client(host, port)
client.set_timeout(30.0)
world = client.load_world('Town01')

set_sync_mode(client, True)

map = world.get_map()
planner = GlobalRoutePlanner(map, resolution=1.0)

traj_output_dir = Path('paper_plots/traj_plot')
traj_output_dir.mkdir(parents=True, exist_ok=True)

routes_file = Path('carla_gym/envs/scenario_descriptions/LeaderBoard/Town01/routes.xml')
route_descriptions_dict = config_utils.parse_routes_file(routes_file)
camera_x = 197.14505004882812
camera_y = 164.2880915403366
traj_radius = 199.2049560546875
for route_id, route_description in route_descriptions_dict.items():
    route_config = route_description['ego_vehicles']
    spawn_transform = route_config['hero'][0]
    target_transforms = route_config['hero'][1:]
    current_location = spawn_transform.location
    global_plan_world_coord = []
    for tt in target_transforms:
        next_target_location = tt.location
        route_trace = planner.trace_route(current_location, next_target_location)
        global_plan_world_coord += route_trace
        current_location = next_target_location
    ds_ids = downsample_route(global_plan_world_coord, 50)

    image_width = 512
    image_height = 512
    camera_fov = 90
    camera_z = 1.2 * np.tan((90 - camera_fov / 2) * np.pi / 180) * traj_radius
    camera = Camera(world, image_width, image_height, camera_fov, camera_x, camera_y, camera_z, -90, 0)
    world.tick()
    result = camera.get()
    image = Image.fromarray(result)
    draw = ImageDraw.Draw(image)
    last_point = global_plan_world_coord[0][0].transform.location
    meters_to_pixel_x = image_width / (2 * np.tan(camera_fov * np.pi / 180 / 2) * camera_z)
    meters_to_pixel_y = -image_height / (2 * np.tan(camera_fov * np.pi / 180 / 2) * camera_z)
    last_point_x = meters_to_pixel_x * (last_point.y - camera_y) + image_width / 2
    last_point_y = meters_to_pixel_y * (last_point.x - camera_x) + image_height / 2
    radius = 2
    n_points = len(global_plan_world_coord)
    rgb_diff = -255 / n_points
    rgb = 255
    draw.ellipse([last_point_x - radius, last_point_y - radius, last_point_x + radius, last_point_y + radius], fill=(255, int(rgb), 0))
    rgb += rgb_diff
    for point_idx in range(1, len(global_plan_world_coord)):
        point_loc = global_plan_world_coord[point_idx][0].transform.location
        point_x = meters_to_pixel_x * (point_loc.y - camera_y) + image_width / 2
        point_y = meters_to_pixel_y * (point_loc.x - camera_x) + image_height / 2
        draw.line((last_point_x, last_point_y, point_x, point_y), width=2, fill=(255, int(rgb), 0))
        rgb += rgb_diff
        last_point_x = point_x
        last_point_y = point_y
        if point_idx in ds_ids:
            draw.ellipse([point_x - radius, point_y - radius, point_x + radius, point_y + radius], fill=(255, 255, 0))

    draw.ellipse([last_point_x - radius, last_point_y - radius, last_point_x + radius, last_point_y + radius], fill=(255, int(rgb), 0))
    image_path = traj_output_dir / 'route_{:0>2d}.png'.format(route_id)
    image.save(image_path.as_posix())