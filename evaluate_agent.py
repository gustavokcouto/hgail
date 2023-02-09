import queue
import carla
import numpy as np
import pandas as pd
import torch as th
from PIL import Image, ImageDraw

import pathlib
import tqdm

from rl_birdview.models.ppo_policy import PpoPolicy
from rl_train import get_obs_configs, get_env_wrapper_configs
from carla_gym.envs import LeaderboardEnv
from rl_birdview.utils.rl_birdview_wrapper import RlBirdviewWrapper
import carla_gym.utils.transforms as trans_utils


reward_configs = {
    'hero': {
        'entry_point': 'reward.valeo_action:ValeoAction',
        'kwargs': {}
    }
}

terminal_configs = {
    'hero': {
        'entry_point': 'terminal.valeo_no_det_px:ValeoNoDetPx',
        'kwargs': {}
    }
}

env_configs = {
    'carla_map': 'Town01',
    'weather_group': 'train',
    'routes_group': 'train'
}

camera_fov = 90.0
camera_z = 14
image_size = 256

fixed_camera_x = 93.53582000732422
fixed_camera_y = 198.314697265625
fixed_traj_radius = 51.98500061035156


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


def plot_route(top_rgb, global_route, route_idx, ev_transform):
    draw = ImageDraw.Draw(top_rgb)
    meters_to_pixel = image_size / (2 * np.tan(camera_fov * np.pi / 180 / 2) * camera_z)
    pre_point_list = []
    post_point_list = []
    for point_idx, route_point in enumerate(global_route):
        point_loc = route_point[0].transform.location
        point_dist = point_loc.distance(ev_transform.location)
        if point_dist > 2.0 and point_dist < 50:
            relative_loc = point_loc - ev_transform.location
            relative_loc = trans_utils.vec_global_to_ref(relative_loc, ev_transform.rotation)
            point_x = meters_to_pixel * relative_loc.y + image_size/2
            point_y = -1 * meters_to_pixel * relative_loc.x + image_size/2
            if point_idx < route_idx:
                pre_point_list.append((point_x, point_y))
            else:
                post_point_list.append((point_x, point_y))

    last_point = None
    rgb = 255
    for point in pre_point_list:
        if last_point is not None:
            draw.line([last_point, point], fill=(255, 255, 0), width=4)
        last_point = point

    last_point = None
    for point in post_point_list:
        if last_point is not None:
            draw.line([last_point, point], fill=(255, 255, 0), width=4)
        last_point = point

    return top_rgb


def plot_traj(episode_dir, x_pos_list, y_pos_list, yaw_list, output_dir):
    for i_step in range(len(x_pos_list)):
        image_path = episode_dir / '{:0>4d}.png'.format(i_step)
        top_rgb = Image.open(image_path)
        draw = ImageDraw.Draw(top_rgb)
        meters_to_pixel = image_size / (2 * np.tan(camera_fov * np.pi / 180 / 2) * camera_z)
        pre_point_list = []
        post_point_list = []
        step_location = carla.Location(x=x_pos_list[i_step], y=y_pos_list[i_step])
        step_rotation = carla.Rotation(yaw=yaw_list[i_step])
        for point_idx, (x_pos, y_pos) in enumerate(zip(x_pos_list, y_pos_list)):
            point_loc = carla.Location(x=x_pos, y=y_pos)
            point_dist = point_loc.distance(step_location)
            if point_dist > 2.0 and point_dist < 50:
                relative_loc = point_loc - step_location
                relative_loc = trans_utils.vec_global_to_ref(relative_loc, step_rotation)
                point_x = meters_to_pixel * relative_loc.y + image_size/2
                point_y = -1 * meters_to_pixel * relative_loc.x + image_size/2
                if point_idx < i_step:
                    pre_point_list.append((point_x, point_y))
                else:
                    post_point_list.append((point_x, point_y))

        last_point = None
        rgb = 255
        # rgb_diff = -255 / (len(pre_point_list) + len(post_point_list))
        for point in pre_point_list:
            if last_point is not None:
                draw.line([last_point, point], fill=(0, 0, 255), width=4)
            # rgb += rgb_diff
            last_point = point

        last_point = None
        for point in post_point_list:
            if last_point is not None:
                draw.line([last_point, point], fill=(0, 0, 255), width=4)
            # rgb += rgb_diff
            last_point = point

        top_rgb.save(output_dir / '{:0>4d}.png'.format(i_step))


def plot_fixed(episode_dir, x_pos_list, y_pos_list, output_dir):
    fixed_location = carla.Location(x=fixed_camera_x, y=fixed_camera_y)
    fixed_rotation = carla.Rotation(yaw=0)
    for i_step in range(len(x_pos_list)):
        image_path = episode_dir / '{:0>4d}.png'.format(i_step)
        top_rgb = Image.open(image_path)
        draw = ImageDraw.Draw(top_rgb)
        meters_to_pixel = image_size / (2 * np.tan(camera_fov * np.pi / 180 / 2) * camera_z)
        pre_point_list = []
        post_point_list = []
        step_location = carla.Location(x=x_pos_list[i_step], y=y_pos_list[i_step])
        for point_idx, (x_pos, y_pos) in enumerate(zip(x_pos_list, y_pos_list)):
            point_loc = carla.Location(x=x_pos, y=y_pos)
            point_dist = point_loc.distance(step_location)
            if point_dist > 2.0 and point_dist < 50:
                relative_loc = point_loc - fixed_location
                relative_loc = trans_utils.vec_global_to_ref(relative_loc, fixed_rotation)
                point_x = meters_to_pixel * relative_loc.y + image_size/2
                point_y = -1 * meters_to_pixel * relative_loc.x + image_size/2
                if point_idx < i_step:
                    pre_point_list.append((point_x, point_y))
                else:
                    post_point_list.append((point_x, point_y))

        last_point = None
        rgb = 255
        # rgb_diff = -255 / (len(pre_point_list) + len(post_point_list))
        for point in pre_point_list:
            if last_point is not None:
                draw.line([last_point, point], fill=(0, 0, 255), width=4)
            # rgb += rgb_diff
            last_point = point

        last_point = None
        for point in post_point_list:
            if last_point is not None:
                draw.line([last_point, point], fill=(0, 0, 255), width=4)
            # rgb += rgb_diff
            last_point = point

        top_rgb.save(output_dir / '{:0>4d}.png'.format(i_step))


def plot_route_fixed(top_rgb, global_route, route_idx, ev_transform):
    draw = ImageDraw.Draw(top_rgb)
    meters_to_pixel = image_size / (2 * np.tan(camera_fov * np.pi / 180 / 2) * camera_z)
    pre_point_list = []
    post_point_list = []
    fixed_location = carla.Location(x=fixed_camera_x, y=fixed_camera_y)
    fixed_rotation = carla.Rotation(yaw=0)
    for point_idx, route_point in enumerate(global_route):
        point_loc = route_point[0].transform.location
        point_dist = point_loc.distance(ev_transform.location)
        if point_dist > 2.0 and point_dist < 50:
            relative_loc = point_loc - fixed_location
            relative_loc = trans_utils.vec_global_to_ref(relative_loc, fixed_rotation)
            point_x = meters_to_pixel * relative_loc.y + image_size/2
            point_y = -1 * meters_to_pixel * relative_loc.x + image_size/2
            if point_idx < route_idx:
                pre_point_list.append((point_x, point_y))
            else:
                post_point_list.append((point_x, point_y))

    last_point = None
    rgb = 255
    if len(pre_point_list) + len(post_point_list) > 0:
        rgb_diff = -255 / (len(pre_point_list) + len(post_point_list))
    else:
        rgb_diff = 0
    for point in pre_point_list:
        if last_point is not None:
            draw.line([last_point, point], fill=(255, int(rgb), 0), width=4)
        rgb += rgb_diff
        last_point = point

    last_point = None
    for point in post_point_list:
        if last_point is not None:
            draw.line([last_point, point], fill=(255, int(rgb), 0), width=4)
        rgb += rgb_diff
        last_point = point

    return top_rgb


def evaluate_agent(ckpt_path):
    obs_configs = get_obs_configs(rgb=True)
    obs_configs['hero'].update({'top_rgb': {
        'module': 'camera.rgb',
        'fov': 90,
        'width': image_size,
        'height': image_size,
        'location': [0.0, 0.0, camera_z],
        'rotation': [0.0, -90.0, 0],
        'debug': True
    }})

    env_wrapper_configs = get_env_wrapper_configs()
    ckpt_key = ckpt_path.stem
    eval_file_dir = pathlib.Path('agent_eval') / ckpt_key
    eval_file_dir.mkdir(parents=True, exist_ok=True)

    env = LeaderboardEnv(obs_configs=obs_configs, reward_configs=reward_configs,
                    terminal_configs=terminal_configs, host='localhost', port=2000,
                    seed=2021, no_rendering=False, **env_configs, routes_file='eval_routes.xml')
    saved_variables = th.load(ckpt_path, map_location='cuda')
    policy_kwargs = saved_variables['policy_init_kwargs']
    policy = PpoPolicy(**policy_kwargs)
    policy.load_state_dict(saved_variables['policy_state_dict'])
    device = 'cuda'
    policy.to(device)

    routes_completed = []
    fixed_camera = Camera(env._world, image_size, image_size, 90, fixed_camera_x, fixed_camera_y, camera_z, -90, 0)
    for route_id in tqdm.tqdm(range(30)):
        episode_dir = eval_file_dir / ('route_%02d' % route_id)
        (episode_dir / 'birdview').mkdir(parents=True)
        (episode_dir / 'fake_birdview').mkdir(parents=True)
        (episode_dir / 'central_rgb').mkdir(parents=True)
        (episode_dir / 'left_rgb').mkdir(parents=True)
        (episode_dir / 'right_rgb').mkdir(parents=True)
        (episode_dir / 'top_rgb').mkdir(parents=True)
        (episode_dir / 'route_plot').mkdir(parents=True)
        (episode_dir / 'traj_plot').mkdir(parents=True)
        (episode_dir / 'traj_route_plot').mkdir(parents=True)
        (episode_dir / 'fixed_camera').mkdir(parents=True)
        (episode_dir / 'fixed_traj_camera').mkdir(parents=True)
        (episode_dir / 'fixed_route_plot').mkdir(parents=True)
        (episode_dir / 'fixed_traj_route_plot').mkdir(parents=True)

        env.set_task_idx(route_id)
        i_step = 0
        x_pos_list = []
        y_pos_list = []
        z_pos_list = []
        yaw_list = []
        c_route_list = []
        done_list = []
        yaw_list = []
        route_completed_list = []
        raw_obs = env.reset()
        ego_vehicle = env._ev_handler.ego_vehicles['hero']
        global_route = ego_vehicle._global_route
        c_route = False
        done_ep = False
        while not c_route and not done_ep:
            obs = raw_obs['hero']
            obs = RlBirdviewWrapper.process_obs(obs, env_wrapper_configs['input_states'])
            obs_dict = dict([(k, np.stack([obs[k]])) for k in obs])

            actions, _, _, _, _, _, fake_birdview = policy.forward(obs_dict, deterministic=True)
            acc, steer = actions[0].astype(np.float64)
            control = carla.VehicleControl(throttle=acc, steer=steer)
            driver_control = {'hero': control}
            birdview = obs['birdview']
            for i_mask in range(1):
                birdview_mask = birdview[i_mask * 3: i_mask * 3 + 3]
                birdview_mask = np.transpose(birdview_mask, [1, 2, 0]).astype(np.uint8)
                Image.fromarray(birdview_mask).save(episode_dir / 'birdview' / '{:0>4d}_{:0>2d}.png'.format(i_step, i_mask))

            for i_mask in range(1):
                birdview_mask = fake_birdview[0][i_mask * 3: i_mask * 3 + 3]
                birdview_mask = np.transpose(birdview_mask, [1, 2, 0]).astype(np.uint8)
                Image.fromarray(birdview_mask).save(episode_dir / 'fake_birdview' / '{:0>4d}_{:0>2d}.png'.format(i_step, i_mask))

            central_rgb = obs['central_rgb']
            central_rgb = np.transpose(central_rgb, [1, 2, 0]).astype(np.uint8)
            Image.fromarray(central_rgb).save(episode_dir / 'central_rgb' / '{:0>4d}.png'.format(i_step))
            left_rgb = obs['left_rgb']
            left_rgb = np.transpose(left_rgb, [1, 2, 0]).astype(np.uint8)
            Image.fromarray(left_rgb).save(episode_dir / 'left_rgb' / '{:0>4d}.png'.format(i_step))
            right_rgb = obs['right_rgb']
            right_rgb = np.transpose(right_rgb, [1, 2, 0]).astype(np.uint8)
            Image.fromarray(right_rgb).save(episode_dir / 'right_rgb' / '{:0>4d}.png'.format(i_step))
            top_rgb = raw_obs['hero']['top_rgb']['data']
            top_rgb = top_rgb.astype(np.uint8)
            Image.fromarray(top_rgb).save(episode_dir / 'top_rgb' / '{:0>4d}.png'.format(i_step))

            fixed_img = fixed_camera.get()
            Image.fromarray(fixed_img).save(episode_dir / 'fixed_camera' / '{:0>4d}.png'.format(i_step))

            ev_transform = ego_vehicle.vehicle.get_transform()
            ev_loc = ev_transform.location
            ev_rot = ev_transform.rotation

            route_plot = Image.fromarray(top_rgb)
            route_idx = len(global_route) - len(ego_vehicle._global_route)
            route_plot = plot_route(route_plot, global_route, route_idx, ev_transform)
            route_plot.save(episode_dir / 'route_plot' / '{:0>4d}.png'.format(i_step))

            fixed_route_plot = Image.fromarray(fixed_img)
            route_idx = len(global_route) - len(ego_vehicle._global_route)
            fixed_route_plot = plot_route_fixed(fixed_route_plot, global_route, route_idx, ev_transform)
            fixed_route_plot.save(episode_dir / 'fixed_route_plot' / '{:0>4d}.png'.format(i_step))

            x_pos_list.append(ev_loc.x)
            y_pos_list.append(ev_loc.y)
            z_pos_list.append(ev_loc.z)
            yaw_list.append(ev_rot.yaw)
            raw_obs, _, done, info = env.step(driver_control)

            c_route = info['hero']['route_completion']['is_route_completed']
            route_completed_in_m = info['hero']['route_completion']['route_completed_in_m']
            route_completed_list.append(route_completed_in_m)

            i_step += 1
            done_ep = done['hero']
            c_route_list.append(c_route)
            done_list.append(done_ep)

        ep_df = pd.DataFrame({
                'x': x_pos_list,
                'y': y_pos_list,
                'z': z_pos_list,
                'yaw': yaw_list,
                'c_route': c_route_list,
                'done': done_list,
                'route_completed_in_m': route_completed_list
            })
        
        plot_traj(episode_dir / 'top_rgb', x_pos_list, y_pos_list, yaw_list, episode_dir / 'traj_plot')

        plot_traj(episode_dir / 'route_plot', x_pos_list, y_pos_list, yaw_list, episode_dir / 'traj_route_plot')

        plot_fixed(episode_dir / 'fixed_camera', x_pos_list, y_pos_list, episode_dir / 'fixed_traj_camera')

        plot_fixed(episode_dir / 'fixed_route_plot', x_pos_list, y_pos_list, episode_dir / 'fixed_traj_route_plot')

        ep_df.to_json(episode_dir / 'episode.json')

        routes_completed.append(c_route)

    ep_df = pd.DataFrame({
        'routes_completed': routes_completed
    })

    ep_df.to_json(eval_file_dir / 'eval.json')


if __name__ == '__main__':
    # ckpt_dir = pathlib.Path('ckpt')
    # ckpt_list = [ckpt_file for ckpt_file in ckpt_dir.iterdir() if ckpt_file.stem!='ckpt_latest']
    # ckpt_list.sort(key=lambda ckpt_file: int(ckpt_file.stem.split('_')[1]))
    # ckpt_idx = 0
    # for _ in range(12):
    #     evaluate_agent(ckpt_list[ckpt_idx])
    #     ckpt_idx += 4
    ckpt_file = pathlib.Path('/home/casa/projects/paper_results/hgail/ckpt/ckpt_1351680.pth')
    evaluate_agent(ckpt_file)
