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
import gym


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
    'carla_map': 'Town02',
    'weather_group': 'simple',
    'routes_group': 'train'
}


def evaluate_agent(ckpt_path):
    obs_configs = get_obs_configs(rgb=True)

    env_wrapper_configs = get_env_wrapper_configs(rgb=True)
    ckpt_key = ckpt_path.stem
    eval_file_dir = pathlib.Path('agent_eval') / ckpt_key
    eval_file_dir.mkdir(parents=True, exist_ok=True)

    env = LeaderboardEnv(obs_configs=obs_configs, reward_configs=reward_configs,
                    terminal_configs=terminal_configs, host='localhost', port=2000,
                    seed=2021, no_rendering=False, **env_configs)
    saved_variables = th.load(ckpt_path, map_location='cuda')
    observation_space = {}
    observation_space['birdview'] = gym.spaces.Box(low=0, high=255, shape=(3, 192, 192), dtype=np.uint8)
    rgb_channels = 3
    rgb_space = gym.spaces.Box(low=0, high=255, shape=(rgb_channels, 144, 256), dtype=np.uint8)
    observation_space['central_rgb'] = rgb_space
    observation_space['left_rgb'] = rgb_space
    observation_space['right_rgb'] = rgb_space
    observation_space['state'] = gym.spaces.Box(low=-10.0, high=30.0, shape=(6,), dtype=np.float32)
    observation_space['traj'] = gym.spaces.Box(low=-10.0, high=30.0, shape=(10,), dtype=np.float32)

    observation_space = gym.spaces.Dict(**observation_space)
    policy_kwargs = {
        'observation_space': observation_space,
        'action_space': gym.spaces.Box(low=np.array([0, -1]), high=np.array([1, 1]), dtype=np.float32),
        'policy_head_arch': [256, 256],
        'value_head_arch': [256, 256],
        'features_extractor_entry_point': 'rl_birdview.models.torch_layers:XtMaCNN',
        'features_extractor_kwargs': {'states_neurons': [256,256]},
        'distribution_entry_point': 'rl_birdview.models.distributions:BetaDistribution',
        'fake_birdview': True
    }
    policy = PpoPolicy(**policy_kwargs)
    policy.load_state_dict(saved_variables['policy_state_dict'])
    device = 'cuda'
    policy.to(device)
    policy = policy.eval()

    routes_completed = []
    for route_id in tqdm.tqdm(range(6)):
        episode_dir = eval_file_dir / ('route_%02d' % route_id)
        (episode_dir / 'birdview').mkdir(parents=True)
        (episode_dir / 'fake_birdview').mkdir(parents=True)
        (episode_dir / 'central_rgb').mkdir(parents=True)
        (episode_dir / 'left_rgb').mkdir(parents=True)
        (episode_dir / 'right_rgb').mkdir(parents=True)
        (episode_dir / 'traj_plot').mkdir(parents=True)

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

            traj_plot = obs['traj_plot'].numpy()
            traj_plot = np.transpose(traj_plot, [1, 2, 0]).astype(np.uint8)
            zero_img = np.zeros([traj_plot.shape[0], traj_plot.shape[0], 3], dtype=np.uint8)
            zero_img[:,:,0:1] = traj_plot
            Image.fromarray(zero_img).save(episode_dir / 'traj_plot' / '{:0>4d}.png'.format(i_step))

            ev_transform = ego_vehicle.vehicle.get_transform()
            ev_loc = ev_transform.location
            ev_rot = ev_transform.rotation

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
        
        ep_df.to_json(episode_dir / 'episode.json')

        routes_completed.append(c_route)

    ep_df = pd.DataFrame({
        'routes_completed': routes_completed
    })

    ep_df.to_json(eval_file_dir / 'eval.json')


if __name__ == '__main__':
    ckpt_file = pathlib.Path('ckpt/ckpt_811008.pth')
    evaluate_agent(ckpt_file)
