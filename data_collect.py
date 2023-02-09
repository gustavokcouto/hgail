import carla
import numpy as np
import pandas as pd
import tqdm

from PIL import Image
from pathlib import Path
import torch as th

from carla_gym.envs import LeaderboardEnv
from carla_gym.core.task_actor.scenario_actor.agents.constant_speed_agent import ConstantSpeedAgent
from carla_gym.utils.expert_noiser import ExpertNoiser

from rl_train import get_obs_configs, get_env_wrapper_configs
from rl_birdview.utils.rl_birdview_wrapper import RlBirdviewWrapper

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
    'weather_group': 'dynamic_1.0',
    'routes_group': 'train'
}


if __name__ == '__main__':
    obs_configs = get_obs_configs(rgb=True)
    env_wrapper_configs = get_env_wrapper_configs()
    env = LeaderboardEnv(obs_configs=obs_configs, reward_configs=reward_configs,
                         terminal_configs=terminal_configs, host="localhost", port=2000,
                         seed=2021, no_rendering=False, **env_configs)
    expert_file_dir = Path('gail_experts')
    expert_file_dir.mkdir(parents=True, exist_ok=True)
    obs_metrics = ['control', 'vel_xy', 'linear_speed', 'vec', 'traj', 'cmd', 'command', 'state']
    for route_id in tqdm.tqdm(range(10)):
        env.set_task_idx(route_id)
        for ep_id in range(1):
            episode_dir = expert_file_dir / ('route_%02d' % route_id) / ('ep_%02d' % ep_id)
            (episode_dir / 'birdview_masks').mkdir(parents=True)
            (episode_dir / 'central_rgb').mkdir(parents=True)
            (episode_dir / 'left_rgb').mkdir(parents=True)
            (episode_dir / 'right_rgb').mkdir(parents=True)

            longitudinal_noiser = ExpertNoiser('Throttle', frequency=15, intensity=10, min_noise_time_amount=2.0)
            lateral_noiser = ExpertNoiser('Spike', frequency=25, intensity=4, min_noise_time_amount=0.5)

            obs = env.reset()
            basic_agent = ConstantSpeedAgent(env._ev_handler.ego_vehicles['hero'], None, 6.0)
            ep_dict = {}
            for state_key in obs_metrics:
                ep_dict[state_key] = []
            ep_dict['done'] = []
            ep_dict['actions'] = []
            actions_ep = []
            i_step = 0
            c_route = False
            while not c_route:
                state_list = []
                obs = obs['hero']
                obs_dict = RlBirdviewWrapper.process_obs(obs, env_wrapper_configs['input_states'])
                for state_key in obs_metrics:
                    ep_dict[state_key].append(obs_dict[state_key])
                ep_dict['done'].append([int(c_route)])
                action = basic_agent.get_action()
                control = carla.VehicleControl(throttle=action[0], steer=action[1])
                # control, _, _ = longitudinal_noiser.compute_noise(control, obs['speed']['forward_speed'][0] * 3.6)
                # control, _, _ = lateral_noiser.compute_noise(control, obs['speed']['forward_speed'][0] * 3.6)
                ep_dict['actions'].append([control.throttle, control.steer])
                birdview = obs['birdview']['masks']
                for i_mask in range(1):
                    birdview_mask = birdview[i_mask * 3: i_mask * 3 + 3]
                    birdview_mask = np.transpose(birdview_mask, [1, 2, 0]).astype(np.uint8)
                    Image.fromarray(birdview_mask).save(episode_dir / 'birdview_masks' / '{:0>4d}_{:0>2d}.png'.format(i_step, i_mask))

                central_rgb = obs_dict['central_rgb']
                central_rgb = np.transpose(central_rgb, [1, 2, 0]).astype(np.uint8)
                Image.fromarray(central_rgb).save(episode_dir / 'central_rgb' / '{:0>4d}.png'.format(i_step))

                left_rgb = obs_dict['left_rgb']
                left_rgb = np.transpose(left_rgb, [1, 2, 0]).astype(np.uint8)
                Image.fromarray(left_rgb).save(episode_dir / 'left_rgb' / '{:0>4d}.png'.format(i_step))

                right_rgb = obs_dict['right_rgb']
                right_rgb = np.transpose(right_rgb, [1, 2, 0]).astype(np.uint8)
                Image.fromarray(right_rgb).save(episode_dir / 'right_rgb' / '{:0>4d}.png'.format(i_step))

                driver_control = {'hero': control}
                obs, reward, done, info = env.step(driver_control)
                c_route = info['hero']['route_completion']['is_route_completed']

                i_step += 1


            ep_df = pd.DataFrame(ep_dict)
            ep_df.to_json(episode_dir / 'episode.json')