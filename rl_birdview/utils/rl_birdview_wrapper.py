# modified from https://github.com/zhejz/carla-roach/blob/main/agents/rl_birdview/utils/rl_birdview_wrapper.py

import gym
import numpy as np
import cv2
import carla
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw

import carla_gym.utils.transforms as trans_utils
import carla_gym.core.task_actor.common.navigation.route_manipulation as gps_util
from rl_birdview.models.discriminator import traj_plotter, traj_plotter_rgb


eval_num_zombie_vehicles = {
    'Town01': 120,
    'Town02': 70,
    'Town03': 70,
    'Town04': 150,
    'Town05': 120,
    'Town06': 120
}
eval_num_zombie_walkers = {
    'Town01': 120,
    'Town02': 70,
    'Town03': 70,
    'Town04': 80,
    'Town05': 120,
    'Town06': 80
}


class RlBirdviewWrapper(gym.Wrapper):
    def __init__(self, env, input_states=[], acc_as_action=False, env_id=0, resume=False):
        assert len(env._obs_configs) == 1
        self._ev_id = list(env._obs_configs.keys())[0]
        self._input_states = input_states
        self._acc_as_action = acc_as_action
        self._render_dict = {}
        self._env_id = env_id

        observation_space = {}
        observation_space['birdview'] = env.observation_space[self._ev_id]['birdview']['masks']
        if 'rgb' in input_states:
            rgb_height, rgb_width, _  = env.observation_space[self._ev_id]['central_rgb']['data'].shape
            rgb_channels = 3
            rgb_space = gym.spaces.Box(low=0, high=255, shape=(rgb_channels, rgb_height, rgb_width), dtype=np.uint8)
            observation_space['central_rgb'] = rgb_space
            observation_space['left_rgb'] = rgb_space
            observation_space['right_rgb'] = rgb_space

        if 'speed' in self._input_states:
            observation_space['speed'] = env.observation_space[self._ev_id]['speed']['speed_xy']
        if 'speed_limit' in self._input_states:
            observation_space['speed_limit'] = env.observation_space[self._ev_id]['control']['speed_limit']
        if 'control' in self._input_states:
            observation_space['control'] = gym.spaces.Box(low=-10.0, high=30.0, shape=(4,), dtype=np.float32)
        if 'acc_xy' in self._input_states:
            observation_space['acc_xy'] = env.observation_space[self._ev_id]['velocity']['acc_xy']
        if 'vel_xy' in self._input_states:
            observation_space['vel_xy'] = env.observation_space[self._ev_id]['velocity']['vel_xy']
        if 'vel_ang_z' in self._input_states:
            observation_space['vel_ang_z'] = env.observation_space[self._ev_id]['velocity']['vel_ang_z']
        if 'linear_speed' in self._input_states:
            observation_space['linear_speed'] = gym.spaces.Box(low=-10.0, high=30.0, shape=(1,), dtype=np.float32)
        if 'vec' in self._input_states:
            observation_space['vec'] = gym.spaces.Box(low=-10.0, high=30.0, shape=(2,), dtype=np.float32)
        if 'traj' in self._input_states:
            observation_space['traj'] = gym.spaces.Box(low=-10.0, high=30.0, shape=(10,), dtype=np.float32)
            observation_space['traj_plot'] = gym.spaces.Box(low=0, high=255, shape=(1, 192, 192), dtype=np.uint8)
            observation_space['traj_plot_rgb'] = gym.spaces.Box(low=0, high=255, shape=(1, 144, 256), dtype=np.uint8)
        if 'cmd' in self._input_states:
            observation_space['cmd'] = gym.spaces.Box(low=-10.0, high=30.0, shape=(6,), dtype=np.float32)
        if 'command' in self._input_states:
            observation_space['command'] = gym.spaces.Box(low=-10.0, high=30.0, shape=(1,), dtype=np.float32)
        if 'state' in self._input_states:
            observation_space['state'] = gym.spaces.Box(low=-10.0, high=30.0, shape=(6,), dtype=np.float32)

        env.observation_space = gym.spaces.Dict(**observation_space)

        if self._acc_as_action:
            # act: acc(throttle/brake), steer
            env.action_space = gym.spaces.Box(low=np.array([0, -1]), high=np.array([1, 1]), dtype=np.float32)
        else:
            # act: throttle, steer, brake
            env.action_space = gym.spaces.Box(low=np.array([0, -1, 0]), high=np.array([1, 1, 1]), dtype=np.float32)

        super(RlBirdviewWrapper, self).__init__(env)

        self.output_path = Path('runs/env_info/{}'.format(self._env_id))
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.eval_mode = False
        self.ep_info = []
        if resume:
            ep_df = pd.read_csv(self.output_path / '{}.csv'.format(self._env_id), on_bad_lines='skip')
            self.ep_count = ep_df.iloc[-1]['ep_count']
        else:
            self.ep_count = 0
        self.eval_mode = False

    def reset_monitor(self):
        ep_df = pd.DataFrame(self.ep_info)
        if self.ep_count < 2:
            ep_df.to_csv(
                self.output_path / '{}.csv'.format(self._env_id),
                index=False
            )
        else:
            ep_df.to_csv(
                self.output_path / '{}.csv'.format(self._env_id),
                index=False,
                mode='a',
                header=False
            )

        self.ep_count += 1
        self.ep_info = []

    def reset(self):
        self.reset_monitor()
        if self.eval_mode:
            self.env.eval_mode = True
            self.env._task['num_zombie_vehicles'] = eval_num_zombie_vehicles[self.env._carla_map]
            self.env._task['num_zombie_walkers'] = eval_num_zombie_walkers[self.env._carla_map]
            for ev_id in self.env._ev_handler._terminal_configs:
                self.env._ev_handler._terminal_configs[ev_id]['kwargs']['eval_mode'] = True
        else:
            self.env.eval_mode = False
            self.env.set_task_idx(np.random.choice(self.env.num_tasks))
            for ev_id in self.env._ev_handler._terminal_configs:
                self.env._ev_handler._terminal_configs[ev_id]['kwargs']['eval_mode'] = False

        obs_ma = self.env.reset()
        action_ma = {self._ev_id: carla.VehicleControl(manual_gear_shift=True, gear=1)}
        obs_ma, _, _, _ = self.env.step(action_ma)
        action_ma = {self._ev_id: carla.VehicleControl(manual_gear_shift=False)}
        obs_ma, _, _, _ = self.env.step(action_ma)

        snap_shot = self.env._world.get_snapshot()
        self.env._timestamp = {
            'step': 0,
            'frame': 0,
            'relative_wall_time': 0.0,
            'wall_time': snap_shot.timestamp.platform_timestamp,
            'relative_simulation_time': 0.0,
            'simulation_time': snap_shot.timestamp.elapsed_seconds,
            'start_frame': snap_shot.timestamp.frame,
            'start_wall_time': snap_shot.timestamp.platform_timestamp,
            'start_simulation_time': snap_shot.timestamp.elapsed_seconds
        }

        obs = self.process_obs(obs_ma[self._ev_id], self._input_states)

        self._render_dict['prev_obs'] = obs
        self._render_dict['prev_im_render'] = obs_ma[self._ev_id]['birdview']['rendered']
        return obs

    def step(self, action):
        action_ma = {self._ev_id: self.process_act(action, self._acc_as_action)}

        obs_ma, reward_ma, done_ma, info_ma = self.env.step(action_ma)

        self.step_monitor(done_ma, obs_ma)

        obs = self.process_obs(obs_ma[self._ev_id], self._input_states)
        reward = reward_ma[self._ev_id]
        done = done_ma[self._ev_id]
        info = info_ma[self._ev_id]

        self._render_dict = {
            'timestamp': self.env.timestamp,
            'obs': self._render_dict['prev_obs'],
            'prev_obs': obs,
            'im_render': self._render_dict['prev_im_render'],
            'prev_im_render': obs_ma[self._ev_id]['birdview']['rendered'],
            'action': action,
            'reward_debug': info['reward_debug'],
            'terminal_debug': info['terminal_debug']
        }
        return obs, reward, done, info

    def step_monitor(self, done_dict, obs_dict):
        command = obs_dict['hero']['route_plan']['command'][0]
        done = done_dict['hero']
        ego_vehicle = self.env._ev_handler.ego_vehicles['hero']
        ev_transform = ego_vehicle.vehicle.get_transform()
        ev_velocity = ego_vehicle.vehicle.get_velocity()
        step_info = {
            'command': command,
            'x': ev_transform.location.x,
            'y': ev_transform.location.y,
            'yaw': ev_transform.rotation.yaw,
            'velocity_x': ev_velocity.x,
            'velocity_y': ev_velocity.y,
            'velocity_z': ev_velocity.z,
            'n_epoch': self.n_epoch,
            'num_timesteps': self.num_timesteps,
            'eval_mode': self.eval_mode,
            'done': done
        }
        step_info['ep_count'] = self.ep_count
        self.ep_info.append(step_info)

    def render(self, mode='human'):
        '''
        train render: used in train_rl.py
        '''
        self._render_dict['action_value'] = self.action_value
        self._render_dict['action_log_probs'] = self.action_log_probs
        self._render_dict['action_mu'] = self.action_mu
        self._render_dict['action_sigma'] = self.action_sigma
        return self.im_render(self._render_dict)

    @staticmethod
    def im_render(render_dict):
        im_birdview = render_dict['im_render']
        h, w, c = im_birdview.shape
        im = np.zeros([h, w*2, c], dtype=np.uint8)
        im[:h, :w] = im_birdview

        action_str = np.array2string(render_dict['action'], precision=2, separator=',', suppress_small=True)
        mu_str = np.array2string(render_dict['action_mu'], precision=2, separator=',', suppress_small=True)
        sigma_str = np.array2string(render_dict['action_sigma'], precision=2, separator=',', suppress_small=True)
        state_str = np.array2string(render_dict['obs']['state'], precision=2, separator=',', suppress_small=True)

        txt_t = f'step:{render_dict["timestamp"]["step"]:5}, frame:{render_dict["timestamp"]["frame"]:5}'
        im = cv2.putText(im, txt_t, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        txt_1 = f'a{action_str} v:{render_dict["action_value"]:5.2f} p:{render_dict["action_log_probs"]:5.2f}'
        im = cv2.putText(im, txt_1, (3, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        txt_2 = f's{state_str}'
        im = cv2.putText(im, txt_2, (3, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        txt_3 = f'a{mu_str} b{sigma_str}'
        im = cv2.putText(im, txt_3, (w, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        for i, txt in enumerate(render_dict['reward_debug']['debug_texts'] +
                                render_dict['terminal_debug']['debug_texts']):
            im = cv2.putText(im, txt, (w, (i+2)*12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        return im

    @staticmethod
    def process_obs(obs, input_states, train=True):
        if 'cmd' in input_states or 'command' in input_states:
            # VOID = -1
            # LEFT = 1
            # RIGHT = 2
            # STRAIGHT = 3
            # LANEFOLLOW = 4
            # CHANGELANELEFT = 5
            # CHANGELANERIGHT = 6
            command = obs['gnss']['command'][0]
            if command < 0:
                command = 4
            command -= 1

        if 'vec' in input_states or 'traj' in input_states:
            ev_gps = obs['gnss']['gnss']
            # imu nan bug
            compass = 0.0 if np.isnan(obs['gnss']['imu'][-1]) else obs['gnss']['imu'][-1]
            ref_rot_in_global = carla.Rotation(yaw=np.rad2deg(compass)-90.0)
            ev_loc = gps_util.gps_to_location(ev_gps)

            traj_points = obs['gnss']['traj_points']
            traj_locs = []
            point_idx = 0
            while (point_idx + 1) * 3 <= traj_points.shape[0]:
                gps_point = traj_points[point_idx * 3:(point_idx + 1) * 3]
                target_vec_in_global = gps_util.gps_to_location(gps_point) - ev_loc
                loc_in_ev = trans_utils.vec_global_to_ref(target_vec_in_global, ref_rot_in_global)
                traj_locs.append(loc_in_ev)
                point_idx += 1

            gps_point = obs['gnss']['target_gps']
            target_vec_in_global = gps_util.gps_to_location(gps_point) - ev_loc
            loc_in_ev = trans_utils.vec_global_to_ref(target_vec_in_global, ref_rot_in_global)

        obs_dict = {}
        if 'speed' in input_states:
            obs_dict['speed'] = obs['speed']['speed_xy']
        if 'speed_limit' in input_states:
            obs_dict['speed_limit'] = obs['control']['speed_limit']
        if 'control' in input_states:
            obs_dict['control'] = np.concatenate([
                obs['control']['throttle'],
                obs['control']['steer'],
                obs['control']['brake'],
                obs['control']['gear'] / 5.0
            ])

        if 'acc_xy' in input_states:
            obs_dict['acc_xy'] = obs['velocity']['acc_xy']
        if 'vel_xy' in input_states:
            obs_dict['vel_xy'] = obs['velocity']['vel_xy']
        if 'vel_ang_z' in input_states:
            obs_dict['vel_ang_z'] = obs['velocity']['vel_ang_z']
        if 'linear_speed' in input_states:
            speed_factor = 12.0
            linear_speed = np.array([obs['speed']['forward_speed'][0] / speed_factor])
            obs_dict['linear_speed'] = linear_speed
        if 'vec' in input_states:
            vec_array = np.array([loc_in_ev.x, loc_in_ev.y])
            obs_dict['vec'] = vec_array
        if 'traj' in input_states:
            traj_vec = []
            for traj_loc_in_ev in traj_locs:
                traj_vec.extend([traj_loc_in_ev.x / 100.0, traj_loc_in_ev.y / 100.0])
            obs_dict['traj'] = np.array(traj_vec)
            obs_dict['traj_plot'] = traj_plotter(obs_dict['traj'])
            obs_dict['traj_plot_rgb'] = traj_plotter_rgb(obs_dict['traj'])

        if 'cmd' in input_states:
            cmd_one_hot = [0] * 6
            cmd_one_hot[command] = 1
            cmd_array = np.array(cmd_one_hot)
            obs_dict['cmd'] = cmd_array
        if 'command' in input_states:
            obs_dict['command'] = np.array([command])
        if 'state' in input_states:
            state_list = []
            state_list.append(obs['control']['throttle'])
            state_list.append(obs['control']['steer'])
            state_list.append(obs['control']['brake'])
            state_list.append(obs['control']['gear']/5.0)
            state_list.append(obs['velocity']['vel_xy'])
            obs_dict['state'] = np.concatenate(state_list)

        birdview = obs['birdview']['masks']
        obs_dict.update({
            'birdview': birdview
        })

        if 'rgb' in input_states:
            central_rgb = obs['central_rgb']['data']
            central_rgb = np.transpose(central_rgb, [2, 0, 1])

            left_rgb = obs['left_rgb']['data']
            left_rgb = np.transpose(left_rgb, [2, 0, 1])

            right_rgb = obs['right_rgb']['data']
            right_rgb = np.transpose(right_rgb, [2, 0, 1])

            obs_dict.update({
                'central_rgb': central_rgb,
                'left_rgb': left_rgb,
                'right_rgb': right_rgb
            })

        return obs_dict

    @staticmethod
    def process_act(action, acc_as_action, train=True):
        if not train:
            action = action[0]
        if acc_as_action:
            acc, steer = action.astype(np.float64)
            if acc >= 0.0:
                throttle = acc
                brake = 0.0
            else:
                throttle = 0.0
                brake = np.abs(acc)
        else:
            throttle, steer, brake = action.astype(np.float64)

        throttle = np.clip(throttle, 0, 1)
        steer = np.clip(steer, -1, 1)
        brake = np.clip(brake, 0, 1)
        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        return control
