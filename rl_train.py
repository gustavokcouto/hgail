# modified from https://github.com/zhejz/carla-roach/blob/main/train_rl.py

from pathlib import Path
import wandb
import torch as th

from carla_gym.envs import EndlessEnv, LeaderboardEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList

from rl_birdview.utils.rl_birdview_wrapper import RlBirdviewWrapper
from rl_birdview.models.ppo import PPO
from rl_birdview.models.ppo_policy import PpoPolicy
from rl_birdview.models.discriminator import Discriminator
from rl_birdview.utils.wandb_callback import WandbCallback

FAKE_BIRDVIEW = True
RESUME_LAST_TRAIN = True
GAIL = True
RGB_GAIL = False
TRAJ_PLOT = False

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
    'num_zombie_vehicles': [0, 150],
    'num_zombie_walkers': [0, 300],
    'weather_group': 'dynamic_1.0'
}

env_eval_configs = {
    'weather_group': 'simple',
    'routes_group': 'train'
}

multi_env_configs = [
    {"host": "192.168.0.6", "port": 2000, 'carla_map': 'Town01'},
    {"host": "192.168.0.6", "port": 2002, 'carla_map': 'Town01'},
    {"host": "192.168.0.6", "port": 2004, 'carla_map': 'Town01'},
    {"host": "192.168.0.6", "port": 2006, 'carla_map': 'Town01'},
    {"host": "192.168.0.6", "port": 2008, 'carla_map': 'Town01'},
    {"host": "192.168.0.6", "port": 2010, 'carla_map': 'Town01'},
]

multi_env_eval_configs = [
    {"host": "192.168.0.6", "port": 2012, 'carla_map': 'Town02'},
]

def get_obs_configs(rgb=False):
    obs_configs = {
        'hero': {
            'speed': {
                'module': 'actor_state.speed'
            },
            'control': {
                'module': 'actor_state.control'
            },
            'velocity': {
                'module': 'actor_state.velocity'
            },
            'birdview': {
                'module': 'birdview.chauffeurnet',
                'width_in_pixels': 192,
                'pixels_ev_to_bottom': 40,
                'pixels_per_meter': 5.0,
                'history_idx': [-16, -11, -6, -1],
                'scale_bbox': True,
                'scale_mask_col': 1.0
            },
            'route_plan': {
                'module': 'navigation.waypoint_plan',
                'steps': 20
            }
        }
    }
    if rgb:
        obs_configs['hero'].update({    
            'gnss': {
                'module': 'navigation.gnss'
            },    
            'central_rgb': {
                'module': 'camera.rgb',
                'fov': 90,
                'width': 256,
                'height': 144,
                'location': [1.2, 0.0, 1.3],
                'rotation': [0.0, 0.0, 0.0]
            },
            'left_rgb': {
                'module': 'camera.rgb',
                'fov': 90,
                'width': 256,
                'height': 144,
                'location': [1.2, -0.25, 1.3],
                'rotation': [0.0, 0.0, -45.0]
            },
            'right_rgb': {
                'module': 'camera.rgb',
                'fov': 90,
                'width': 256,
                'height': 144,
                'location': [1.2, 0.25, 1.3],
                'rotation': [0.0, 0.0, 45.0]
            }
        })
    return obs_configs


def get_env_wrapper_configs(rgb=True):
    env_wrapper_configs = {
        'input_states': ['control', 'state', 'vel_xy'],
        'acc_as_action': True
    }
    if rgb:
        env_wrapper_configs['input_states'].extend(['linear_speed', 'vec', 'cmd', 'command', 'traj', 'rgb'])
    return env_wrapper_configs


def env_maker(env_id, config, obs_configs, env_wrapper_configs, rendering=True):
    env = EndlessEnv(obs_configs=obs_configs, reward_configs=reward_configs,
                    terminal_configs=terminal_configs,
                    seed=2021, no_rendering=(not rendering), **env_configs, **config)
    env = RlBirdviewWrapper(env, **env_wrapper_configs, env_id=env_id, resume=RESUME_LAST_TRAIN)
    return env


def env_eval_maker(env_id, config, obs_configs, env_wrapper_configs, rendering=True):
    env = LeaderboardEnv(obs_configs=obs_configs, reward_configs=reward_configs,
                    terminal_configs=terminal_configs,
                    seed=2021, no_rendering=(not rendering), **env_eval_configs, **config)
    env = RlBirdviewWrapper(env, **env_wrapper_configs, env_id=env_id, resume=RESUME_LAST_TRAIN)
    return env


if __name__ == '__main__':
    generate_rgb = FAKE_BIRDVIEW or RGB_GAIL
    obs_configs = get_obs_configs(generate_rgb)
    env_wrapper_configs = get_env_wrapper_configs(generate_rgb)

    env = SubprocVecEnv([lambda env_id=env_id, config=config: env_maker(env_id, config, obs_configs, env_wrapper_configs, rendering=generate_rgb) for env_id, config in enumerate(multi_env_configs)])
    env_eval = SubprocVecEnv([lambda env_id=env_id, config=config: env_eval_maker(env_id, config, obs_configs, env_wrapper_configs, rendering=generate_rgb) for env_id, config in enumerate(multi_env_eval_configs)])
    if not RGB_GAIL:
        features_extractor_entry_point = 'rl_birdview.models.torch_layers:XtMaCNN'
    else:
        features_extractor_entry_point = 'rl_birdview.models.torch_layers:RGBXtMaCNN'

    policy_kwargs = {
        'observation_space': env.observation_space,
        'action_space': env.action_space,
        'policy_head_arch': [256, 256],
        'value_head_arch': [256, 256],
        'features_extractor_entry_point': features_extractor_entry_point,
        'features_extractor_kwargs': {'states_neurons': [256,256]},
        'distribution_entry_point': 'rl_birdview.models.distributions:BetaDistribution',
        'fake_birdview': FAKE_BIRDVIEW,
        'rgb_gail': RGB_GAIL,
        'traj_plot': TRAJ_PLOT
    }
    discriminator_kwargs = {
        'observation_space': env.observation_space,
        'action_space': env.action_space,
        'batch_size': 256,
        'disc_head_arch': [256, 256],
        'rgb_gail': RGB_GAIL,
        'traj_plot': TRAJ_PLOT
    }
    output_dir = Path('outputs')
    output_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint_path = output_dir / 'checkpoint.txt'
    wandb_run_id = None

    train_kwargs = {
        'initial_learning_rate': 2e-5,
        'gail': GAIL,
        'n_steps_total': 12288,
        'batch_size': 256,
        'n_epochs': 20,
        'gamma': 0.99,
        'gae_lambda': 0.9,
        'clip_range': 0.2,
        'clip_range_vf': 0.2,
        'ent_coef': 0.01,
        'explore_coef': 0.05,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'lr_decay': 0.96,
        'use_exponential_lr_decay': True,
        'gail_gamma': 0.004,
        'gail_gamma_decay': 1.0,
        'update_adv': False,
    }

    if RESUME_LAST_TRAIN:
        with open(last_checkpoint_path, 'r') as f:
            wb_run_path = f.read()
        api = wandb.Api()
        wandb_run = api.run(wb_run_path)
        wandb_run_id = wandb_run.id
        all_ckpts = [ckpt_file for ckpt_file in wandb_run.files() if 'ckpt_latest' in ckpt_file.name]
        ckpt_file = all_ckpts[0]
        ckpt_file.download(replace=True)
        ckpt_file_path = ckpt_file.name
        saved_variables = th.load(ckpt_file_path, map_location='cuda')
        train_kwargs = saved_variables['train_init_kwargs']

        policy = PpoPolicy(**saved_variables['policy_init_kwargs'])
        policy.load_state_dict(saved_variables['policy_state_dict'])

        discriminator = Discriminator(**saved_variables['discriminator_init_kwargs'])
        discriminator.load_state_dict(saved_variables['discriminator_state_dict'])

    else:
        policy = PpoPolicy(**policy_kwargs)
        discriminator = Discriminator(**discriminator_kwargs)
        if FAKE_BIRDVIEW:
            policy.gan_fake_birdview.pretrain()

    agent = PPO(
        policy=policy,
        discriminator=discriminator,
        env=env,
        **train_kwargs
    )

    wb_callback = WandbCallback(env, env_eval, wandb_run_id)
    callback = CallbackList([wb_callback])
    with open(last_checkpoint_path, 'w') as log_file:
        log_file.write(wandb.run.path)
    agent.learn(total_timesteps=1e8, callback=callback)
