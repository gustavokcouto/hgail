import wandb
import torch as th

from rl_birdview.models.ppo_policy import PpoPolicy
from rl_train import get_obs_configs, get_env_wrapper_configs
from stable_baselines3.common.vec_env import SubprocVecEnv
from rl_train import get_obs_configs, get_env_wrapper_configs, env_eval_maker
from rl_birdview.utils.wandb_callback import WandbCallback


multi_env_eval_configs = [
    {"host": "localhost", "port": 2000, 'carla_map': 'Town02'},
]


if __name__ == '__main__':
    generate_rgb = True
    obs_configs = get_obs_configs(generate_rgb)
    env_wrapper_configs = get_env_wrapper_configs(generate_rgb)
    env_eval = SubprocVecEnv([lambda env_id=env_id, config=config: env_eval_maker(env_id, config, obs_configs, env_wrapper_configs, rendering=generate_rgb) for env_id, config in enumerate(multi_env_eval_configs)])
    for i in range(env_eval.num_envs):
        env_eval.set_attr('n_epoch', 0, indices=i)
        env_eval.set_attr('last_epoch', 0, indices=i)
        env_eval.set_attr('num_timesteps', 0, indices=i)
    _eval_step = int(1e5)
    n_steps_total = 12288
    wb_run_path = 'gustavokcouto/gail-carla2/1r49je2y'
    api = wandb.Api()
    wandb_run = api.run(wb_run_path)
    wandb_run_id = wandb_run.id
    parse_timesteps = lambda ckpt_file: int(ckpt_file.name.replace('ckpt/ckpt_', '').replace('.pth', ''))
    all_ckpts = [ckpt_file for ckpt_file in wandb_run.files() if 'ckpt/ckpt_' in ckpt_file.name]
    all_ckpts = [ckpt_file for ckpt_file in all_ckpts if not 'ckpt_latest' in ckpt_file.name]
    all_ckpts.sort(key=parse_timesteps)
    print(all_ckpts)
    wandb.init(project='gail-carla2')
    for ckpt_file in all_ckpts:
        num_timesteps = parse_timesteps(ckpt_file)
        print(num_timesteps)
        if (num_timesteps % _eval_step) < n_steps_total:
            ckpt_file.download(replace=True)
            ckpt_file_path = ckpt_file.name
            saved_variables = th.load(ckpt_file_path, map_location='cuda')
            train_kwargs = saved_variables['train_init_kwargs']

            policy = PpoPolicy(**saved_variables['policy_init_kwargs'])
            policy.load_state_dict(saved_variables['policy_state_dict'])
            policy = policy.to(policy.device)
            eval_video_path = f'video/eval_town2_{num_timesteps}.mp4'
            avg_ep_stat, avg_route_completion, ep_events = WandbCallback.evaluate_policy_town2(env_eval, policy, eval_video_path)
            wandb.log(avg_ep_stat, step=num_timesteps)
            wandb.log(avg_route_completion, step=num_timesteps)