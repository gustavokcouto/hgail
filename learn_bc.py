import torch.optim as optim
import numpy as np
import tqdm
import torch as th
from pathlib import Path
import wandb

from rl_train import get_obs_configs, get_env_wrapper_configs, env_maker 
from stable_baselines3.common.vec_env import SubprocVecEnv
from rl_birdview.utils.wandb_callback import WandbCallback
from rl_birdview.models.discriminator import ExpertDataset
from rl_birdview.models.ppo_policy import PpoPolicy


multi_env_configs = [
    {"host": "192.168.0.4", "port": 2000},
    {"host": "192.168.0.4", "port": 2002},
    {"host": "192.168.0.4", "port": 2004},
    {"host": "192.168.0.4", "port": 2006},
    {"host": "192.168.0.4", "port": 2008},
    {"host": "192.168.0.4", "port": 2010}
]

def learn_bc(policy, device, expert_loader, eval_loader, env, resume_last_train):
    output_dir = Path('outputs')
    output_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint_path = output_dir / 'checkpoint.txt'

    ckpt_dir = Path('ckpt')
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if resume_last_train:
        with open(last_checkpoint_path, 'r') as f:
            wb_run_path = f.read()
        api = wandb.Api()
        wandb_run = api.run(wb_run_path)
        wandb_run_id = wandb_run.id
        # all_ckpts = [ckpt_file for ckpt_file in wandb_run.files() if 'ckpt_latest' in ckpt_file.name]
        # ckpt_file = all_ckpts[0]
        # ckpt_file.download(replace=True)
        # ckpt_file_path = ckpt_file.name
        ckpt_path = (ckpt_dir / 'ckpt_latest.pth').as_posix()
        saved_variables = th.load(ckpt_path, map_location='cuda')
        train_kwargs = saved_variables['train_init_kwargs']
        start_ep = train_kwargs['start_ep']
        i_steps = train_kwargs['i_steps']

        policy.load_state_dict(saved_variables['policy_state_dict'])
        wandb.init(project='gail-carla2', mode='offline', id=wandb_run_id, resume='must')
    else:
        run = wandb.init(project='gail-carla2', mode='offline', reinit=True)
        gan_episodes = 100
        gan_steps = 0
        for _ in range(gan_episodes):
            train_debug = train_gan(policy.gan_fake_birdview, expert_loader)
            val_debug = val_gan(policy.gan_fake_birdview, eval_loader)
            gan_steps += len(expert_loader.dataset)
            wandb.log(train_debug, step=gan_steps)
            wandb.log(val_debug, step=gan_steps)

        run = run.finish()
        run = wandb.init(project='gail-carla2', mode='offline', reinit=True)
        with open(last_checkpoint_path, 'w') as log_file:
            log_file.write(wandb.run.path)
        start_ep = 0
        i_steps = 0

    video_path = Path('video')
    video_path.mkdir(parents=True, exist_ok=True)

    optimizer = optim.Adam(policy.parameters(), lr=1e-5)
    max_grad_norm = 0.5
    episodes = 200
    ent_weight = 0.01
    min_eval_loss = np.inf
    eval_step = int(1e5)
    steps_last_eval = 0
    expert_fake_birdview_loader = policy.gan_fake_birdview.fill_expert_dataset(expert_loader)

    for i_episode in tqdm.tqdm(range(start_ep, episodes)):
        total_loss = 0
        i_batch = 0
        policy = policy.train()
        # Expert dataset
        for expert_batch in expert_loader:
            expert_obs_dict, expert_action = expert_batch
            obs_tensor_dict = {
                'state': expert_obs_dict['state'].float().to(device),
                'birdview': expert_obs_dict['birdview'].float().to(device)
            }
            fake_birdview = expert_fake_birdview_loader.index_select(dim=0, index=expert_obs_dict['item_idx'].int())
            fake_birdview = fake_birdview.to(policy.device)
            expert_action = expert_action.to(device)

            # Get BC loss
            alogprobs, entropy_loss = policy.evaluate_actions_bc(obs_tensor_dict, fake_birdview, expert_action)
            bcloss = -alogprobs.mean()

            loss = bcloss + ent_weight * entropy_loss
            total_loss += loss
            i_batch += 1
            i_steps += expert_obs_dict['state'].shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_eval_loss = 0
        i_eval_batch = 0
        eval_fake_birdview_loader = policy.gan_fake_birdview.fill_expert_dataset(expert_loader)
        for expert_batch in eval_loader:
            expert_obs_dict, expert_action = expert_batch
            obs_tensor_dict = {
                'state': expert_obs_dict['state'].float().to(device),
                'birdview': expert_obs_dict['birdview'].float().to(device)
            }
            fake_birdview = eval_fake_birdview_loader.index_select(dim=0, index=expert_obs_dict['item_idx'].int())
            fake_birdview = fake_birdview.to(policy.device)
            expert_action = expert_action.to(device)

            # Get BC loss
            with th.no_grad():
                alogprobs, entropy_loss = policy.evaluate_actions_bc(obs_tensor_dict, fake_birdview, expert_action)
            bcloss = -alogprobs.mean()

            eval_loss = bcloss + ent_weight * entropy_loss
            total_eval_loss += eval_loss
            i_eval_batch += 1
        
        loss = total_loss / i_batch
        eval_loss = total_eval_loss / i_eval_batch
        wandb.log({
            'loss': loss,
            'eval_loss': eval_loss,
        }, step=i_steps)

        if i_steps - steps_last_eval > eval_step:
            eval_video_path = (video_path / f'bc_eval_{i_steps}.mp4').as_posix()
            for i in range(env.num_envs):
                env.set_attr('n_epoch', i_episode, indices=i)
                env.set_attr('num_timesteps', i_steps, indices=i)
            avg_ep_stat, avg_route_completion, ep_events = WandbCallback.evaluate_policy(env, policy, eval_video_path)
            env.reset()
            wandb.log(avg_ep_stat, step=i_steps)
            wandb.log(avg_route_completion, step=i_steps)
            steps_last_eval = i_steps

        if min_eval_loss > eval_loss:
            ckpt_path = (ckpt_dir / f'bc_ckpt_{i_episode}_min_eval.pth').as_posix()
            th.save(
                {'policy_state_dict': policy.state_dict()},
               ckpt_path
            )
            min_eval_loss = eval_loss

        train_init_kwargs = {
            'start_ep': i_episode,
            'i_steps': i_steps
        } 
        ckpt_path = (ckpt_dir / 'ckpt_latest.pth').as_posix()
        th.save({'policy_state_dict': policy.state_dict(),
                 'train_init_kwargs': train_init_kwargs},
                ckpt_path)
        wandb.save(f'./{ckpt_path}')
    run = run.finish()
        

def train_gan(gan_fake_birdview, train_dataloader):
    gan_disc_losses, gan_generator_losses, gan_pixel_losses, gan_losses = [], [], [], []
    for i, batch in enumerate(train_dataloader):
        obs_dict, _ = batch
        gan_disc_loss, gan_generator_loss, gan_pixel_loss, gan_loss = gan_fake_birdview.train_batch(obs_dict)
        gan_disc_losses.append(gan_disc_loss)
        gan_generator_losses.append(gan_generator_loss)
        gan_pixel_losses.append(gan_pixel_loss)
        gan_losses.append(gan_loss)

    train_debug = {
        "train_gan/gan_disc_loss": np.mean(gan_disc_losses),
        "train_gan/gan_generator_loss": np.mean(gan_generator_losses),
        "train_gan/gan_pixel_loss": np.mean(gan_pixel_losses),
        "train_gan/gan_loss": np.mean(gan_losses)
    }

    return train_debug


def val_gan(gan_fake_birdview, val_dataloader):
    gan_disc_losses, gan_generator_losses, gan_pixel_losses, gan_losses = [], [], [], []
    for i, batch in enumerate(val_dataloader):
        obs_dict, _ = batch
        gan_disc_loss, gan_generator_loss, gan_pixel_loss, gan_loss = gan_fake_birdview.val_batch(obs_dict)
        gan_disc_losses.append(gan_disc_loss)
        gan_generator_losses.append(gan_generator_loss)
        gan_pixel_losses.append(gan_pixel_loss)
        gan_losses.append(gan_loss)

    val_debug = {
        "val_gan/gan_disc_loss": np.mean(gan_disc_losses),
        "val_gan/gan_generator_loss": np.mean(gan_generator_losses),
        "val_gan/gan_pixel_loss": np.mean(gan_pixel_losses),
        "val_gan/gan_loss": np.mean(gan_losses)
    }

    return val_debug


if __name__ == '__main__':
    resume_last_train = True
    obs_configs = get_obs_configs(rgb=True)
    env_wrapper_configs = get_env_wrapper_configs(rgb=True)
    env = SubprocVecEnv([lambda env_id=env_id, config=config: env_maker(env_id, config, obs_configs, env_wrapper_configs, rendering=True) for env_id, config in enumerate(multi_env_configs)])
    env.reset()

    # network
    policy_kwargs = {
        'observation_space': env.observation_space,
        'action_space': env.action_space,
        'policy_head_arch': [256, 256],
        'value_head_arch': [256, 256],
        'features_extractor_entry_point': 'rl_birdview.models.torch_layers:XtMaCNN',
        'features_extractor_kwargs': {'states_neurons': [256,256]},
        'distribution_entry_point': 'rl_birdview.models.distributions:BetaDistribution',
        'fake_birdview': True
    }

    device = 'cuda'

    policy = PpoPolicy(**policy_kwargs)
    policy.to(device)

    batch_size = 32

    gail_train_loader = th.utils.data.DataLoader(
        ExpertDataset(
            'gail_experts',
            n_routes=8,
            n_eps=1,
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    
    gail_val_loader = th.utils.data.DataLoader(
        ExpertDataset(
            'gail_experts',
            n_routes=2,
            n_eps=1,
            route_start=8
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    learn_bc(policy, device, gail_train_loader, gail_val_loader, env, resume_last_train)