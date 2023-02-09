# modified from https://github.com/rohitrango/BC-regularized-GAIL/blob/master/a2c_ppo_acktr/algo/gail.py

from pathlib import Path
from typing import Union, Dict, Tuple, Any
from functools import partial
import time
import gym
import torch as th
import torch.nn as nn
import numpy as np
import pandas as pd

from torchvision import transforms
from rl_birdview.models.torch_layers import DiscXtMaCNN, RGBDiscXtMaCNN
from PIL import Image
from stable_baselines3.common.running_mean_std import RunningMeanStd


class Discriminator(nn.Module):

    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 batch_size,
                 disc_head_arch=[256, 256],
                 rgb_gail=False,
                 start_update=0):

        super(Discriminator, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.batch_size = batch_size
        self.rgb_gail = rgb_gail

        if th.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.optimizer_class = th.optim.Adam

        if self.rgb_gail:
            self.features_extractor = RGBDiscXtMaCNN(observation_space, action_space)
            self.optimizer_kwargs = {'lr':1.0e-4, 'eps': 1e-5}
            self.thre = 10
            self.pre_epoch = 6
            self.epoch = 4

        else:
            self.features_extractor = DiscXtMaCNN(observation_space, action_space)
            self.optimizer_kwargs = {'lr':1.0e-4, 'eps': 1e-5}
            self.thre = 10
            self.pre_epoch = 6
            self.epoch = 4

        # best_so_far
        # self.net_arch = [dict(pi=[256, 128, 64], vf=[128, 64])]
        self.disc_head_arch = list(disc_head_arch)
        self.activation_fn = nn.ReLU

        self._build()
        self.expert_loader = th.utils.data.DataLoader(
                ExpertDataset(
                    'gail_experts',
                    n_routes=8,
                    n_eps=1,
                ),
                batch_size=self.batch_size,
                shuffle=True,
            )
        self.max_grad_norm = 0.5
        self.i_update = start_update
        self.cliprew_down = -10
        self.cliprew_up = 10

        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())
        self.disc_debug = {}

    def _build(self) -> None:
        last_layer_dim_disc = self.features_extractor.features_dim
        disc_net = []
        for layer_size in self.disc_head_arch:
            disc_net.append(nn.Linear(last_layer_dim_disc, layer_size))
            disc_net.append(self.activation_fn())
            last_layer_dim_disc = layer_size

        disc_net.append(nn.Linear(last_layer_dim_disc, 1))
        self.disc_head = nn.Sequential(*disc_net).to(self.device)

        self.optimizer = self.optimizer_class(self.parameters(), **self.optimizer_kwargs)

    def forward(self, obs_dict, action):
        '''
        used in collect_rollouts(), do not clamp actions
        '''
        obs_dict = dict([(obs_key, obs_item.to(self.device)) for obs_key, obs_item in obs_dict.items()])

        if self.rgb_gail:
            rgb_array = []
            for rgb_key in ['central_rgb', 'left_rgb', 'right_rgb']:
                rgb_img = obs_dict[rgb_key].float() / 255.0
                rgb_array.append(rgb_img)

            rgb = th.cat(rgb_array, dim=1).to(self.device)
            cmd = obs_dict['cmd'].to(self.device)
            traj = obs_dict['traj'].to(self.device)
            state = obs_dict['state'].to(self.device)
            action = action.to(self.device)
            features = self.features_extractor(rgb, cmd, traj, state, action)
        else:
            birdview = obs_dict['birdview'].to(self.device)
            birdview = birdview.float() / 255.0
            state = obs_dict['state'].to(self.device)
            action = action.to(self.device)
            features = self.features_extractor(birdview, state, action)

        disc = self.disc_head(features)

        return disc

    def get_init_kwargs(self) -> Dict[str, Any]:
        init_kwargs = dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
            batch_size=self.batch_size,
            disc_head_arch=self.disc_head_arch,
            rgb_gail=self.rgb_gail,
            start_update=self.i_update
        )
        return init_kwargs

    @classmethod
    def load(cls, path):
        if th.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        saved_variables = th.load(path, map_location=device)
        # Create policy object
        model = cls(**saved_variables['policy_init_kwargs'])
        # Load weights
        model.load_state_dict(saved_variables['policy_state_dict'])
        model.to(device)
        return model, saved_variables['train_init_kwargs']

    def compute_grad_pen(self,
                         expert_obs_dict,
                         expert_action,
                         policy_obs_dict,
                         policy_action,
                         lambda_=10):

        # Change state values
        alpha = th.rand(expert_obs_dict['birdview'].size(0), 1, 1, 1)
        mixup_obs_dict = {}
        expert_obs_dict = dict([(obs_key, obs_item.to(self.device)) for obs_key, obs_item in expert_obs_dict.items()])
        expert_action = expert_action.to(self.device)
        policy_obs_dict = dict([(obs_key, obs_item.to(self.device)) for obs_key, obs_item in policy_obs_dict.items()])
        policy_action = policy_action.to(self.device)
        if self.rgb_gail:
            for obs_key in ['central_rgb', 'left_rgb', 'right_rgb']:
                alpha_obs = alpha.expand_as(expert_obs_dict[obs_key]).to(expert_obs_dict[obs_key].device)
                mixup_obs_dict[obs_key] = alpha_obs * expert_obs_dict[obs_key] + \
                    (1 - alpha_obs) * policy_obs_dict[obs_key]
                mixup_obs_dict[obs_key].requires_grad = True
            
            alpha = alpha.view(expert_obs_dict['central_rgb'].size(0), 1)

            for obs_key in ['cmd', 'traj', 'state']:
                alpha_obs = alpha.expand_as(expert_obs_dict[obs_key]).to(expert_obs_dict[obs_key].device)
                mixup_obs_dict[obs_key] = alpha_obs * expert_obs_dict[obs_key] + \
                    (1 - alpha_obs) * policy_obs_dict[obs_key]
                mixup_obs_dict[obs_key].requires_grad = True

        else:
            alpha_obs = alpha.expand_as(expert_obs_dict['birdview']).to(expert_obs_dict['birdview'].device)
            mixup_obs_dict['birdview'] = alpha_obs * expert_obs_dict['birdview'] + \
                (1 - alpha_obs) * policy_obs_dict['birdview']
            mixup_obs_dict['birdview'].requires_grad = True

            alpha = alpha.view(expert_obs_dict['birdview'].size(0), 1)

            alpha_obs = alpha.expand_as(
                expert_obs_dict['state']).to(expert_obs_dict['state'].device)
            mixup_obs_dict['state'] = alpha_obs * expert_obs_dict['state'] + \
                (1 - alpha_obs) * policy_obs_dict['state']
            mixup_obs_dict['state'].requires_grad = True

        alpha_action = alpha.expand_as(expert_action).to(expert_action.device)
        mixup_action = alpha_action * expert_action + \
            (1 - alpha_action) * policy_action
        mixup_action.requires_grad = True

        inputs_grad = (*mixup_obs_dict.values(), mixup_action)

        disc = self.forward(mixup_obs_dict, mixup_action)
        ones = th.ones(disc.size()).to(disc.device)
        
        grad = th.autograd.grad(
            outputs=disc,
            inputs=inputs_grad,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        grad = grad.view(grad.size(0), -1)

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def update(self, rollout_buffer):
        loss = 0
        expert_ac_loss = 0
        policy_ac_loss = 0
        g_loss = 0.0
        gp = 0.0
        update_samples = 0
        policy_reward = 0
        expert_reward = 0

        disc_epoch = self.epoch
        if self.i_update < self.thre:
            disc_epoch += (self.pre_epoch - self.epoch) * \
                (self.thre - self.i_update) / self.thre  # Warm up
            disc_epoch = int(disc_epoch)

        for _ in range(disc_epoch):
            for expert_batch, policy_batch in zip(self.expert_loader, rollout_buffer.get(self.batch_size)):
                policy_obs_dict = policy_batch.observations
                policy_action = policy_batch.actions

                expert_obs_dict, expert_action = expert_batch

                if policy_obs_dict['birdview'].size() != expert_obs_dict['birdview'].size():
                    continue

                policy_d = self.forward(policy_obs_dict, policy_action)
                policy_reward += policy_d.sum().item()

                expert_d = self.forward(expert_obs_dict, expert_action)
                expert_reward += expert_d.sum().item()

                expert_loss = th.mean(th.tanh(expert_d))
                policy_loss = th.mean(th.tanh(policy_d))

                n_samples = policy_obs_dict['state'].shape[0]
                expert_ac_loss += (expert_loss).item() * n_samples
                policy_ac_loss += (policy_loss).item() * n_samples

                wd = expert_loss - policy_loss
                grad_pen = self.compute_grad_pen(expert_obs_dict, expert_action, policy_obs_dict, policy_action)

                loss += (-wd + grad_pen).item() * n_samples
                g_loss += (wd).item() * n_samples
                gp += (grad_pen).item() * n_samples
                update_samples += n_samples

                self.optimizer.zero_grad()

                (-wd + grad_pen).backward()
                nn.utils.clip_grad_norm_(
                    self.parameters(), self.max_grad_norm)
                self.optimizer.step()

        self.disc_debug = {
            "disc/loss": loss/update_samples,
            "disc/policy_reward": policy_reward/update_samples,
            "disc/expert_reward": expert_reward/update_samples,
            "disc/gail_loss": g_loss/update_samples,
            "disc/gradient_penalty": gp/update_samples,
            "disc/expert_loss": expert_ac_loss/update_samples,
            "disc/policy_loss": policy_ac_loss/update_samples,
            "disc/epochs": disc_epoch,
            "disc/update_samples": update_samples
        }
        
        self.i_update += 1
        return

    def predict_reward(self, obs_dict, action, gamma, dones, update_rms=True):
        with th.no_grad():
            d = self.forward(obs_dict, action)

            s = th.sigmoid(d)
            reward = -(1 - s).log()
            reward = reward.cpu()
            reward = th.clamp(reward, self.cliprew_down, self.cliprew_up)

            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                non_terminal = 1.0 - dones
                self.returns = self.returns * non_terminal * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

            return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)


class ExpertDataset(th.utils.data.Dataset):
    def __init__(self, dataset_directory, n_routes=1, n_eps=1, route_start=0, ep_start=0):
        self.dataset_path = Path(dataset_directory)
        self.length = 0
        self.get_idx = []
        self.trajs_states = []
        self.trajs_actions = []

        for route_idx in range(route_start, route_start + n_routes):
            for ep_idx in range(ep_start, ep_start + n_eps):
                route_path = self.dataset_path / ('route_%02d' % route_idx) / ('ep_%02d' % ep_idx)
                route_df = pd.read_json(route_path / 'episode.json')
                traj_length = route_df.shape[0]
                self.length += traj_length
                for step_idx in range(traj_length):
                    self.get_idx.append((route_idx, ep_idx, step_idx))
                    state_dict = {}
                    for state_key in route_df.columns:
                        state_dict[state_key] = route_df.iloc[step_idx][state_key]
                    self.trajs_states.append(state_dict)
                    self.trajs_actions.append(th.Tensor(route_df.iloc[step_idx]['actions']))

        self.trajs_actions = th.stack(self.trajs_actions)
        self.actual_obs = [None for _ in range(self.length)]
        self.preprocess = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return self.length

    def process_image(self, image_path):
        image_array = Image.open(image_path)
        image_array = np.transpose(image_array, [2, 0, 1])
        image_tensor = th.as_tensor(image_array)
        return image_tensor

    def __getitem__(self, j):
        route_idx, ep_idx, step_idx = self.get_idx[j]
        if self.actual_obs[j] is None:
            # Load only the first time, images in uint8 are supposed to be light
            ep_dir = self.dataset_path / 'route_{:0>2d}/ep_{:0>2d}'.format(route_idx, ep_idx)
            masks_list = []
            for mask_index in range(1):
                mask_tensor = self.process_image(ep_dir / 'birdview_masks/{:0>4d}_{:0>2d}.png'.format(step_idx, mask_index))
                masks_list.append(mask_tensor)
            birdview = th.cat(masks_list)

            central_rgb = self.process_image(ep_dir / 'central_rgb/{:0>4d}.png'.format(step_idx))
            left_rgb = self.process_image(ep_dir / 'left_rgb/{:0>4d}.png'.format(step_idx))
            right_rgb = self.process_image(ep_dir / 'right_rgb/{:0>4d}.png'.format(step_idx))

            obs_dict = {
                'birdview': birdview,
                'central_rgb': central_rgb,
                'left_rgb': left_rgb,
                'right_rgb': right_rgb,
                'item_idx': j
            }

            state_dict = self.trajs_states[j]
            for state_key in state_dict:
                obs_dict[state_key] = th.Tensor(state_dict[state_key])
            self.actual_obs[j] = obs_dict
        else:
            obs_dict = self.actual_obs[j]

        return obs_dict, self.trajs_actions[j]
