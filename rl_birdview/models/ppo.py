# modified from https://github.com/zhejz/carla-roach/blob/main/agents/rl_birdview/models/ppo.py

import logging
import time
import torch as th
import numpy as np
from collections import deque
from torch.nn import functional as F

from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance

from .ppo_buffer import PpoBuffer
from rl_birdview.models.discriminator import ExpertDataset


class PPO():
    def __init__(self, policy, discriminator, env,
                 initial_learning_rate: float = 1e-5,
                 gail = True,
                 n_steps_total: int = 8192,
                 batch_size: int = 256,
                 n_epochs: int = 20,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.9,
                 clip_range: float = 0.2,
                 clip_range_vf: float = None,
                 ent_coef: float = 0.05,
                 explore_coef: float = 0.05,
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 lr_decay=1.0,
                 use_exponential_lr_decay=False,
                 update_adv=False,
                 start_num_timesteps: int = 0,
                 gail_gamma=0.0,
                 gail_gamma_decay=1.0,
                 start_update: int = 0):

        self.policy = policy
        self.discriminator = discriminator
        self.env = env
        self.learning_rate = initial_learning_rate
        self.gail = gail
        self.n_steps_total = n_steps_total
        self.n_steps = n_steps_total//env.num_envs
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.ent_coef = ent_coef
        self.explore_coef = explore_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.lr_decay = lr_decay
        self.use_exponential_lr_decay = use_exponential_lr_decay
        self.update_adv = update_adv
        self.start_num_timesteps = start_num_timesteps
        self.num_timesteps = start_num_timesteps
        self.gail_gamma = gail_gamma
        self.gail_gamma_decay = gail_gamma_decay

        self._last_obs = None
        self._last_dones = None
        self.ep_stat_buffer = None

        self.buffer = PpoBuffer(self.n_steps, self.env.observation_space, self.env.action_space,
                                gamma=self.gamma, gae_lambda=self.gae_lambda, n_envs=self.env.num_envs)
        self.policy = self.policy.to(self.policy.device)
        self.discriminator = self.discriminator.to(self.discriminator.device)

        model_parameters = filter(lambda p: p.requires_grad, self.policy.parameters())
        total_params = sum([np.prod(p.size()) for p in model_parameters])
        disc_parameters = filter(lambda p: p.requires_grad, self.discriminator.parameters())
        total_params += sum([np.prod(p.size()) for p in disc_parameters])
        print(f'trainable parameters: {total_params/1000000:.2f}M')
        self.logger = logging.getLogger(__name__)
        self.train_debug = {}
        self.i_update = start_update

    def collect_rollouts(self, env: VecEnv, callback: BaseCallback,
                         rollout_buffer: PpoBuffer, n_rollout_steps: int) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        rollout_buffer.reset()

        self.action_statistics = []
        self.mu_statistics = []
        self.sigma_statistics = []
        start_route_completion = []

        while n_steps < n_rollout_steps:
            actions, values, log_probs, mu, sigma, _, fake_birdview = self.policy.forward(self._last_obs)
            self.action_statistics.append(actions)
            self.mu_statistics.append(mu)
            self.sigma_statistics.append(sigma)

            new_obs, rewards, dones, infos = env.step(actions)

            if callback.on_step() is False:
                return False

            if n_steps == 0:
                for info in infos:
                    start_route_completion.append(info['route_completion'])

            n_steps += 1
            self.num_timesteps += env.num_envs

            # update_info_buffer
            for i in np.where(dones)[0]:
                self.ep_stat_buffer.append(infos[i]['episode_stat'])
                if n_steps < n_rollout_steps:
                    route_completion = infos[i]['route_completion']
                    for dict_key in route_completion:
                        route_completion[dict_key] -= start_route_completion[i][dict_key]
                    self.route_completion_buffer.append(route_completion)

            for i in np.where(self._last_dones)[0]:
                start_route_completion[i] = infos[i]['route_completion']

            rollout_buffer.add(self._last_obs, actions, rewards, self._last_dones, values, log_probs, mu, sigma, fake_birdview, infos)
            self._last_obs = new_obs
            self._last_dones = dones

        # update_info_buffer
        for info_idx in range(len(infos)):
            route_completion = infos[info_idx]['route_completion']
            for dict_key in route_completion:
                route_completion[dict_key] -= start_route_completion[info_idx][dict_key]
            self.route_completion_buffer.append(route_completion)
        last_values = self.policy.forward_value(self._last_obs)
        if self.gail:
            self.discriminator.update(self.buffer)
            for step in range(rollout_buffer.buffer_size):
                obs_dict = dict([(obs_key, th.as_tensor(obs_item[step])) for obs_key, obs_item in rollout_buffer.observations.items()])
                rollout_buffer.rewards[step] = self.discriminator.predict_reward(
                    obs_dict,
                    th.as_tensor(rollout_buffer.actions[step]),
                    rollout_buffer.gamma,
                    th.as_tensor(rollout_buffer.dones[step]),
                ).detach().numpy().reshape(-1)
        rollout_buffer.compute_returns_and_advantage(last_values, dones=self._last_dones)

        return True

    def train(self):
        for param_group in self.policy.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

        entropy_losses, exploration_losses, pg_losses, bc_losses, value_losses, losses = [], [], [], [], [], []
        clip_fractions = []
        approx_kl_divs = []

        # train for gradient_steps epochs
        epoch = 0
        data_len = int(self.buffer.buffer_size * self.buffer.n_envs / self.batch_size)
        if self.policy.fake_birdview:
            data_loader_fake_birdview = self.policy.gan_fake_birdview.fill_expert_dataset(self.discriminator.expert_loader)
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            self.buffer.start_caching(self.batch_size)
            # while self.buffer.sample_queue.qsize() < 3:
                # time.sleep(0.01)
            for i in range(data_len):

                if self.buffer.sample_queue.empty():
                    while self.buffer.sample_queue.empty():
                        # print(f'buffer_empty: {self.buffer.sample_queue.qsize()}')
                        time.sleep(0.01)
                rollout_data = self.buffer.sample_queue.get()

                values, log_prob, entropy_loss, exploration_loss, distribution = self.policy.evaluate_actions(
                    rollout_data.observations, rollout_data.actions, rollout_data.exploration_suggests, rollout_data.fake_birdviews,
                    detach_values=False)
                # Normalize advantage
                advantages = rollout_data.advantages
                # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                clip_fraction = th.mean((th.abs(ratio - 1) > self.clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                # Expert dataset
                for expert_batch in self.discriminator.expert_loader:
                    expert_obs_dict, expert_action = expert_batch
                    obs_tensor_dict = dict([(obs_key, obs_item.float().to(self.policy.device)) for obs_key, obs_item in expert_obs_dict.items()])
                    fake_birdview = None
                    if self.policy.fake_birdview:
                        fake_birdview = data_loader_fake_birdview.index_select(dim=0, index=expert_obs_dict['item_idx'].int())
                        fake_birdview = fake_birdview.to(self.policy.device)
                    expert_action = expert_action.to(self.policy.device)
                    # Get BC loss
                    alogprobs, _ = self.policy.evaluate_actions_bc(obs_tensor_dict, fake_birdview, expert_action)
                    bcloss = -alogprobs.mean()
                    # action loss is weighted sum
                    action_loss = self.gail_gamma * bcloss + (1 - self.gail_gamma) * policy_loss
                    # Multiply this coeff with decay factor
                    break

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(values - rollout_data.old_values,
                                                                     -self.clip_range_vf, self.clip_range_vf)
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)

                loss = action_loss + self.vf_coef * value_loss \
                    + self.ent_coef * entropy_loss + self.explore_coef * exploration_loss

                losses.append(loss.item())
                pg_losses.append(policy_loss.item())
                bc_losses.append(bcloss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                exploration_losses.append(exploration_loss.item())

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                with th.no_grad():
                    old_distribution = self.policy.action_dist.proba_distribution(
                        rollout_data.old_mu, rollout_data.old_sigma)
                    kl_div = th.distributions.kl_divergence(old_distribution.distribution, distribution)

                approx_kl_divs.append(kl_div.mean().item())

            # update advantages
            if self.update_adv:
                self.buffer.update_values(self.policy)
                last_values = self.policy.forward_value(self._last_obs)
                self.buffer.compute_returns_and_advantage(last_values, dones=self._last_dones)

        explained_var = explained_variance(self.buffer.returns.flatten(), self.buffer.values.flatten())

        if self.use_exponential_lr_decay:
            self.learning_rate *= self.lr_decay

        if self.gail_gamma is not None:
            self.gail_gamma *= self.gail_gamma_decay

        # Logs
        self.train_debug.update({
            "train/entropy_loss": np.mean(entropy_losses),
            "train/exploration_loss": np.mean(exploration_losses),
            "train/policy_gradient_loss": np.mean(pg_losses),
            "train/behavior_cloning_loss": np.mean(bc_losses),
            "train/value_loss": np.mean(value_losses),
            "train/last_epoch_kl": np.mean(approx_kl_divs),
            "train/clip_fraction": np.mean(clip_fractions),
            "train/loss": np.mean(losses),
            "train/explained_variance": explained_var,
            "train/clip_range": self.clip_range,
            "train/train_epoch": epoch,
            "train/learning_rate": self.learning_rate,
            "train/gail_gamma": self.gail_gamma
        })


    def train_gan(self):
        gan_disc_losses, gan_generator_losses, gan_pixel_losses, gan_losses = [], [], [], []
        gan_epochs = 4
        gan_batch_size = 32
        for _ in range(gan_epochs):
            for policy_batch in self.buffer.get(gan_batch_size):
                gan_disc_loss, gan_generator_loss, gan_pixel_loss, gan_loss = self.policy.gan_fake_birdview.train_batch(policy_batch.observations, self.num_timesteps)
                gan_disc_losses.append(gan_disc_loss)
                gan_generator_losses.append(gan_generator_loss)
                gan_pixel_losses.append(gan_pixel_loss)
                gan_losses.append(gan_loss)

        self.train_debug.update({
            "train_gan/gan_disc_loss": np.mean(gan_disc_losses),
            "train_gan/gan_generator_loss": np.mean(gan_generator_losses),
            "train_gan/gan_pixel_loss": np.mean(gan_pixel_losses),
            "train_gan/gan_loss": np.mean(gan_losses)
        })
    
    def learn(self, total_timesteps, callback=None, seed=2021):
        # reset env seed
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)
        self.env.seed(seed)

        self.start_time = time.time()

        self.t_train_values = 0.0

        self._last_obs = self.env.reset()
        self._last_dones = np.zeros((self.env.num_envs,), dtype=np.bool)

        callback.init_callback(self)

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            self.ep_stat_buffer = []
            self.route_completion_buffer = []
            callback.on_rollout_start()
            t0 = time.time()
            self.policy = self.policy.train()
            continue_training = self.collect_rollouts(self.env, callback, self.buffer, self.n_steps)
            self.t_rollout = time.time() - t0
            callback.on_rollout_end()

            if continue_training is False:
                break

            t0 = time.time()
            self.train()
            if self.policy.fake_birdview:
                self.train_gan()
            self.t_train = time.time() - t0
            callback.on_training_end()
            self.i_update += 1

        return self

    def _get_init_kwargs(self):
        init_kwargs = dict(
            initial_learning_rate=self.learning_rate,
            gail=self.gail,
            n_steps_total=self.n_steps_total,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            clip_range_vf=self.clip_range_vf,
            ent_coef=self.ent_coef,
            explore_coef=self.explore_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
            lr_decay=self.lr_decay,
            use_exponential_lr_decay=self.use_exponential_lr_decay,
            update_adv=self.update_adv,
            gail_gamma=self.gail_gamma,
            gail_gamma_decay=self.gail_gamma_decay,
            start_num_timesteps=self.num_timesteps,
            start_update=self.i_update
        )
        return init_kwargs

    def save(self, path: str) -> None:
        th.save({'policy_state_dict': self.policy.state_dict(),
                 'discriminator_state_dict': self.discriminator.state_dict(),
                 'policy_init_kwargs': self.policy.get_init_kwargs(),
                 'discriminator_init_kwargs': self.discriminator.get_init_kwargs(),
                 'train_init_kwargs': self._get_init_kwargs()},
                path)

    def get_env(self):
        return self.env
