# modified from https://github.com/zhejz/carla-roach/blob/main/agents/rl_birdview/models/torch_layers.py

"""Policies: abstract base class and concrete implementations."""

import torch as th
import torch.nn as nn
import numpy as np
import gym

from torchvision.models.resnet import resnet18, resnet34
from torchvision import transforms


class XtMaCNN(nn.Module):
    '''
    Inspired by https://github.com/xtma/pytorch_car_caring
    '''

    def __init__(self, observation_space, features_dim=256, states_neurons=[256]):
        super().__init__()
        self.features_dim = features_dim

        n_input_channels = observation_space['birdview'].shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 8, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space['birdview'].sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten+states_neurons[-1], 512), nn.ReLU(),
                                    nn.Linear(512, features_dim), nn.ReLU())

        states_neurons = [observation_space['state'].shape[0]] + states_neurons
        self.state_linear = []
        for i in range(len(states_neurons)-1):
            self.state_linear.append(nn.Linear(states_neurons[i], states_neurons[i+1]))
            self.state_linear.append(nn.ReLU())
        self.state_linear = nn.Sequential(*self.state_linear)

        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, birdview, state):
        x = self.cnn(birdview)
        latent_state = self.state_linear(state)

        x = th.cat((x, latent_state), dim=1)
        x = self.linear(x)
        return x


class DiscXtMaCNN(nn.Module):
    '''
    Inspired by https://github.com/xtma/pytorch_car_caring
    '''

    def __init__(self, observation_space, action_space, features_dim=256, states_neurons=[256]):
        super().__init__()
        self.features_dim = features_dim

        n_input_channels = observation_space['birdview'].shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 8, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space['birdview'].sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten+states_neurons[-1], 512), nn.ReLU(),
                                    nn.Linear(512, features_dim), nn.ReLU())

        states_neurons = [observation_space['state'].shape[0] + action_space.shape[0]] + states_neurons
        self.state_linear = []
        for i in range(len(states_neurons)-1):
            self.state_linear.append(nn.Linear(states_neurons[i], states_neurons[i+1]))
            self.state_linear.append(nn.ReLU())
        self.state_linear = nn.Sequential(*self.state_linear)

        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, birdview, state, action):
        x = self.cnn(birdview)
        liner_input = th.cat((state, action), dim=1)
        latent_linear = self.state_linear(liner_input)

        x = th.cat((x, latent_linear), dim=1)
        x = self.linear(x)
        return x


class RGBDiscXtMaCNN(nn.Module):
    '''
    Inspired by https://github.com/xtma/pytorch_car_caring
    '''

    def __init__(self, observation_space, action_space, features_dim=256, states_neurons=[256]):
        super().__init__()
        self.features_dim = features_dim

        n_input_channels = 9

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 8, kernel_size=5, stride=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 16, kernel_size=5, stride=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(gym.spaces.Box(low=0, high=255, shape=(9, observation_space['central_rgb'].shape[1], observation_space['central_rgb'].shape[2]), dtype=np.uint8).sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten+states_neurons[-1], 512), nn.ReLU(),
                                    nn.Linear(512, features_dim), nn.ReLU())

        states_neurons = [observation_space['state'].shape[0] + action_space.shape[0] + 16] + states_neurons
        self.state_linear = []
        for i in range(len(states_neurons)-1):
            self.state_linear.append(nn.Linear(states_neurons[i], states_neurons[i+1]))
            self.state_linear.append(nn.ReLU())
        self.state_linear = nn.Sequential(*self.state_linear)

        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, rgb, cmd, traj, state, action):
        x = self.cnn(rgb)
        liner_input = th.cat((cmd, traj, state, action), dim=1)
        latent_linear = self.state_linear(liner_input)

        x = th.cat((x, latent_linear), dim=1)
        x = self.linear(x)
        return x


class RGBXtMaCNN(nn.Module):
    '''
    Inspired by https://github.com/xtma/pytorch_car_caring
    '''

    def __init__(self, observation_space, features_dim=256, states_neurons=[256]):
        super().__init__()
        self.features_dim = features_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(9, 8, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(gym.spaces.Box(low=0, high=255, shape=(9, observation_space['central_rgb'].shape[1], observation_space['central_rgb'].shape[2]), dtype=np.uint8).sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten+states_neurons[-1], 512), nn.ReLU(),
                                    nn.Linear(512, features_dim), nn.ReLU())

        states_neurons = [observation_space['state'].shape[0] + 16] + states_neurons
        self.state_linear = []
        for i in range(len(states_neurons)-1):
            self.state_linear.append(nn.Linear(states_neurons[i], states_neurons[i+1]))
            self.state_linear.append(nn.ReLU())
        self.state_linear = nn.Sequential(*self.state_linear)

        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, rgb, cmd, traj, state):
        x = self.cnn(rgb)

        linear_input = th.cat((cmd, traj, state), 1)
        latent_state = self.state_linear(linear_input)

        x = th.cat((x, latent_state), dim=1)
        x = self.linear(x)
        return x
