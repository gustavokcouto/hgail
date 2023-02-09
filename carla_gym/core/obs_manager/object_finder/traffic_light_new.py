#!/usr/bin/env python

# Copyright (c) 2022 authors.
# authors: Zhejun Zhang, Alexander Liniger, Dengxin Dai, Fisher Yu and Luc van Gool
#
# This work is licensed under the terms of the CC-BY-NC 4.0.
# For a copy, see <https://creativecommons.org/licenses/by-nc/4.0/deed>.

import numpy as np
import carla
from gym import spaces
from carla_gym.core.obs_manager.obs_manager import ObsManagerBase


class ObsManager(ObsManagerBase):
    # Template config
    # obs_configs = {
    #     "module": "object_finder.traffic_light_new",
    # }
    def __init__(self, obs_configs):
        self._parent_actor = None
        super(ObsManager, self).__init__()

    def _define_obs_space(self):
        self.obs_space = spaces.Dict({
            'at_red_light': spaces.Discrete(2),
            'trigger_location': spaces.Box(low=-5000, high=5000, shape=(3,), dtype=np.float32),
            'trigger_square': spaces.Box(low=-5000, high=5000, shape=(5, 3), dtype=np.float32)
        })

    def attach_ego_vehicle(self, parent_actor):
        self._parent_actor = parent_actor

    def get_observation(self):
        obs = {
            'at_red_light': int(self._parent_actor.vehicle.is_at_traffic_light()),
            'trigger_location': np.zeros((3,), dtype=np.float32),
            'trigger_square': np.zeros((5, 3), dtype=np.float32)
        }
        return obs

    def clean(self):
        self._parent_actor = None