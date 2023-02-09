#!/usr/bin/env python

# Copyright (c) 2022 authors.
# authors: Zhejun Zhang, Alexander Liniger, Dengxin Dai, Fisher Yu and Luc van Gool
#
# This work is licensed under the terms of the CC-BY-NC 4.0.
# For a copy, see <https://creativecommons.org/licenses/by-nc/4.0/deed>.

# base class for observation managers


class ObsManagerBase(object):

    def __init__(self):
        self._define_obs_space()

    def _define_obs_space(self):
        raise NotImplementedError

    def attach_ego_vehicle(self, parent_actor):
        raise NotImplementedError

    def get_observation(self):
        raise NotImplementedError

    def clean(self):
        raise NotImplementedError
