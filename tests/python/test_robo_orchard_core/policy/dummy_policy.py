# Project RoboOrchard
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

from __future__ import annotations

import gymnasium as gym

from robo_orchard_core.policy.base import ClassType, PolicyConfig, PolicyMixin


class DummyPolicy(PolicyMixin):
    def __init__(
        self,
        cfg: "DummyPolicyConfig",
        observation_space: gym.Space | None = None,
        action_space: gym.Space | None = None,
    ):
        super().__init__(cfg, observation_space, action_space)

    def reset(self, *args, **kwargs):
        """Reset the policy. No specific action is needed for dummy policy."""
        print("reset called with args:", args, "kwargs:", kwargs)
        # return f"reset with args: {args}, kwargs: {kwargs}"
        return "reset", args, kwargs

    def act(self, *args, **kwargs):
        print("act called with args:", args, "kwargs:", kwargs)
        return "act", args, kwargs


class DummyPolicyConfig(PolicyConfig):
    class_type: ClassType[DummyPolicy] = DummyPolicy
