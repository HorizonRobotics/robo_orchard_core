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

from typing import Any, Type

import gymnasium as gym

from robo_orchard_core.policy.base import ACTType, PolicyConfig, PolicyMixin

__all__ = ["RandomPolicy", "RandomPolicyConfig"]


class RandomPolicy(PolicyMixin[Any, ACTType]):
    """A random policy that samples actions uniformly from the action space."""

    def __init__(
        self,
        cfg: PolicyConfig[PolicyMixin] | None = None,
        observation_space: gym.Space[Any] | None = None,
        action_space: gym.Space[ACTType] | None = None,
    ):
        if cfg is None:
            cfg = RandomPolicyConfig()
        super().__init__(
            cfg=cfg,
            observation_space=observation_space,
            action_space=action_space,
        )

    def act(self, obs: Any) -> ACTType:
        """Sample a random action from the action space."""

        if self.action_space is None:
            raise ValueError("Action space is not defined for the policy.")

        return self.action_space.sample()

    def reset(self) -> None:
        """Reset the policy. No specific action is needed for random policy."""
        pass

    @property
    def is_deterministic(self) -> bool:
        """Check if the policy is deterministic."""
        return True


class RandomPolicyConfig(PolicyConfig[RandomPolicy]):
    class_type: Type[RandomPolicy] = RandomPolicy
