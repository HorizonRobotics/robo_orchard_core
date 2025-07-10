# Project RoboOrchard
#
# Copyright (c) 2024 Horizon Robotics. All Rights Reserved.
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

"""Basic interface for environments."""

from __future__ import annotations
from abc import ABCMeta, abstractmethod

import gymnasium as gym
from typing_extensions import (
    Generic,
    Sequence,
    TypeVar,
)

from robo_orchard_core.utils.config import (
    ClassConfig,
    ClassInitFromConfigMixin,
)

EnvStepReturnType = TypeVar("EnvStepReturnType")
#:
EnvType = TypeVar("EnvType", bound="EnvBase")
#:
EnvType_co = TypeVar("EnvType_co", bound="EnvBase", covariant=True)


class EnvBaseCfg(ClassConfig[EnvType_co], Generic[EnvType_co]):
    """The configuration for the environment.

    Template Args:
        EnvType_co: The type of the environment class.

    """

    def __call__(self) -> EnvType_co:
        """Create an instance of the environment."""
        return self.create_instance_by_cfg()


EnvBaseCfgType_co = TypeVar(
    "EnvBaseCfgType_co", bound=EnvBaseCfg, covariant=True
)


class EnvBase(
    ClassInitFromConfigMixin,
    Generic[EnvStepReturnType],
    metaclass=ABCMeta,
):
    """Base class for all environments.

    The environment is a class that comprise of all components and
    functions required to interact with.

    Specifically, the environment provides a `step` function, which
    takes in an action and returns the information about observations,
    rewards, and other information.


    Template Args:
        EnvStepReturnType: The type of the return value of the `step` function.

    """

    @abstractmethod
    def step(self, *args, **kwargs) -> EnvStepReturnType:
        """Interface of takeing a step in the environment.

        Usually, this function takes in an action and returns the
        observations, rewards, and other information.

        User should implement this function in the subclass.

        """
        raise NotImplementedError

    @abstractmethod
    def reset(
        self,
        seed: int | None = None,
        env_ids: Sequence[int] | None = None,
        **kwargs,
    ) -> EnvStepReturnType:
        """Reset the environment."""

        raise NotImplementedError

    @abstractmethod
    def close(self):
        """Close the environment."""
        raise NotImplementedError

    @property
    @abstractmethod
    def num_envs(self) -> int:
        """The number of instances of the environment that are running.

        For example, if the environment is a single instance, then this
        should return 1. This is the case for most classical environments.

        In the case of reforcement learning, usually a vectorized environment
        is used to run multiple instances of the environment in parallel to
        speed up the training process. In this case, this function should
        return the number of instances of the environment that are running.

        The number of instances of the environment is used in other parts of
        the code to manage the environment.

        """
        raise NotImplementedError

    @property
    def action_space(self) -> gym.Space:
        """The action space of the environment.

        This is used to determine the type of actions that can be taken in
        the environment. The action space is usually defined in the
        environment configuration.

        Returns:
            gym.Space: The action space of the environment.
        """
        raise NotImplementedError(
            "The action space is not defined in the environment."
        )

    @property
    def observation_space(self) -> gym.Space:
        """The observation space of the environment.

        This is used to determine the type of observations that can be
        received from the environment. The observation space is usually
        defined in the environment configuration.

        Returns:
            gym.Space: The observation space of the environment.
        """
        raise NotImplementedError(
            "The observation space is not defined in the environment."
        )
