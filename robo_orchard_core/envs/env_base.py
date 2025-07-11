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
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import torch
from typing_extensions import (
    Generic,
    Sequence,
    TypeVar,
)

from robo_orchard_core.policy import PolicyMixin, RandomPolicy
from robo_orchard_core.utils.config import (
    ClassConfig,
    ClassInitFromConfigMixin,
)

ObsType = TypeVar("ObsType")
RewardsType = TypeVar("RewardsType")


@dataclass
class EnvStepReturn(Generic[ObsType, RewardsType]):
    """The return type of the `step` function in the environment.

    This class extends gymnasium's Env step return type to be a dataclass
    with generic types for observations and rewards.


    Template Args:
        ObsType: The type of the observations returned by the environment.
        RewardsType: The type of the rewards returned by the environment.

    """

    observations: ObsType
    """The observations returned by the environment after taking a step."""
    rewards: RewardsType | None
    """The rewards returned by the environment after taking a step."""
    terminated: torch.Tensor | None
    """Whether the environment has reached a terminal state.

    Usually the environment is considered to be in a terminal state
    when the agent has reached the goal or failed to reach the goal.

    User should call `reset` function to reset the environment
    after the environment has reached a terminal state.
    """
    truncated: torch.Tensor | None
    """Whether the truncation condition has been met, for example, a time
    limit, or the agent has gone out of bounds.

    This field can be used to end the episode early without considering
    the `terminated` state. Different from `terminated`, `truncated` is
    usually a passive condition that triggers when the evironment has
    to stop without considering anything.
    """

    info: dict[str, Any] | None
    """Auxiliary information returned by the environment. """


EnvStepReturnType = TypeVar(
    "EnvStepReturnType", bound=EnvStepReturn, covariant=True
)
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


    This class is a simplified version of the gymnasium environment
    interface, which is used to define the environment for reinforcement
    learning tasks. We simplify the interface to provide a more generic
    and interface for environments.


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

    @property
    def unwrapped_env(self) -> Any:
        """Return the unwrapped environment.

        In some cases, the current environment is a wrapper around
        another environment. This property returns the unwrapped
        environment instance.

        Returns:
            EnvType: The unwrapped environment instance.
        """
        return self

    def rollout(
        self,
        max_steps: int,
        init_obs: Any,
        policy: PolicyMixin | None = None,
    ) -> list[EnvStepReturnType]:
        """Roll out the environment for a number of steps.

        This function is used to run the environment for a number of steps
        and return the results of each step.


        Args:
            max_steps (int): The maximum number of steps to roll out the
                environment.
            policy (PolicyMixin | None, optional): The policy to use for
                taking actions in the environment. If None, random actions
                will be taken. Defaults to None.

        Returns:
            list[EnvStepReturnType]: A list of results from each step of the
                environment. Each result is an instance of `EnvStepReturnType`
                containing observations, rewards, and other information.
        """
        if policy is None:
            policy = RandomPolicy(
                observation_space=self.observation_space,
                action_space=self.action_space,
            )

        results = []
        for i in range(max_steps):
            if i == 0:
                obs = init_obs
            action = policy(obs)
            # if stochastic policy, the action should be sampled
            # from the distribution.
            if not policy.is_deterministic:
                action = action()
            step_ret = self.step(action)
            results.append(step_ret)
            obs = step_ret.observations
        return results
