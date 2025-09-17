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
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any, Generic

import gymnasium as gym
from typing_extensions import Self, TypeVar

from robo_orchard_core.utils.config import ClassConfig, ClassType

if TYPE_CHECKING:
    from robo_orchard_core.policy.remote import (
        RayRemoteClassConfig,
        RemotePolicyConfig,
    )


OBSType = TypeVar("OBSType")
ACTType = TypeVar("ACTType")


__all__ = ["PolicyMixin", "PolicyConfig", "ClassType", "PolicyConfigType_co"]


class PolicyMixin(Generic[OBSType, ACTType], metaclass=ABCMeta):
    """Policy interface.

    A policy is a function or a model that maps observations from the
    environment to actions. It can be deterministic or stochastic, and it is
    typically used in reinforcement learning to decide what action to take
    based on the current state of the environment.

    For stochastic policies, the `act` method may return a callable that
    generates an action when called, allowing for sampling from a distribution
    over actions. For deterministic policies, it directly returns the action.


    Template Args:
        OBSType: The type of the observation space.
        ACTType: The type of the action space.

    """

    observation_space: gym.Space[OBSType] | None
    """The observation space of the policy.

    Can be None if not applicable.
    """

    action_space: gym.Space[ACTType] | None
    """The action space of the policy.

    Can be None if not applicable.
    """

    def __init__(
        self,
        cfg: PolicyConfig[PolicyMixin],
        observation_space: gym.Space[OBSType] | None = None,
        action_space: gym.Space[ACTType] | None = None,
    ):
        """Initialize the policy with a configuration and optional spaces."""
        self.observation_space = observation_space
        self.action_space = action_space
        self.cfg = cfg

    @abstractmethod
    def reset(self, *args, **kwargs) -> None:
        """Reset the policy to its initial state.

        This method is called at the beginning of an episode or when the
        environment is reset. It allows the policy to prepare for a new
        sequence of observations and actions.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def act(self, obs: OBSType) -> ACTType:
        """Return an action based on the given observation.

        Args:
            obs: The observation from the environment.

        Returns:
            An action or a callable that returns an action.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def __call__(self, obs: OBSType) -> ACTType:
        """A convenience method to call the policy."""
        return self.act(obs)


PolicyType = TypeVar("PolicyType", bound=PolicyMixin, covariant=True)


class PolicyConfig(ClassConfig[PolicyType]):
    """The configuration for a policy.

    This is a generic configuration class that can be used to create
    instances of policies. It specifies the type of the policy class to be
    created.


    Template Args:
        PolicyType: The type of the policy class to be created. It must be a
            subclass of :class:`PolicyMixin`. This allows for type checking and
            ensures that the created policy adheres to the expected interface.

    """

    class_type: ClassType[PolicyType]
    """The type of the policy class to be created."""

    def __call__(
        self,
        observation_space: gym.Space | None = None,
        action_space: gym.Space | None = None,
    ):
        """Create an instance of the policy with the given spaces.

        Args:
            observation_space (gym.Space | None, optional): The observation
                space of the policy. Defaults to None.
            action_space (gym.Space | None, optional): The action space of the
                policy. Defaults to None.

        """
        return self.class_type(
            cfg=self,
            observation_space=observation_space,
            action_space=action_space,
        )

    def as_remote(
        self,
        remote_class_config: RayRemoteClassConfig | None = None,
        ray_init_config: dict[str, Any] | None = None,
        check_init_timeout: int = 60,
    ) -> RemotePolicyConfig[Self]:
        """Convert this PolicyConfig to a RemotePolicyConfig.

        This method wraps the current policy configuration into a
        RemotePolicyConfig, which can be used to create a remote policy
        instance that runs in a separate process or machine using Ray.

        Args:
            remote_class_config (RayRemoteClassConfig | None, optional):
                Configuration for the remote class. If None, defaults to
                RayRemoteClassConfig(). Defaults to None.
            ray_init_config (dict[str, Any] | None, optional): Configuration
                for initializing Ray. If None, use default. Defaults to None.
            check_init_timeout (int, optional): Timeout in seconds for checking
                if the remote actor is initialized. Defaults to 60.

        Returns:
            RemotePolicyConfig: The converted remote policy configuration.

        """
        from robo_orchard_core.policy.remote import (
            RemotePolicyConfig,
            as_remote_policy,
        )

        if isinstance(self, RemotePolicyConfig):
            return as_remote_policy(
                self.instance_config,
                remote_class_config=remote_class_config,
                ray_init_config=ray_init_config,
                check_init_timeout=check_init_timeout,
            )
        else:
            return as_remote_policy(
                self,
                remote_class_config=remote_class_config,
                ray_init_config=ray_init_config,
                check_init_timeout=check_init_timeout,
            )


PolicyConfigType_co = TypeVar(
    "PolicyConfigType_co", bound=PolicyConfig, covariant=True
)
