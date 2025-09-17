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
import concurrent
import concurrent.futures
from typing import Any, Generic

import gymnasium as gym
import ray

from robo_orchard_core.policy.base import (
    PolicyConfig,
    PolicyConfigType_co,
    PolicyMixin,
)
from robo_orchard_core.utils.config import (
    ClassType,
    ConfigInstanceOf,
)
from robo_orchard_core.utils.ray import (
    RayRemoteClassConfig,
    RayRemoteInstance,
    RayRemoteInstanceConfig,
)

__all__ = [
    "RemotePolicy",
    "RemotePolicyConfig",
]


class RemotePolicy(RayRemoteInstance[PolicyMixin], PolicyMixin):
    """The policy class which runs the policy in ray remote actor.

    Args:
        cfg (RayPolicyConfig): The configuration for the Ray policy.
        observation_space (gym.Space | None): The observation space of the
            policy. Default is None.
        action_space (gym.Space | None): The action space of the policy.
            Default is None.

    Raises:
        RayActorNotAliveError: If the Ray actor fails to be alive within the
            specified timeout.

    """

    cfg: RemotePolicyConfig

    def __init__(
        self,
        cfg: RemotePolicyConfig,
        observation_space: gym.Space | None = None,
        action_space: gym.Space | None = None,
    ):
        super().__init__(
            cfg=cfg,
            observation_space=observation_space,  # type: ignore
            action_space=action_space,  # type: ignore
        )

    @property
    def is_deterministic(self) -> bool:
        return ray.get(self._remote.is_deterministic.remote())

    def act(self, *args, **kwargs) -> Any:
        return self.async_act(*args, **kwargs).result()

    def async_act(self, *args, **kwargs) -> concurrent.futures.Future:
        return self._remote.act.remote(*args, **kwargs).future()

    def reset(self, *args, **kwargs) -> Any:
        return self.async_reset(*args, **kwargs).result()

    def async_reset(self, *args, **kwargs) -> concurrent.futures.Future:
        return self._remote.reset.remote(*args, **kwargs).future()


class RemotePolicyConfig(
    PolicyConfig[RemotePolicy],
    RayRemoteInstanceConfig[RemotePolicy, PolicyConfigType_co],
    Generic[PolicyConfigType_co],
):
    class_type: ClassType[RemotePolicy] = RemotePolicy

    instance_config: ConfigInstanceOf[PolicyConfigType_co]

    def as_remote(
        self,
        remote_class_config: RayRemoteClassConfig | None = None,
        ray_init_config: dict[str, Any] | None = None,
        check_init_timeout: int = 60,
    ) -> RemotePolicyConfig[PolicyConfigType_co]:
        return as_remote_policy(
            cfg=self.instance_config,
            remote_class_config=remote_class_config,
            ray_init_config=ray_init_config,
            check_init_timeout=check_init_timeout,
        )


def as_remote_policy(
    cfg: PolicyConfigType_co,
    remote_class_config: RayRemoteClassConfig | None = None,
    ray_init_config: dict[str, Any] | None = None,
    check_init_timeout: int = 60,
) -> RemotePolicyConfig[PolicyConfigType_co]:
    """Convert a PolicyConfig to a RemotePolicyConfig.

    Args:
        cfg (PolicyConfig): The policy configuration to convert.
        remote_class_config (RayRemoteClassConfig | None): The Ray remote
            class configuration. If None, a default configuration will be used.
            Default is None.
        ray_init_config (dict[str, Any] | None): The configuration for
            initializing Ray. If None, use default. Default is None.
        check_init_timeout (int): Timeout in seconds for checking if the remote
            actor is initialized. Default is 60.

    Returns:
        RemotePolicyConfig: The converted remote policy configuration.
    """
    if remote_class_config is None:
        remote_class_config = RayRemoteClassConfig()
    return RemotePolicyConfig(
        instance_config=cfg,
        remote_class_config=remote_class_config,
        ray_init_config=ray_init_config,
        check_init_timeout=check_init_timeout,
    )
