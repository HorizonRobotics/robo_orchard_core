# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
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
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar

import ray

from robo_orchard_core.envs.env_base import (
    EnvBase,
    EnvBaseCfg,
    EnvBaseCfgType_co,
    EnvStepReturn,
)
from robo_orchard_core.envs.rollout import EnvRolloutReturn
from robo_orchard_core.utils.config import ClassType, ConfigInstanceOf
from robo_orchard_core.utils.ray import (
    RayRemoteClassConfig,
    RayRemoteInstance,
    RayRemoteInstanceConfig,
)

if TYPE_CHECKING:
    from robo_orchard_core.policy.base import PolicyMixin
from robo_orchard_core.utils.logging import LoggerManager

logger = LoggerManager().get_child(__name__)

__all__ = ["RemoteEnv", "RemoteEnvCfg"]

T = TypeVar("T", bound=type)


class RemoteEnv(RayRemoteInstance[EnvBase], EnvBase):
    """The environment class which runs the env in ray remote actor.

    This environment is useful for running multiple environments in parallel
    in distributed setting.


    Args:
        cfg (RemoteEnvCfg): The configuration for the Ray environment.

    Raises:
        RayActorNotAliveError: If the Ray actor fails to be alive within the
            specified timeout.

    """

    cfg: RemoteEnvCfg

    def __init__(self, cfg: RemoteEnvCfg):
        super().__init__(cfg=cfg)

    @property
    def unwrapped_env(self):
        """For ray env, return the remote env instance."""
        return self._remote

    def async_step(
        self, *args, **kwargs
    ) -> concurrent.futures.Future[EnvStepReturn]:
        """Asynchronously step the environment."""
        return self._remote.step.remote(*args, **kwargs).future()

    def step(self, *args, **kwargs) -> EnvStepReturn:
        return ray.get(self._remote.step.remote(*args, **kwargs))

    def reset(self, *args, **kwargs) -> tuple[Any, dict]:
        return ray.get(self._remote.reset.remote(*args, **kwargs))

    def async_reset(
        self, *args, **kwargs
    ) -> concurrent.futures.Future[tuple[Any, dict]]:
        return self._remote.reset.remote(*args, **kwargs).future()

    def rollout(
        self,
        max_steps: int,
        init_obs: Any,
        policy: PolicyMixin | None = None,
        env_step_callback: Callable[[Any, Any], None] | None = None,
        terminal_condition: Callable[[Any], bool] | None = None,
        keep_last_results: int = -1,
    ) -> EnvRolloutReturn:
        return self.async_rollout(
            max_steps=max_steps,
            init_obs=init_obs,
            policy=policy,
            env_step_callback=env_step_callback,
            terminal_condition=terminal_condition,
            keep_last_results=keep_last_results,
        ).result()

    def async_rollout(
        self,
        max_steps: int,
        init_obs: Any,
        policy: PolicyMixin | None = None,
        env_step_callback: Callable[[Any, Any], None] | None = None,
        terminal_condition: Callable[[Any], bool] | None = None,
        keep_last_results: int = -1,
    ) -> concurrent.futures.Future[EnvRolloutReturn]:
        """Asynchronous version of `rollout`."""

        return self._remote.rollout.remote(
            max_steps=max_steps,
            init_obs=init_obs,
            policy=policy,
            env_step_callback=env_step_callback,
            terminal_condition=terminal_condition,
            keep_last_results=keep_last_results,
        ).future()

    def close(self):
        if hasattr(self, "_remote") and self._remote is not None:
            try:
                ray.get(self._remote.close.remote())
                ray.kill(self._remote)
            except Exception:
                pass
            del self._remote

    @property
    def num_envs(self) -> int:
        return ray.get(self._remote.num_envs.remote())

    @property
    def action_space(self) -> Any:
        return ray.get(self._remote.action_space.remote())

    @property
    def observation_space(self) -> Any:
        return ray.get(self._remote.observation_space.remote())


class RemoteEnvCfg(
    EnvBaseCfg[RemoteEnv],
    RayRemoteInstanceConfig[RemoteEnv, EnvBaseCfgType_co],
    Generic[EnvBaseCfgType_co],
):
    class_type: ClassType[RemoteEnv] = RemoteEnv
    instance_config: ConfigInstanceOf[EnvBaseCfgType_co]

    def as_remote(
        self,
        remote_class_config: RayRemoteClassConfig | None = None,
        ray_init_config: dict[str, Any] | None = None,
        check_init_timeout: int = 60,
    ) -> RemoteEnvCfg[EnvBaseCfgType_co]:
        return as_remote_env(
            cfg=self.instance_config,
            remote_class_config=remote_class_config,
            ray_init_config=ray_init_config,
            check_init_timeout=check_init_timeout,
        )


def as_remote_env(
    cfg: EnvBaseCfgType_co,
    remote_class_config: RayRemoteClassConfig | None = None,
    ray_init_config: dict[str, Any] | None = None,
    check_init_timeout: int = 60,
) -> RemoteEnvCfg[EnvBaseCfgType_co]:
    if remote_class_config is None:
        remote_class_config = RayRemoteClassConfig()
    return RemoteEnvCfg(
        instance_config=cfg,
        remote_class_config=remote_class_config,
        ray_init_config=ray_init_config,
        check_init_timeout=check_init_timeout,
    )
