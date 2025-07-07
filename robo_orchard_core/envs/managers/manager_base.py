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


"""The base class and interface for managers and their configurations.

You can refer to :py:mod:`robo_orchard_core.envs.managers.manager_term_base`
for more information about the manager term.

"""

from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import Any, Generic, Sequence

from typing_extensions import TypeVar

from robo_orchard_core.envs.env_base import EnvType_co
from robo_orchard_core.utils.config import (
    ClassConfig,
    ClassInitFromConfigMixin,
)

#:
ManagerConfigType_co = TypeVar(
    "ManagerConfigType_co", bound="ManagerBaseCfg", covariant=True
)
#:
ManagerType_co = TypeVar(
    "ManagerType_co",
    bound="ManagerBase",
    covariant=True,
    default="ManagerBase",
)


class ManagerBase(
    ClassInitFromConfigMixin,
    Generic[EnvType_co, ManagerConfigType_co],
    metaclass=ABCMeta,
):
    """Base class for managers that manage multiple terms.

    Similar to `ManagerTermBase` in Isaac Lab but independent of simulation
    frameworks. Use `pydantic` style configuration as well.

    Args:
        cfg (ManagerBaseCfg): The configuration of the manager.
        env (EnvType_co): The environment that the manager is associated with.

    """

    def __init__(self, cfg: ManagerConfigType_co, env: EnvType_co):
        self.cfg = cfg
        self._env = env

    @abstractmethod
    def reset(self, env_ids: Sequence[int] | None = None) -> Any:
        """Resets the manager.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.
        """
        raise NotImplementedError


class ManagerBaseCfg(
    ClassConfig[ManagerType_co],
    Generic[ManagerType_co],
):
    """Configuration class for the manager."""

    def __call__(self, env: Any, **kwargs) -> ManagerType_co:
        """Creates an instance of the manager term.

        Args:
            env(Any): The environment.
            **kwargs: The keyword arguments to be passed to update the
                configuration.

        """
        return self.create_instance_by_cfg(env, **kwargs)
