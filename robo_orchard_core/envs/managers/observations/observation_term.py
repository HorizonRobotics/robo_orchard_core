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

from __future__ import annotations
from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from typing import Any, Generic

import gymnasium as gym
from typing_extensions import TypeVar

from robo_orchard_core.envs.managers.manager_base import (
    EnvType_co,
)
from robo_orchard_core.envs.managers.manager_term_base import (
    ManagerTermBase,
    ManagerTermBaseCfg,
)
from robo_orchard_core.envs.managers.scene_entity_cfg import SceneEntityCfg

ObsTermType_co = TypeVar(
    "ObsTermType_co", bound="ObservationTermBase", covariant=True
)
ObsCfgType_co = TypeVar(
    "ObsCfgType_co", bound="ObservationTermCfg", covariant=True
)

EntityCfgType_co = TypeVar(
    "EntityCfgType_co",
    bound=SceneEntityCfg,
    covariant=True,
    default=SceneEntityCfg,
)

ReturnType = TypeVar("ReturnType", default=Any)


class ObservationTermBase(
    ManagerTermBase[EnvType_co, ObsCfgType_co],
    Generic[EnvType_co, ObsCfgType_co, ReturnType],
    metaclass=ABCMeta,
):
    """The base class for all observation terms.

    An observation term is a class that generates the observation
    from the environment.

    Template Args:
        EnvType_co: The type of the environment.
        ObsCfgType: The configuration of the observation term.
        ReturnType: The type of the return value of the observation term.

    Args:
        cfg (ObservationTermCfg): The configuration of the observation term.
        env (EnvBase): The environment.

    """

    def __init__(self, cfg: ObsCfgType_co, env: EnvType_co):
        super().__init__(cfg, env)  # type: ignore

    @abstractmethod
    def __call__(self) -> ReturnType:
        """The implementation of the observation term.

        All subclasses should implement this method to return the observation.

        """
        raise NotImplementedError

    @abstractmethod
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Resets the observation term.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.

        """
        raise NotImplementedError

    @property
    def observation_space(self) -> gym.Space[ReturnType]:
        """The observation space of the observation term.

        Returns:
            gym.Space[ReturnType]: The observation space of the observation
                term.

        """
        raise NotImplementedError(
            "The observation space is not defined for this observation term."
        )


class ObservationTermCfg(
    ManagerTermBaseCfg[ObsTermType_co],
    Generic[ObsTermType_co, EntityCfgType_co],
):
    """The base configuration for all observation terms.

    Any observation term configuration should inherit from this class.


    Template Args:
        ObsTermType_co: The type of the observation term.
        EntityCfgType_co: The type of the entity configuration.


    """

    asset_cfg: EntityCfgType_co
    """The scene entity configuration for the asset."""
