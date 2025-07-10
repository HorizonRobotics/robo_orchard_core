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
from abc import abstractmethod
from typing import Any, Generic

import gymnasium as gym
import torch
from typing_extensions import TypeVar

from robo_orchard_core.envs.managers.manager_base import (
    EnvType_co,
)
from robo_orchard_core.envs.managers.manager_term_base import (
    ManagerTermBase,
    ManagerTermBaseCfg,
)
from robo_orchard_core.envs.managers.scene_entity_cfg import SceneEntityCfg

EntityCfgType_co = TypeVar(
    "EntityCfgType_co",
    bound=SceneEntityCfg,
    covariant=True,
    default=SceneEntityCfg,
)
ActionTermCfgType_co = TypeVar(
    "ActionTermCfgType_co",
    bound="ActionTermCfg",
    covariant=True,
    default="ActionTermCfg",
)


class ActionTermBase(ManagerTermBase[EnvType_co, ActionTermCfgType_co]):
    """The base class for all action terms.

    An action term is a class that generates the action from the environment.

    Args:
        cfg (ActionTermCfg): The configuration of the action term.
        env (EnvBase): The environment.

    """

    def __init__(self, cfg: ActionTermCfgType_co, env: EnvType_co):
        super().__init__(cfg, env)  # type: ignore
        self._prepare_asset()

        self._raw_actions: torch.Tensor = torch.tensor([])
        self._processed_actions: torch.Tensor = torch.tensor([])

    @property
    def action_space(self) -> gym.Space[Any]:
        """The action space of the action term.

        The space describes the valid input actions(raw_actions to be
        processed) that can be sent to the action term.

        Returns:
            gym.Space: The action space of the action term.
        """
        raise NotImplementedError(
            "The action space is not defined for the action term. "
            "Please override the `action_space` property in the action term."
        )

    @property
    def raw_actions(self) -> torch.Tensor:
        """The raw actions."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """The processed actions."""
        return self._processed_actions

    @abstractmethod
    def _prepare_asset(self) -> None:
        """Prepare the asset for the action term.

        This function is called at the beginning of creating the action term.

        Any action term that requires asset preparation should override this
        function to implement how to get the asset from the environment.
        """

    @abstractmethod
    def _process_actions_impl(self, raw_actions: torch.Tensor) -> torch.Tensor:
        """Process the raw input actions to generate the processed actions."""
        raise NotImplementedError

    @abstractmethod
    def apply(self):
        """Apply the processed actions to the asset.

        This function is called at every simulation step or when timer is
        triggered.
        """
        raise NotImplementedError

    def process(self, raw_actions: torch.Tensor):
        """Process the raw input actions to generate the processed actions.

        This function is called when actions are sent to the environment. The
        raw actions are stored in the `_raw_actions` attribute and the
        processed actions are stored in the `_processed_actions` attribute.

        User should implement the `_process_actions_impl` function to process
        the raw actions and return the processed actions.

        """
        if self._raw_actions.shape != raw_actions.shape:
            self._raw_actions = raw_actions.clone()
        else:
            self._raw_actions[:] = raw_actions

        processed_actions = self._process_actions_impl(raw_actions)

        if self._processed_actions.shape != processed_actions.shape:
            self._processed_actions = processed_actions.clone()
        else:
            self._processed_actions[:] = processed_actions


ActionTermBaseType_co = TypeVar(
    "ActionTermBaseType_co",
    bound=ActionTermBase,
    covariant=True,
    default=ActionTermBase,
)


class ActionTermCfg(
    ManagerTermBaseCfg[ActionTermBaseType_co],
    Generic[ActionTermBaseType_co, EntityCfgType_co],
):
    """The base configuration for all action terms.

    Template Args:
        ActionTermBaseType_co: The type of the term transform configuration.
        EntityCfgType_co: The type of the entity configuration.

    """

    asset_cfg: EntityCfgType_co
    """The scene entity configuration for the asset."""

    debug_vis: bool = False
    """Whether to visualize debug information. Defaults to False."""
