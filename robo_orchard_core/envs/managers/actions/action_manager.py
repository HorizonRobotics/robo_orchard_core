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

from typing import Any, Dict, Generic, List, Mapping, Sequence

import gymnasium as gym
import torch
from typing_extensions import TypeVar

from robo_orchard_core.envs.managers.actions.action_term import (
    ActionTermBase,
    ActionTermCfg,
)
from robo_orchard_core.envs.managers.manager_base import (
    EnvType_co,
    ManagerBase,
    ManagerBaseCfg,
)
from robo_orchard_core.utils.config import ClassType_co

ActionTermCfgType_co = TypeVar(
    "ActionTermCfgType_co",
    bound=ActionTermCfg,
    covariant=True,
    default=ActionTermCfg,
)


ActManagerConfigType_co = TypeVar(
    "ActManagerConfigType_co",
    bound="ActionManagerCfg",
    covariant=True,
    default="ActionManagerCfg",
)


class ActionManager(ManagerBase[EnvType_co, ActManagerConfigType_co]):
    """The manager for the action terms.

    Template Args:
        EnvType_co: The environment type.
        ActManagerConfigType_co: The configuration type.

    Args:
        env (EnvBase): The environment.
        cfg (ActionManagerCfg): The configuration for the action manager.

    """

    def __init__(
        self,
        cfg: ActManagerConfigType_co,
        env: EnvType_co,
    ):
        super().__init__(cfg, env)

        self._term_cfgs: Mapping[str, ActionTermCfg] = self.cfg.terms
        self._term_names: List[str] = list(self._term_cfgs.keys())
        self._terms: Dict[str, ActionTermBase] = self.cfg.create_terms(env)

        self._prev_actions: Dict[str, torch.Tensor] = {
            term_name: torch.tensor([]) for term_name in self._term_names
        }
        self._actions: Dict[str, torch.Tensor] = {
            term_name: torch.tensor([]) for term_name in self._term_names
        }

    @property
    def active_terms(self) -> List[str]:
        """Returns the active terms."""
        return self._term_names

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        for term in self._terms.values():
            term.reset(env_ids=env_ids)

    @property
    def action(self) -> Dict[str, torch.Tensor]:
        """Returns the actions."""
        return self._actions

    @property
    def prev_action(self) -> Dict[str, torch.Tensor]:
        """Returns the previous actions."""
        return self._prev_actions

    @property
    def action_space(self) -> gym.spaces.Dict:
        """Returns the action space of the action manager.

        The action space is a dictionary where the keys are the term names
        and the values are the action spaces of the terms.

        Returns:
            gym.spaces.Dict: The action space of the action manager.

        """
        return gym.spaces.Dict(
            {
                term_name: term.action_space
                for term_name, term in self._terms.items()
            }
        )

    def process(self, actions: Dict[str, torch.Tensor]) -> None:
        """Processes the actions.

        Actions are (str, torch.Tensor) pairs where the key is the term
        name and the value is the action tensor for that term.

        This function will update the action cache for each term.

        Args:
            actions (Dict[str, torch.Tensor]): The actions to process.

        """
        self._copy_actions(src=self._actions, dst=self._prev_actions)
        self._copy_actions(src=actions, dst=self._actions)

        for term_name, action in actions.items():
            term = self._terms[term_name]
            term.process(action)

    def apply(self) -> None:
        """Applies the action terms to the environment."""
        for term in self._terms.values():
            term.apply()

    def _copy_actions(
        self, src: Dict[str, torch.Tensor], dst: Dict[str, torch.Tensor]
    ) -> None:
        # copy the actions
        for term_name, src_action in src.items():
            dst_action = dst[term_name]
            src_action = src[term_name]
            if src_action.shape != dst_action.shape:
                dst[term_name] = src_action.clone()
            else:
                dst_action[:] = src_action


class ActionManagerCfg(
    ManagerBaseCfg[ActionManager],
    Generic[ActionTermCfgType_co],
):
    """The configuration for the action manager.

    Template Args:
        ActionTermCfgType_co: The configuration type for the action terms.
    """

    class_type: ClassType_co[ActionManager] = ActionManager

    terms: Mapping[str, ActionTermCfgType_co]
    """The configuration for the action terms."""

    def create_terms(self, env: Any) -> Dict[str, ActionTermBase]:
        """Creates the action terms.

        Args:
            env (EnvType): The environment.

        Returns:
            Dict[str, ActionTermBase]: The action terms.

        """
        return {
            term_name: term_cfg(env=env)
            for term_name, term_cfg in self.terms.items()
        }
