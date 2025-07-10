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

from typing import Any, Dict, List, Mapping, Sequence

import gymnasium as gym
import numpy as np
import torch
from typing_extensions import TypeAlias, TypeVar

from robo_orchard_core.envs.managers.manager_base import (
    EnvType_co,
    ManagerBase,
    ManagerBaseCfg,
)
from robo_orchard_core.envs.managers.manager_term_base import (
    ManagerTermGroupCfg,
)
from robo_orchard_core.envs.managers.observations.observation_term import (
    ObsCfgType_co,
    ObservationTermBase,
    ObservationTermCfg,
)
from robo_orchard_core.utils.config import ClassType_co


class ObservationGroupCfg(ManagerTermGroupCfg[ObsCfgType_co]):
    """The configuration for the observation group.

    Template Args:
        EnvType: The type of the environment.
        ObsTermType_co: The type of the observation term.
        ObsCfgType_co: The type of the observation term configuration.

    """

    concatenate_terms: bool = False
    """Whether to concatenate the observation terms.

    If True, the observation terms are concatenated along the last dimension.
    Otherwise, they are kept separate and returned as a dictionary.

    User should make sure that the observation terms have the same shape
    when concatenating them.

    Note that the observation terms must return a tensor when concatenating
    the terms.

    """


ObsManagerConfigType_co = TypeVar(
    "ObsManagerConfigType_co",
    bound="ObservationManagerCfg",
    covariant=True,
    default="ObservationManagerCfg",
)


ObsReturnType: TypeAlias = dict[str, Any | dict[str, Any]]


class ObservationManager(ManagerBase[EnvType_co, ObsManagerConfigType_co]):
    """Manager for the observation terms.

    This class manages the observation terms and groups them into groups. It
    also provides methods to get the observation signals from given
    environment.


    Template Args:
        EnvType_co: The type of the environment.
        ObsManagerConfigType_co: The type of the configuration for the manager.

    Args:
        cfg (ObservationManagerCfg): The configuration for the manager.
        env (EnvBase): The environment object.

    """

    obs_return_type: type[ObsReturnType] = ObsReturnType

    def __init__(self, cfg: ObsManagerConfigType_co, env: EnvType_co):
        super().__init__(cfg, env)

        self._group_term_names: dict[str, list[str]] = (
            self.cfg.get_term_names()
        )
        self._group_term_cfgs: dict[str, dict[str, ObservationTermCfg]] = (
            self.cfg.get_term_cfgs()
        )
        self._group_terms: dict[str, dict[str, ObservationTermBase]] = {
            key: cfg.create_terms(env) for key, cfg in self.cfg.groups.items()
        }
        self._concatenate_flags: dict[str, bool] = (
            self.cfg.get_concatenate_flags()
        )

    @property
    def active_terms(self) -> dict[str, list[str]]:
        """The names of the observation terms in the groups."""
        return self._group_term_names

    @property
    def group_term_cfgs(self) -> dict[str, dict[str, ObservationTermCfg]]:
        """The configuration of the observation terms in the groups."""
        return self._group_term_cfgs

    @property
    def group_obs_concatenate(self) -> dict[str, bool]:
        """The concatenate flags for the observation groups."""
        return self._concatenate_flags

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        for group in self._group_terms.values():
            for term in group.values():
                term.reset(env_ids)

    def get_observations(
        self, group_names: Sequence[str] | None = None
    ) -> ObsReturnType:
        """Returns the observations for the groups.

        Args:
            group_names (Sequence[str] | None): The names of the groups.
                Defaults to None, in which case all groups are considered.

        Returns:
            dict[str, Any | dict[str, Any]]: The observations
            for the groups.

        """
        if group_names is None:
            group_names = list(self._group_terms.keys())
        return {key: self._get_observation(key) for key in group_names}

    def _get_observation(self, group_name: str) -> Any | dict[str, Any]:
        """Returns the observation for one group."""
        group = self._group_terms[group_name]
        if self.cfg.groups[group_name].concatenate_terms:
            terms = [term() for term in group.values()]
            for term in terms:
                if not isinstance(term, torch.Tensor):
                    raise ValueError(
                        "The observation term must return a tensor"
                        "when concatenating the terms."
                        f"Got {term} instead."
                    )
            return torch.cat(terms, dim=-1)
        else:
            return {key: term() for key, term in group.items()}

    @property
    def observation_space(self) -> gym.spaces.Dict:
        """The observation space of the environment.

        This is used to determine the type of observations that can be
        received from the environment. The observation space is usually
        defined in the environment configuration.

        Returns:
            gym.Space: The observation space of the environment.
        """
        ret = {}
        for group_name, group in self._group_terms.items():
            if self.cfg.groups[group_name].concatenate_terms:
                # Concatenate the observation terms
                low = []
                high = []
                shape = None
                for term in group.values():
                    s = term.observation_space
                    assert isinstance(s, gym.spaces.Box), (
                        "Observation terms must return a Box space when "
                        "concatenating the terms."
                    )
                    low.append(s.low)
                    high.append(s.high)
                    if shape is None:
                        shape = list(s.shape)
                    else:
                        assert len(shape) == len(s.shape), (
                            "Observation terms must have the same shape when "
                            "concatenating the terms."
                        )
                        for i in range(len(shape) - 1):
                            assert shape[i] == s.shape[i], (
                                "Observation terms must have the same shape "
                                "when concatenating the terms."
                            )
                        shape[-1] += s.shape[-1]
                    ret[group_name] = gym.spaces.Box(
                        low=np.stack(low, axis=-1),
                        high=np.stack(high, axis=-1),
                        shape=shape,
                    )
            else:
                # Keep the observation terms separate
                ret[group_name] = gym.spaces.Dict(
                    {
                        key: term.observation_space
                        for key, term in group.items()
                    }
                )
        return gym.spaces.Dict(ret)


ObservationManagerType_co = TypeVar(
    "ObservationManagerType_co",
    bound=ObservationManager,
    covariant=True,
    default=ObservationManager,
)


class ObservationManagerCfg(ManagerBaseCfg[ObservationManager]):
    """The configuration for the observation manager.

    Template Args:
        EnvType: The type of the environment.

    Args:
        groups (Mapping[str, ObservationGroupCfg]): The configuration
            for the observation groups.

    """

    class_type: ClassType_co[ObservationManager] = ObservationManager

    groups: Mapping[str, ObservationGroupCfg]
    """The configuration for the observation groups."""

    def __post_init__(self):
        """Post initialization.

        This method performs the following checks:
            - Check if the observation term names are unique.

        """
        # check if the observation term names are unique
        term_names = []
        for group in self.groups.values():
            for term_name in group.terms.keys():
                if term_name in term_names:
                    raise ValueError(
                        f"Duplicate observation term name: {term_name}."
                        "Please make sure the observation term name does not"
                        "conflict with other terms across all groups."
                    )
                term_names.append(term_name)

    def get_term_names(
        self, group_names: Sequence[str] | None = None
    ) -> Dict[str, List[str]]:
        """Returns the names of the observation terms in the groups.

        Args:
            group_names (Sequence[str] | None): The names of the groups.
                Defaults to None, in which case all groups are considered.

        Returns:
            Dict[str, List[str]]: The names of the observation terms in the
                groups.

        """
        ret = {}
        if group_names is None:
            group_names = list(self.groups.keys())
        for key in group_names:
            ret[key] = list(self.groups[key].terms.keys())
        return ret

    def get_term_cfgs(
        self, group_names: Sequence[str] | None = None
    ) -> Dict[str, Dict[str, ObservationTermCfg]]:
        """Returns the configuration of the observation terms in the groups.

        Args:
            group_names (Sequence[str] | None): The names of the groups.
                Defaults to None, in which case all groups are considered.

        Returns:
            Dict[str, Dict[str, ObservationTermCfg]]: The configuration of the
                observation terms in the groups.
        """
        ret = {}
        if group_names is None:
            group_names = list(self.groups.keys())
        for key in group_names:
            ret[key] = self.groups[key].terms
        return ret

    def get_concatenate_flags(
        self, group_names: Sequence[str] | None = None
    ) -> Dict[str, bool]:
        """Returns the concatenate flags for the observation groups.

        Args:
            group_names (Sequence[str] | None): The names of the groups.
                Defaults to None, in which case all groups are considered.

        Returns:
            Dict[str, bool]: The concatenate flags for the observation groups.
        """
        ret = {}
        if group_names is None:
            group_names = list(self.groups.keys())
        for key in group_names:
            ret[key] = self.groups[key].concatenate_terms
        return ret
