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

"""Base class for all manager terms."""

from __future__ import annotations
from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from typing import Any, Generic, Mapping

from typing_extensions import TypeVar

from robo_orchard_core.envs.env_base import EnvType_co
from robo_orchard_core.utils.config import (
    ClassConfig,
    ClassInitFromConfigMixin,
    Config,
)

#:
TermConfigType_co = TypeVar(
    "TermConfigType_co", bound="ManagerTermBaseCfg", covariant=True
)
#:
ManagerTermType_co = TypeVar(
    "ManagerTermType_co", bound="ManagerTermBase", covariant=True
)


class ManagerTermBase(
    ClassInitFromConfigMixin,
    Generic[EnvType_co, TermConfigType_co],
    metaclass=ABCMeta,
):
    """Base class for all manager terms.

    Similar to `ManagerTermBase` in Isaac Lab, this class is the base class for
    all manager terms. A Manager term is the basic block to interact with the
    environment. Each manager term is responsible for managing a specific
    aspect of the environment, such as observations, rewards, etc.

    Users can define their own manager terms by inheriting from this class
    and implementing the missing methods.

    Different from the Isaac Lab, the manager term is independent of simulation
    engines. It use `pydantic` style configuration to define the configuration
    of the manager term to make everything structured and easy for
    serialization/deserialization and validation with json format.

    Args:
        cfg (ManagerTermBaseCfg): The configuration of the manager term.
        env (EnvBase): The environment.

    """

    cfg: TermConfigType_co
    _env: EnvType_co

    def __init__(self, cfg: TermConfigType_co, env: EnvType_co):
        self.cfg = cfg
        self._env = env

    @abstractmethod
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Resets the manager term.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.
        """
        raise NotImplementedError


class ManagerTermBaseCfg(
    ClassConfig[ManagerTermType_co],
):
    """Configuration class for the manager term.

    Template Args:
        ManagerTermType_co: The type of the manager term.
    """

    def __call__(self, env: Any, **kwargs) -> ManagerTermType_co:
        """Creates an instance of the manager term.

        Args:
            env(EnvType): The environment.
            **kwargs: The keyword arguments to be passed to update the
                configuration.

        """
        return self.create_instance_by_cfg(env, **kwargs)


class ManagerTermGroupCfg(Config, Generic[TermConfigType_co]):
    """The configuration for the observation group.

    Template Args:
        TermConfigType_co: The type of the term configuration.

    Args:
        terms (dict[str, ConfigType_co]): The configuration for the
            observation terms.

    """

    terms: Mapping[str, TermConfigType_co]
    """The configuration for the observation terms."""

    def create_terms(self, env: Any):
        """Creates the observation terms."""
        return {key: cfg(env=env) for key, cfg in self.terms.items()}


TransformCfgType_co = TypeVar(
    "TransformCfgType_co", bound="TermTransformCfg", covariant=True
)


class TermTransformBase(
    ClassInitFromConfigMixin, Generic[TransformCfgType_co], metaclass=ABCMeta
):
    """Base class for defining term transforms.

    Term transforms are used to modify the results from the manager terms.
    All transforms should inherit from this class and implement missing
    methods.

    In torch, there is a similar concept in Dataset and DataLoader classes.
    Each transform is a callable that takes the data and returns the modified
    data. The transforms can be chained together.

    Transform of terms are widely used to preprocess the results before feeding
    them to next stage.  For example, the results of an observation term can be
    normalized using a transform.

    """

    def __init__(self, cfg: TransformCfgType_co):
        self.cfg = cfg

    @abstractmethod
    def reset(self, env_ids: Sequence[int] | None = None):
        """Resets the Modifier.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.
        """
        raise NotImplementedError

    @abstractmethod
    def __call__(self, data: Any) -> Any:
        """Abstract method for defining the transform function.

        Args:
            data: The data to be transformed.

        Returns:
            Modified data. Shape is the same as the input data.
        """
        raise NotImplementedError


TermTransformBaseType_co = TypeVar(
    "TermTransformBaseType_co",
    bound="TermTransformBase",
    default="TermTransformBase",
)


class TermTransformCfg(
    ClassConfig[TermTransformBaseType_co],
):
    """Configuration class for the term transform."""

    def __call__(self, **kwargs) -> TermTransformBaseType_co:
        """Creates an instance of the term transform."""
        return self.create_instance_by_cfg(**kwargs)
