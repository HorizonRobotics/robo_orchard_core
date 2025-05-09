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

"""The config and base class of IK Controller."""

from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import Any, Generic, TypeVar

import torch

from robo_orchard_core.envs.env_base import EnvType_co
from robo_orchard_core.utils.config import (
    ClassConfig,
    ClassInitFromConfigMixin,
)

IKControllerConfigType_co = TypeVar(
    "IKControllerConfigType_co",
    bound="IKControllerConfig",
    covariant=True,
)


class IKControllerBase(
    ClassInitFromConfigMixin,
    Generic[IKControllerConfigType_co, EnvType_co],
    metaclass=ABCMeta,
):
    """The Inverse Kinematics Controller abstraction.

    Implementations of subclasses should provide:

    - set_goal: Set the target end-effector goal for the controller.
    - calculate: Calculate the joint angles to reach the goal.

    Template Args:
        IKControllerConfigType_co: The type of the configuration class.
        EnvType_co: The type of the environment class.

    """

    def __init__(self, cfg: IKControllerConfigType_co, env: EnvType_co | None):
        self._cfg = cfg
        self._env = env
        self._num_envs = cfg.num_envs
        self._device = torch.device(cfg.device)

        self._target_pos = torch.zeros(
            (self._num_envs, 3), device=self._device
        )
        self._target_quat = torch.zeros(
            (self._num_envs, 4), device=self._device
        )

    def set_goal(
        self, target_pos: torch.Tensor, target_quat: torch.Tensor
    ) -> None:
        """Set the target body goal for the controller.

        Note:
            The target position and quaternion should be in the root frame, NOT
            the world frame!

        Args:
            target_pos (torch.Tensor): The target position of the body.
                It should be a tensor of shape (num_envs, 3).
            target_quat (torch.Tensor): The target quaternion of
                the body. It should be a tensor of shape (num_envs, 4).

        """
        self._target_pos[:] = target_pos
        self._target_quat[:] = target_quat

    @abstractmethod
    def calculate(
        self, cur_body_pos: torch.Tensor, cur_bocy_quat: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the joint angles to reach the goal.

        Args:
            cur_body_pos (torch.Tensor): The current position of the body.
            cur_body_quat (torch.Tensor): The current quaternion of the body.

        Returns:
            torch.Tensor: The joint angles to reach the goal.
        """
        raise NotImplementedError()


IKControllerBaseType_co = TypeVar(
    "IKControllerBaseType_co",
    bound=IKControllerBase,
    covariant=True,
)


class IKControllerConfig(ClassConfig[IKControllerBaseType_co]):
    """The base configuration for the IK Controller.

    Template Args:
        IKControllerBaseType_co: The type of the IK controller class.
    """

    num_envs: int
    """The number of environments."""

    device: str
    """The device to use for torch tensors."""

    def __call__(
        self, env: Any | None = None, **kwargs
    ) -> IKControllerBaseType_co:
        """Creates an instance of IK Controller.

        Args:
            env(EnvBase): The environment.
            **kwargs: The keyword arguments to be passed to update the
                configuration.

        """
        return self.create_instance_by_cfg(env, **kwargs)
