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
#
# -----------------------------------------------------------------------------
# Portions of this file are derived from Isaac Lab (https://github.com/isaac-sim/IsaacLab).
# The original Isaac Lab code is licensed under the BSD-3-Clause license.
#
# Original Isaac Lab Copyright Notice:
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# You can find the original Isaac Lab source code at:
# https://github.com/isaac-sim/IsaacLab
#
# Modifications to the derived code, if any, are Copyright (c) 2025
# Horizon Robotics and are licensed under the Apache License, Version 2.0.
# The combined work in this file is distributed under the Apache License,
# Version 2.0, subject to the conditions of the BSD-3-Clause license for the
# portions derived from Isaac Lab.

"""Differential IK Controller."""

from __future__ import annotations
from abc import abstractmethod
from typing import Literal, TypeVar

import torch
from pydantic import Field

from robo_orchard_core.controllers.ik import (
    IKControllerBase,
    IKControllerConfig,
)
from robo_orchard_core.envs.env_base import EnvType_co
from robo_orchard_core.utils import math as math_utils
from robo_orchard_core.utils.config import (
    ClassConfig,
    ClassInitFromConfigMixin,
    ClassType_co,
)


class DifferentialIKSolver(ClassInitFromConfigMixin):
    r"""Differential Inverse Kinematics Solver.

    This solver is based on the concept of differential inverse
    kinematics [1, 2] which is a method for computing the change in joint
    positions that yields the desired change in pose. It uses the
    Isaac Lab implementation [3] as a reference.

    .. math::

        \Delta \mathbf{q} &= \mathbf{J}^{\dagger} \Delta \mathbf{x} \\
        \mathbf{q}_{\text{desired}} &= \mathbf{q}_{\text{current}} + \
            \Delta \mathbf{q}

    where :math:`\mathbf{J}^{\dagger}` is the pseudo-inverse of the
    Jacobian matrix :math:`\mathbf{J}`, :math:`\Delta \mathbf{x}` is the
    desired change in pose, and :math:`\mathbf{q}_{\text{current}}` is the
    current joint positions.

    Reference:

    1. `Robot Dynamics Lecture Notes <https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2017/RD_HS2017script.pdf>`_
       by Marco Hutter (ETH Zurich)
    2. `Introduction to Inverse Kinematics <https://www.cs.cmu.edu/~15464-s13/lectures/lecture6/iksurvey.pdf>`_
       by Samuel R. Buss (University of California, San Diego)
    3. Isaac Lab implementation:
       https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/controllers/differential_ik.py
    """

    def __init__(self, cfg: DifferentialIKSolverConfig):
        self.cfg = cfg

    def calculate_delta_joint_pos(
        self, delta_pose: torch.Tensor, jacobian: torch.Tensor
    ) -> torch.Tensor:
        r"""Computes delta joint position that yields change in target pose.

        The method uses the Jacobian mapping from joint-space velocities
        to end-effector velocities to compute the delta-change in the
        joint-space that moves the robot closer to a desired end-effector
        position.

        Args:
            delta_pose: The desired delta pose in shape (N, 3) or (N, 6).
                The delta pose is defined as the difference between the
                desired pose and the current pose, where the translation
                is :math:`\Delta \mathbf{t} = \mathbf{t}_{\text{desired}}
                - \mathbf{t}_{\text{current}}`, and the rotation is
                :math:`\Delta \mathbf{R} = ToAxisAngle(
                \mathbf{q}_{\text{desired}}
                * \mathbf{q}_{\text{current}}^{-1})`.

            jacobian: The geometric jacobian matrix in shape (N, 3, num_joints)
                or (N, 6, num_joints).

        Returns:
            The desired delta in joint space. Shape is (N, num-joints).

        """
        # check dimensions
        if delta_pose.shape[-1] not in [3, 6]:
            raise ValueError(
                f"Unsupported shape for delta pose: {delta_pose.shape}."
            )
        if jacobian.shape[-2] != delta_pose.shape[-1]:
            raise ValueError(
                f"Jacobian shape {jacobian.shape} does not match "
                f"delta pose shape {delta_pose.shape}."
            )

        # compute the delta in joint-space
        if self.cfg.ik_method == "pinv":  # Jacobian pseudo-inverse
            # parameters
            k_val = self.cfg.ik_params["k_val"]
            # computation
            jacobian_pinv = torch.linalg.pinv(jacobian)
            delta_joint_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
            delta_joint_pos = delta_joint_pos.squeeze(-1)
        elif self.cfg.ik_method == "svd":  # adaptive SVD
            # parameters
            k_val = self.cfg.ik_params["k_val"]
            min_singular_value = self.cfg.ik_params["min_singular_value"]
            # computation
            # U: 6xd, S: dxd, V: d x num-joint
            U, S, Vh = torch.linalg.svd(jacobian)
            S_inv = 1.0 / S
            S_inv = torch.where(
                S > min_singular_value, S_inv, torch.zeros_like(S_inv)
            )
            jacobian_pinv = (
                torch.transpose(Vh, dim0=1, dim1=2)[:, :, :6]
                @ torch.diag_embed(S_inv)
                @ torch.transpose(U, dim0=1, dim1=2)
            )
            delta_joint_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
            delta_joint_pos = delta_joint_pos.squeeze(-1)
        elif self.cfg.ik_method == "trans":  # Jacobian transpose
            # parameters
            k_val = self.cfg.ik_params["k_val"]
            # computation
            jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
            delta_joint_pos = k_val * jacobian_T @ delta_pose.unsqueeze(-1)
            delta_joint_pos = delta_joint_pos.squeeze(-1)
        elif self.cfg.ik_method == "dls":  # damped least squares
            # parameters
            lambda_val = self.cfg.ik_params["lambda_val"]
            # computation
            jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
            lambda_matrix = (lambda_val**2) * torch.eye(
                n=jacobian.shape[1], device=jacobian.device
            )
            delta_joint_pos = (
                jacobian_T
                @ torch.inverse(jacobian @ jacobian_T + lambda_matrix)
                @ delta_pose.unsqueeze(-1)
            )
            delta_joint_pos = delta_joint_pos.squeeze(-1)
        else:
            raise ValueError(
                f"Unsupported inverse-kinematics method: {self.cfg.ik_method}"
            )

        return delta_joint_pos


class DifferentialIKSolverConfig(ClassConfig[DifferentialIKSolver]):
    """Config for the DifferentialIKSolver."""

    class_type: ClassType_co[DifferentialIKSolver] = DifferentialIKSolver

    ik_method: Literal["pinv", "svd", "trans", "dls"]
    """Method for computing inverse of Jacobian."""

    ik_params: dict[str, float] = Field(default_factory=lambda: {})
    """Parameters for the inverse-kinematics method.

    Defaults to None, in which case the default parameters
    for the method are used.

    - Moore-Penrose pseudo-inverse ("pinv"):
        - "k_val": Scaling of computed delta-joint positions (default: 1.0).

    - Adaptive Singular Value Decomposition ("svd"):
        - "k_val": Scaling of computed delta-joint positions (default: 1.0).
        - "min_singular_value": Single values less than this are suppressed to
          zero (default: 1e-5).

    - Jacobian transpose ("trans"):
        - "k_val": Scaling of computed delta-joint positions (default: 1.0).

    - Damped Moore-Penrose pseudo-inverse ("dls"):
        - "lambda_val": Damping coefficient (default: 0.01).

    """

    def __post_init__(self):
        # default parameters for different inverse kinematics approaches.
        default_ik_params = {
            "pinv": {"k_val": 1.0},
            "svd": {"k_val": 1.0, "min_singular_value": 1e-5},
            "trans": {"k_val": 1.0},
            "dls": {"lambda_val": 0.01},
        }
        # update parameters for IK-method if not provided
        ik_params = default_ik_params[self.ik_method].copy()
        ik_params.update(self.ik_params)
        self.ik_params = ik_params


# :
DifferentialIKControllerCfgType_co = TypeVar(
    "DifferentialIKControllerCfgType_co",
    bound="DifferentialIKControllerCfg",
    covariant=True,
)


class DifferentialIKController(
    IKControllerBase[DifferentialIKControllerCfgType_co, EnvType_co]
):
    r"""The Differential Inverse Kinematics Controller.

    Implementations of subclasses should provide:

    - _get_joint_positions: Get the current joint positions.
    - _get_jacobian: Get the geometric jacobian matrix.

    The controller uses a differential IK solver to calculate the joint
    angles to reach the goal. The differential IK solver computes the
    velocity in joint space that yields the desired velocity in the
    end-effector/body space.

    User should set a target position and quaternion using the set_goal
    method before calling the calculate method.

    Note:
        This is a very naive IK controller implementation. The target
        position and quaternion should be close to the current position
        and quaternion to follows the local linearization assumption!

    """

    def __init__(
        self, cfg: DifferentialIKControllerCfgType_co, env: EnvType_co | None
    ):
        super().__init__(cfg, env)
        self._delta_pose = torch.zeros(
            (self._num_envs, 6), device=self._device
        )
        self._differeintial_ik_solver = self._cfg.diff_ik_solver_cfg()

    def calculate(
        self,
        body_pos: torch.Tensor,
        body_quat: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the joint angles to reach the goal.

        Note:
            The current body position and quaternion should be in the root
            frame, NOT the world frame!

        Args:
            body_pos (torch.Tensor): The current position of the body with
                respect to the root.
            body_quat (torch.Tensor): The current quaternion of the body with
                respect to the root.

        Returns:
            torch.Tensor: The joint angles.

        """
        jacobian = self._get_jacobian()
        joint_pos = self._get_joint_positions()
        position_error, quat_err = math_utils.pose_diff(
            self._target_pos, self._target_quat, body_pos, body_quat
        )
        self._delta_pose[..., 0:3] = position_error
        self._delta_pose[..., 3:6] = math_utils.quaternion_to_axis_angle(
            quat_err
        )

        # @TODO: Support offset for the end-effector.

        return (
            joint_pos
            + self._differeintial_ik_solver.calculate_delta_joint_pos(
                delta_pose=self._delta_pose,
                jacobian=jacobian,
            )
        )

    @abstractmethod
    def _get_joint_positions(self) -> torch.Tensor:
        """Get the current joint positions.

        Returns:
            torch.Tensor: The current joint positions.
        """
        raise NotImplementedError()

    @abstractmethod
    def _get_jacobian(self) -> torch.Tensor:
        """Get the geometric jacobian matrix in the root frame.

        Returns:
            torch.Tensor: The geometric Jacobian matrix.
        """
        raise NotImplementedError()


DifferentialIKControllerType_co = TypeVar(
    "DifferentialIKControllerType_co",
    bound=DifferentialIKController,
    covariant=True,
)


class DifferentialIKControllerCfg(
    IKControllerConfig[DifferentialIKControllerType_co]
):
    """The configuration for DifferentialIKController."""

    diff_ik_solver_cfg: DifferentialIKSolverConfig
    """Configuration for the differential IK solver."""
