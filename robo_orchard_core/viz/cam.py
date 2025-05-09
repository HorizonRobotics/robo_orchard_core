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
import math
from abc import ABCMeta, abstractmethod

import torch

from robo_orchard_core.datatypes.geometry import BatchPose6D
from robo_orchard_core.utils.math import math_utils
from robo_orchard_core.utils.math.coord_convention import CoordAxis

__all__ = ["CameraMixin", "CameraMoveMixin"]


class CameraMixin(metaclass=ABCMeta):
    """The camera mixin class.

    The camera mixin class provides the basic interface for camera
    visualization. It is used to render the camera view and get the
    camera pose in the world frame.

    """

    @abstractmethod
    def _render(self):
        """Render the camera view.

        Since the camera view may be rendered in the background, this
        method should return nothing.
        """
        raise NotImplementedError

    @abstractmethod
    def get_pose_view_world(self) -> BatchPose6D:
        """Get the camera pose in the world frame."""
        raise NotImplementedError

    @property
    @abstractmethod
    def captured_image(self) -> torch.Tensor:
        """Get the captured image."""
        raise NotImplementedError


class CameraMoveMixin(metaclass=ABCMeta):
    """The camera move mixin class.

    The camera move mixin class provides the basic interface for camera
    movement.

    Args:
        local_coord_axis (CoordAxis): The local coordinate axis of the camera.

    """

    def __init__(self, local_coord_axis: CoordAxis):
        self._local_coord_axis = local_coord_axis

    @abstractmethod
    def _apply_to_view_local(
        self, translation: torch.Tensor | None, quat: torch.Tensor | None
    ):
        """Apply the pose transform to the camera view in the local frame.

        Args:
            translation (torch.Tensor | None): The translation in the
                local frame.
            quat (torch.Tensor | None): The quaternion in the local
                frame.


        """
        raise NotImplementedError

    def move_forward(self, amount: float, move_scale: float):
        """Move the camera in the local forward direction.

        The distance of the move is proportional to the scale of the move:

        .. code-block:: python

            distance = math.exp(math.log(move_scale) + amount) - move_scale


        Args:
            amount (float): The amount to move forward.
            move_scale (float): The scale of the move.

        """
        new_scale = math.exp(math.log(move_scale) + amount)
        move_distance = new_scale - move_scale

        translation = move_distance * self._local_coord_axis.forward
        self._apply_to_view_local(translation=translation, quat=None)

    def move_translation(
        self, amount_down: float, amount_right: float, move_scale: float
    ):
        """Move the camera in the local up and right directions.

        The distance of the move is proportional to the scale of the move:

        .. code-block:: python

                distance = move_scale * amount

        Args:
            amount_down (float): The amount to move down.
            amount_right (float): The amount to move right.
            move_scale (float): The scale of the move.

        """
        translation = (
            amount_down * move_scale * self._local_coord_axis.down
            + amount_right * move_scale * self._local_coord_axis.right
        )
        self._apply_to_view_local(translation=translation, quat=None)

    def first_person_view_rot(self, pitch_down: float, yaw_right: float):
        """Rotate the camera in the local pitch and yaw directions.

        Args:
            pitch_down (float): The radians to pitch down.
            yaw_right (float): The radians to yaw right.

        """

        def to_axis_angle(
            unit_vector: torch.Tensor, angle: float
        ) -> tuple[str, float]:
            assert unit_vector.dim() == 1
            if unit_vector[0] != 0:
                if unit_vector[0] > 0:
                    return "X", angle
                else:
                    return "X", -angle
            elif unit_vector[1] != 0:
                if unit_vector[1] > 0:
                    return "Y", angle
                else:
                    return "Y", -angle
            else:
                if unit_vector[2] > 0:
                    return "Z", angle
                else:
                    return "Z", -angle

        axis_r, angle_r = to_axis_angle(self._local_coord_axis.forward, 0)
        axis_p, angle_p = to_axis_angle(
            self._local_coord_axis.left, pitch_down
        )
        axis_y, angle_y = to_axis_angle(self._local_coord_axis.down, yaw_right)

        # "ZYX" of intrinsic rotation, identical to "xyz" of extrinsic rotation
        axis = "".join([axis_y, axis_p, axis_r])
        angle = torch.asarray([angle_y, angle_p, angle_r], dtype=torch.float64)
        rot_mat = math_utils.euler_angles_to_matrix(angle, axis)
        quat = math_utils.matrix_to_quaternion(rot_mat)

        self._apply_to_view_local(translation=None, quat=quat)
