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
import copy
from typing import Any, Sequence

import torch

from robo_orchard_core.datatypes.dataclass import (
    DataClass,
    TensorToMixin,
)
from robo_orchard_core.datatypes.timestamps import concat_timestamps
from robo_orchard_core.utils.config import TorchTensor


class BatchJointsState(DataClass, TensorToMixin):
    position: TorchTensor | None = None
    """The position of the joint in radians or meters.

    It should be a tensor of shape (B, D), where D is the number of joints and
    B is the batch size.
    """

    velocity: TorchTensor | None = None
    """The velocity of the joint in radians or meters per second.

    It should be a tensor of shape (B, D), where D is the number of joints and
    B is the batch size.
    """

    effort: TorchTensor | None = None
    """The effort/torque applied to the joint in Newton-meters or Newtons.

    It should be a tensor of shape (B, D), where D is the number of joints and
    B is the batch size.
    """

    names: list[str] | None = None
    """The names of the joints. The names are identical to the
    joint names in the robot description file (e.g., URDF or SDF).
    In this case, the name is also referred as frame_id.

    We recommand using the frame_id of the joint in the robot description
    file as the name. If the name and frame_id are not the same,
    use the name field from JointState message in ROS.

    The size of the names list should match the number of joints D.
    """

    timestamps: list[int] | None = None
    """Timestamps of the camera data in nanoseconds(1e-9 seconds)."""

    def __post_init__(self):
        self.check_shape()
        if (
            self.timestamps is not None
            and len(self.timestamps) != self.batch_size
        ):
            raise ValueError(
                "The length of timestamps must match the batch size. "
                f"Expected {self.batch_size}, got {len(self.timestamps)}."
            )

    def __find_any_tensor(self) -> TorchTensor | None:
        if self.position is not None:
            return self.position
        elif self.velocity is not None:
            return self.velocity
        elif self.effort is not None:
            return self.effort
        return None

    @property
    def joint_num(self) -> int:
        """Get the number of joints in the joint state."""

        cur_tensor = self.__find_any_tensor()
        if cur_tensor is None:
            raise ValueError(
                "At least one of position, velocity, or effort must be set."
            )
        return cur_tensor.shape[1]

    @property
    def batch_size(self) -> int:
        """Get the batch size of the joint state."""

        cur_tensor = self.__find_any_tensor()
        if cur_tensor is None:
            raise ValueError(
                "At least one of position, velocity, or effort must be set."
            )
        return cur_tensor.shape[0]

    def check_shape(self):
        def find_tensor() -> tuple[torch.Tensor | None, str]:
            if self.position is not None:
                return self.position, "position"
            elif self.velocity is not None:
                return self.velocity, "velocity"
            elif self.effort is not None:
                return self.effort, "effort"
            return None, ""

        def check_tensor_shape_equal(
            tensor: torch.Tensor,
            other: torch.Tensor | None,
            src_name: str,
            other_name: str | None = None,
        ):
            if other is not None and tensor.shape != other.shape:
                raise ValueError(
                    f"The shape of {src_name} and {other_name} must match. "
                    f"Got {tensor.shape} and {other.shape}."
                )

        cur_tensor, cur_tensor_name = find_tensor()
        if cur_tensor is None:
            raise ValueError(
                "At least one of position, velocity, or effort must be set."
            )

        # check if shape matches
        if cur_tensor.dim() != 2:
            raise ValueError(
                "Data must be a 2D tensor with shape (N, D) where N "
                "is the batch size and D is the number of joints."
            )

        tensor_names = ["position", "velocity", "effort"]
        for name in tensor_names:
            check_tensor_shape_equal(
                cur_tensor, getattr(self, name), cur_tensor_name, name
            )

        # check if names match
        if self.names is not None:
            if len(self.names) != cur_tensor.shape[-1]:
                raise ValueError(
                    "The length of names must match the length of position."
                )

    def concat(
        self, others: Sequence[BatchJointsState], dim: int
    ) -> BatchJointsState:
        """Concatenate two BatchJointsState objects along a given dimension.

        Args:
            other (BatchJointsState): The other BatchJointsState
                to concatenate.
            dim (int): The dimension along which to concatenate. This can be
                0 (batch dimension), -1 (joint dimension), or 1 (joint
                dimension).

        Returns:
            BatchJointsState: A new BatchJointsState with concatenated data.
        """

        assert dim in [0, -1, 1], "dim must be 0, -1, or 1."
        if len(others) == 0:
            raise ValueError(
                "At least one other BatchJointsState must be provided."
            )
        for other in others:
            if not isinstance(other, BatchJointsState):
                raise TypeError(
                    "other must be an instance of BatchJointsState"
                )

        attr_dict: dict[str, Any] = {
            "position": None,
            "velocity": None,
            "effort": None,
        }
        for attr in attr_dict:
            # if self has no value for this attribute, skip it
            if (self_v := getattr(self, attr)) is None:
                attr_dict[attr] = None
                continue
            # otherwise, check that all others have the same attribute
            to_cat_list = [self_v]
            for other in others:
                other_v = getattr(other, attr)
                if other_v is None:
                    raise ValueError(
                        f"Both BatchJointsState objects must have {attr} set."
                    )
                to_cat_list.append(other_v)
            attr_dict[attr] = torch.cat(to_cat_list, dim=dim)

        ret_names = None
        if self.names is not None:
            for other in others:
                if other.names is None:
                    raise ValueError(
                        "Both BatchJointsState objects must have names set."
                    )
            if dim in [-1, 1]:
                # if concatenating along the joint dimension,
                # ensure names match across all objects
                for i, other in enumerate(others):
                    if not others[0].names == other.names:
                        raise ValueError(
                            f"Names must match across all BatchJointsState objects. "  # noqa: E501
                            f"Mismatch at index {i}: {others[0].names} != {other.names}"  # noqa: E501
                        )
                # Concatenate names along the last dimension
                ret_names = self.names + others[0].names  # type: ignore
            elif dim == 0:
                # Concatenate names along the batch dimension
                for i, other in enumerate(others):
                    if self.names != other.names:
                        raise ValueError(
                            f"Names must match across all BatchJointsState objects. "  # noqa: E501
                            f"Mismatch at index {i}: {self.names} != {other.names}"  # noqa: E501
                        )
                ret_names = copy.copy(self.names)
            else:
                raise ValueError("dim must be 0, -1, or 1.")
        attr_dict["names"] = ret_names

        # process timestamps
        timestamps = [self.timestamps] + [other.timestamps for other in others]
        timestamp_condat_dim = "row" if dim == 0 else "col"
        attr_dict["timestamps"] = concat_timestamps(
            timestamps, concat_dim=timestamp_condat_dim
        )

        return BatchJointsState(**attr_dict)
