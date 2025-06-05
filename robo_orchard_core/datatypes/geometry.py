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

"""The geometry dataclass for 3D transformations and poses."""

from __future__ import annotations

import torch
from pydantic import AliasChoices, Field
from typing_extensions import Self

from robo_orchard_core.datatypes.dataclass import DataClass
from robo_orchard_core.utils.config import TorchTensor
from robo_orchard_core.utils.math import (
    CoordConventionType,
    math_utils,
    quaternion_to_matrix,
)
from robo_orchard_core.utils.math.transform.transform3d import (
    Transform3D_M,
)
from robo_orchard_core.utils.torch_utils import Device

__all__ = [
    "Transform3D",
    "BatchTransform3D",
    "Pose",
    "Pose6D",
    "FrameTransform",
    "BatchPose6D",
    "BatchPose",
    "BatchFrameTransform",
]


class Transform3D(DataClass):
    """A 3D transformation of rotation and translation.

    It can be used to represent the transformation of an object in 3D
    space, or relative pose to another object.
    """

    xyz: tuple[float, float, float] | TorchTensor = Field(
        default=(0.0, 0.0, 0.0),
        validation_alias=AliasChoices("xyz", "trans", "pos"),
    )
    """3D ranslation vector or position.

    Defaults to (0.0, 0.0, 0.0)."""

    quat: tuple[float, float, float, float] | TorchTensor = Field(
        default=(1.0, 0.0, 0.0, 0.0),
        validation_alias=AliasChoices("quat", "rot", "orientation"),
    )

    """Quaternion rotation/orientation (w, x, y, z).

    Defaults to (1.0, 0.0, 0.0, 0.0)."""

    @property
    def trans(self):
        return self.xyz

    @property
    def rot(self):
        return self.quat

    def __post_init__(self):
        if isinstance(self.trans, torch.Tensor):
            assert self.trans.dim() == 1 and self.trans.shape[0] == 3, (
                "Translation must be a 1D tensor with shape (3)."
            )
        if isinstance(self.rot, torch.Tensor):
            assert self.rot.dim() == 1 and self.rot.shape[0] == 4, (
                "Rotation must be a 1D tensor with shape (4)."
            )

    def as_BatchTransform3D(self, device: Device = "cpu") -> BatchTransform3D:
        """Convert the Transform3D to a batch of transformations.

        Args:
            device (Device): The device to put the tensors on.

        Returns:
            BatchTransform3D: A BatchTransform3D object with the same
                translation and rotation as the Transform3D.
        """
        return BatchTransform3D(
            xyz=torch.tensor(self.xyz, device=device),
            quat=torch.tensor(self.quat, device=device),
        )


class BatchTransform3D(DataClass):
    """A batch of 3D transformations.

    This class is used to represent a batch of 3D transformations. It is
    useful when dealing with multiple objects or poses at once.
    """

    xyz: TorchTensor = Field(
        validation_alias=AliasChoices("xyz", "trans", "pos"),
    )

    """3D Translation or points. Shape is (N, 3) where N is the batch size."""

    quat: TorchTensor = Field(
        validation_alias=AliasChoices("quat", "rot", "orientation"),
    )
    """Quaternion rotation/orientation (w, x, y, z).

    Shape is (N, 4) where N is the batch size."""

    @classmethod
    def identity(
        cls,
        batch_size: int,
        device: Device = "cpu",
    ) -> Self:
        """Get a batch of identity transformations.

        Args:
            batch_size (int): The batch size.
            device (Device): The device to put the tensors on.

        Returns:
            BatchTransform3D: A batch of identity transformations.
        """
        return cls(
            xyz=torch.zeros(batch_size, 3, device=device),
            quat=torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]] * batch_size, device=device
            ),
        )

    @classmethod
    def from_view(
        cls,
        position: TorchTensor,
        look_at: TorchTensor,
        device: Device = "cpu",
        view_convention: CoordConventionType = "world",
    ) -> Self:
        """Create a batch of transformations from view.

        Args:
            position (TorchTensor): The position of the camera in local frame.
            look_at (TorchTensor): The target to look at in local frame.
            view_convention (CoordConventionType): The view convention to
                apply.

        Returns:
            BatchTransform3D: A batch of transformations.
        """
        rot_mat = math_utils.rotation_matrix_from_view(
            camera_position=position,
            at=look_at,
            device=device,
            view_convention=view_convention,
        )
        quat = math_utils.matrix_to_quaternion(rot_mat)
        return cls(xyz=position, quat=quat)

    @property
    def rot(self) -> torch.Tensor:
        return self.quat

    @property
    def trans(self) -> torch.Tensor:
        return self.xyz

    def __post_init__(self):
        # check batch size equal
        if self.xyz.dim() == 1:
            self.xyz = self.xyz[None]
        if self.quat.dim() == 1:
            self.quat = self.quat[None]

        self.check_shape()

    def batch_size(self) -> int:
        """Get the batch size.

        The batch size is the number of poses/transforms in the batch.

        Returns:
            int: The batch size.
        """
        return self.xyz.shape[0]

    def check_shape(self):
        """Check the shape of the translation and rotation tensors.

        Raises:
            ValueError: If the shapes of the translation and rotation tensors
                are not valid.
        """
        if self.trans.shape[0] != self.rot.shape[0]:
            raise ValueError("The number of xyz and quat must be the same.")
        if self.trans.shape[1] != 3:
            raise ValueError("xyz must have 3 components.")
        if self.rot.shape[1] != 4:
            raise ValueError("quat must have 4 components.")

    def as_Transform3D_M(self) -> Transform3D_M:
        """Convert the BatchTransform3D to matrix form.

        Returns:
            Transform3D_M: A batch of Transform3D_M objects.
        """
        return Transform3D_M.from_rot_trans(
            R=quaternion_to_matrix(self.quat), T=self.xyz
        )

    def transform_points(self, points: torch.Tensor) -> torch.Tensor:
        """Transform a batch of points by the batch of transformations.

        Args:
            points (torch.Tensor): A tensor of shape (N, P, 3) representing
                the batch of points to transform.

        Returns:
            torch.Tensor: A tensor of shape (N, P, 3) representing
                the transformed points.
        """
        if points.dim() == 2:
            points = points[None]  # # (P, 3) -> (1, P, 3)
        if points.dim() != 3:
            raise ValueError("The points tensor must have shape (N, P, 3).")

        N, P, _3 = points.shape
        if N != self.batch_size():
            raise ValueError(
                "The number of points must be the same as the batch size."
            )

        ret = math_utils.quaternion_apply_point(
            quaternion=self.quat, point=points, batch_mode=True
        )  # (N, P, 3)
        ret += self.xyz[:, None]  # (N, 1, 3) -> (N, P, 3)
        return ret

    def compose(self, *others: Self) -> Self:
        """Compose transformations with other transformations.

        The transformations are applied in the order they are passed.
        The following two lines are equivalent:

        .. code-block:: python

            t = t1.compose(t2, t3)
            t = t1.compose(t2).compose(t3)

        Args:
            other (Self): The other batch of transformations.

        Returns:
            Self: A new object with the
                composed transformations.
        """
        q = torch.clone(self.quat)
        t = torch.clone(self.xyz)
        for other in others:
            t, q = math_utils.frame_transform_combine(
                t12=t,
                q12=q,
                t01=other.xyz,
                q01=other.quat,
            )
        return type(self)(xyz=t, quat=q)

    def subtract(self, other: Self) -> Self:
        """Subtract transformations with another.

        .. code-block:: python

            t = t2.subtract(t1)
            t_ = t2.compose(t1.inverse())
            t == t_

            t2_ = t.compose(t1)
            t2 == t2_

        Args:
            other (Self): The other transformation.

        Returns:
            Self: The difference between the two transformations.
        """
        t, q = math_utils.frame_transform_subtract(
            t01=other.xyz,
            q01=other.quat,
            t02=self.xyz,
            q02=self.quat,
        )
        return type(self)(xyz=t, quat=q)

    def inverse(self) -> Self:
        """Get the inverse of the transformations.

        Returns:
            Self: A new object with the inverse transformations.
        """
        q_inv = math_utils.quaternion_invert(self.quat)
        return type(self)(
            xyz=math_utils.quaternion_apply_point(q_inv, -self.xyz), quat=q_inv
        )

    def translate(self, translation: TorchTensor) -> Self:
        """Apply translation to the transformations.

        Args:
            translation (TorchTensor): The translation to apply to.
                Shape should be (3,) or (N, 3) where N is the batch size.

        """
        t, q = math_utils.frame_transform_combine(
            t12=self.xyz,
            q12=self.quat,
            t01=translation,
            q01=torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.xyz.device),
        )
        return type(self)(xyz=t, quat=q)

    def rotate(self, axis_angle: TorchTensor) -> Self:
        """Rotate the transformations by an axis-angle rotation.

        Args:
            axis_angle (TorchTensor): The axis-angle rotation to apply to.
                Shape should be (3,) or (N, 3) where N is the batch size.

        """
        q_new = math_utils.axis_angle_to_quaternion(axis_angle)

        t, q = math_utils.frame_transform_combine(
            t12=self.xyz,
            q12=self.quat,
            t01=torch.tensor([0.0, 0.0, 0.0], device=self.xyz.device),
            q01=q_new,
        )
        return type(self)(xyz=t, quat=q)


class Pose(Transform3D):
    """A 6D pose data class.

    Different from Transform3D, Pose6D is composed of a 3D position and a
    3D orientation. The position and orientation share the same underlying
    data in Transform3D.

    It is more intuitive to use the position property when dealing with poses,
    as you can apply a tranlation to a point, but not to a vector.

    In addition, the Pose also has a frame_id attribute, which is used to
    specify the coordinate frame ID of reference for the pose.

    """

    frame_id: str | None = None
    """The coordinate frame ID of reference for the pose."""

    @property
    def pos(self) -> tuple[float, float, float] | torch.Tensor:
        return self.xyz

    @pos.setter
    def pos(self, value: tuple[float, float, float]):
        self.xyz = value

    @property
    def orientation(self) -> tuple[float, float, float, float] | torch.Tensor:
        return self.quat

    @orientation.setter
    def orientation(self, value: tuple[float, float, float, float]):
        self.quat = value

    def as_BatchPose6D(self, device: Device = "cpu") -> BatchPose6D:
        """Convert the Pose6D to BatchPose6D.

        Args:
            device (Device): The device to put the tensors on.

        Returns:
            BatchPose6D: A BatchPose6D object with the same
                position and orientation as the Pose6D.
        """
        return BatchPose6D(
            xyz=torch.tensor(self.xyz, device=device),
            quat=torch.tensor(self.quat, device=device),
        )


class BatchPose(BatchTransform3D):
    """A batch of 6D poses.

    This class is used to represent a batch of 6D poses. It is useful when
    dealing with multiple objects or poses at once.

    Different from BatchTransform3D, BatchPose6D is composed of a 3D position
    and a 3D orientation. Although the position and orientation share the
    same underlying data in Transform3D, it is more intuitive to use the
    position property when dealing with poses.

    In addition, the Pose also has a frame_id attribute, which is used to
    specify the coordinate frame ID of reference for the pose.

    """

    frame_id: str | None = None
    """The coordinate frame ID of reference for the pose."""

    @property
    def pos(self) -> TorchTensor:
        return self.xyz

    @pos.setter
    def pos(self, value: TorchTensor):
        self.xyz = value

    @property
    def orientation(self) -> TorchTensor:
        return self.quat

    @orientation.setter
    def orientation(self, value: TorchTensor):
        self.quat = value


class FrameTransform(Transform3D):
    """A transformation between two coordinate frames in 3D space.

    A transformation must specify the parent and child frames it connects.
    """

    parent_frame_id: str
    """The coordinate frame ID of the parent frame."""
    child_frame_id: str
    """The coordinate frame ID of the child frame."""


class BatchFrameTransform(BatchTransform3D):
    """A batch of transformations between two coordinate frames in 3D space.

    A transformation must specify the parent and child frames it connects,
    and all sample should share the same parent and child frames.
    """

    parent_frame_id: str
    """The coordinate frame ID of the parent frame."""
    child_frame_id: str
    """The coordinate frame ID of the child frame."""


# for deprecated naming compatibility
Pose6D = Pose
BatchPose6D = BatchPose
