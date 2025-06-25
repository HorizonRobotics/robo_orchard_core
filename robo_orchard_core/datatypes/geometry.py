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
from typing import Sequence

import deprecated
import torch
from pydantic import AliasChoices, Field
from typing_extensions import Self

from robo_orchard_core.datatypes.dataclass import DataClass, TensorToMixin
from robo_orchard_core.datatypes.timestamps import concat_timestamps
from robo_orchard_core.utils.config import TorchTensor
from robo_orchard_core.utils.math import (
    CoordConventionType,
    math_utils,
    quaternion_to_matrix,
)
from robo_orchard_core.utils.math.transform.transform3d import (
    Transform3D_M,
)
from robo_orchard_core.utils.torch_utils import Device, make_device

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


@deprecated.deprecated(
    reason="Transform3D will be replaced by BatchTransform3D for "
    "simplicity and efficiency. ",
    version="0.2.0",
)
class Transform3D(DataClass, TensorToMixin):
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

        # The dimensions of xyz and quat will be expanded in
        # the initialization of BatchTransform3D.
        return BatchTransform3D(
            xyz=torch.tensor(self.xyz, device=device),
            quat=torch.tensor(self.quat, device=device),
        )

    def repeat(
        self, batch_size: int, device: Device = "cpu"
    ) -> BatchTransform3D:
        """Repeat the transformation to create a batch of transformations.

        Args:
            batch_size (int): The number of times to repeat the transformation.
            device (Device, optional): The device to put the tensors on.
                Defaults to "cpu".

        Returns:
            BatchFrameTransform: A batch of transformations with the same
                parent and child frames.
        """
        target_device = make_device(device)
        xyz = self.xyz
        if not isinstance(xyz, torch.Tensor):
            xyz = torch.tensor(xyz, device=device)
        elif xyz.device != target_device:
            xyz = xyz.to(device)

        quat = self.quat
        if not isinstance(quat, torch.Tensor):
            quat = torch.tensor(quat, device=device)
        elif quat.device != target_device:
            quat = quat.to(device)

        return BatchTransform3D(
            xyz=xyz.repeat(batch_size, 1),
            quat=quat.repeat(batch_size, 1),
        )


class BatchTransform3D(DataClass, TensorToMixin):
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

    timestamps: list[int] | None = None
    """Timestamps of the camera data in nanoseconds(1e-9 seconds)."""

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

        if (
            self.timestamps is not None
            and len(self.timestamps) != self.batch_size
        ):
            raise ValueError(
                "The length of timestamps must match the batch size. "
                f"Expected {self.batch_size}, got {len(self.timestamps)}."
            )

    @property
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
        if N != self.batch_size:
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

    def repeat(
        self, batch_size: int, timestamps: list[int] | None = None
    ) -> BatchTransform3D:
        """Repeat the transformation with batch size 1 to batch size N.

        Args:
            batch_size (int): The number of times to repeat the transformation.

        Returns:
            BatchFrameTransform: A batch of transformations.
        """
        if self.batch_size != 1:
            raise ValueError(
                "The batch size of the transformation must be 1 to repeat."
            )

        xyz = self.xyz
        quat = self.quat
        if timestamps is not None and len(timestamps) != batch_size:
            raise ValueError(
                f"The number of timestamps {len(timestamps)} must match "
                f"the batch size {batch_size}."
            )

        return BatchTransform3D(
            xyz=xyz.repeat(batch_size, 1),
            quat=quat.repeat(batch_size, 1),
            timestamps=timestamps,
        )

    def concat(self, others: Sequence[BatchTransform3D]) -> BatchTransform3D:
        """Concatenate two BatchTransform3D objects along batch dimension.

        Args:
            other (BatchTransform3D): The other BatchTransform3D
                to concatenate.

        Returns:
            BatchTransform3D: A new BatchTransform3D with concatenated data.
        """
        return BatchTransform3D(
            xyz=torch.cat([self.xyz] + [other.xyz for other in others], dim=0),
            quat=torch.cat(
                [self.quat] + [other.quat for other in others], dim=0
            ),
            timestamps=concat_timestamps(
                [self.timestamps] + [other.timestamps for other in others]
            ),
        )


@deprecated.deprecated(
    reason="Pose will be replaced by BatchPose for simplicity and efficiency.",
    version="0.2.0",
)
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
            device (Device, optional): The device to put the tensors on.
                Defaults to "cpu".

        Returns:
            BatchPose6D: A BatchPose6D object with the same
                position and orientation as the Pose6D.
        """
        return BatchPose6D(
            xyz=torch.tensor(self.xyz, device=device),
            quat=torch.tensor(self.quat, device=device),
            frame_id=self.frame_id,
        )

    def repeat(self, batch_size: int, device: Device = "cpu") -> BatchPose6D:
        """Repeat the pose to create a batch of poses.

        Args:
            batch_size (int): The number of times to repeat the pose.
            device (Device, optional): The device to put the tensors on.
                Defaults to "cpu".

        Returns:
            BatchPose6D: A batch of poses with the same position
                and orientation.
        """
        parent_impl = super().repeat(batch_size, device=device)
        return BatchPose6D(
            xyz=parent_impl.xyz,
            quat=parent_impl.quat,
            frame_id=self.frame_id,
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

    def as_BatchFrameTransform(
        self, child_frame_id: str, parent_frame_id: str | None = None
    ) -> BatchFrameTransform:
        """Convert the BatchPose to BatchFrameTransform.

        Args:
            child_frame_id (str): The coordinate frame ID of the child frame.
            parent_frame_id (str | None, optional): The coordinate frame ID of
                the parent frame. If None, it will use the frame_id of the
                BatchPose. Defaults to None.

        Returns:
            BatchFrameTransform: A BatchFrameTransform object with the same
                position and orientation as the BatchPose.
        """

        if parent_frame_id is None and self.frame_id is None:
            raise ValueError(
                "Either parent_frame_id or self.frame_id must be specified."
            )
        if parent_frame_id is None:
            parent_frame_id = self.frame_id
        assert parent_frame_id is not None

        return BatchFrameTransform(
            xyz=self.xyz,
            quat=self.quat,
            parent_frame_id=parent_frame_id,
            child_frame_id=child_frame_id,
        )

    def inverse(self, frame_id: str | None = None) -> Self:
        """Get the inverse of the transformations.

        Args:
            frame_id (str | None, optional): The coordinate frame ID of the
                inverse pose. This argument is required if the current
                BatchPose has a frame_id. Defaults to None.

        Returns:
            Self: A new object with the inverse transformations.
        """
        if self.frame_id is not None and frame_id is None:
            raise ValueError(
                "If current BatchPose has a frame_id, "
                "you must specify the frame_id for the inverse."
            )

        p = super().inverse()
        return type(self)(frame_id=frame_id, **p.__dict__)

    def concat(self, others: list[BatchPose]) -> BatchPose:
        """Concatenate two BatchPose objects along batch dimension.

        Args:
            other (BatchPose): The other BatchPose to concatenate.

        Returns:
            BatchPose: A new BatchPose with concatenated data.
        """
        # check that frame id is all None or all the same
        for frame_id in [other.frame_id for other in others]:
            if frame_id != self.frame_id:
                raise ValueError(
                    "All BatchPose objects must have the same frame_id."
                )
        super_ret = super().concat(others)
        return BatchPose(
            frame_id=self.frame_id,
            **super_ret.__dict__,
        )


@deprecated.deprecated(
    reason="FrameTransform will be replaced by BatchFrameTransform for "
    "simplicity and efficiency. ",
    version="0.2.0",
)
class FrameTransform(Transform3D):
    """A transformation between two coordinate frames in 3D space.

    A transformation must specify the parent and child frames it connects.
    """

    parent_frame_id: str
    """The coordinate frame ID of the parent frame."""
    child_frame_id: str
    """The coordinate frame ID of the child frame."""

    def repeat(
        self, batch_size: int, device: Device = "cpu"
    ) -> BatchFrameTransform:
        """Repeat the transformation to create a batch of transformations.

        Args:
            batch_size (int): The number of times to repeat the transformation.
            device (Device, optional): The device to put the tensors on.
                Defaults to "cpu".

        Returns:
            BatchFrameTransform: A batch of transformations with the same
                parent and child frames.
        """
        parent_impl = super().repeat(batch_size, device=device)
        return BatchFrameTransform(
            xyz=parent_impl.xyz,
            quat=parent_impl.quat,
            parent_frame_id=self.parent_frame_id,
            child_frame_id=self.child_frame_id,
        )


class BatchFrameTransform(BatchTransform3D):
    """A batch of transformations between two coordinate frames in 3D space.

    A transformation must specify the parent and child frames it connects,
    and all sample should share the same parent and child frames.
    """

    parent_frame_id: str
    """The coordinate frame ID of the parent frame."""
    child_frame_id: str
    """The coordinate frame ID of the child frame."""

    def repeat(
        self, batch_size: int, timestamps: list[int] | None = None
    ) -> BatchFrameTransform:
        """Repeat the transformation to create a batch of transformations.

        Args:
            batch_size (int): The number of times to repeat the transformation.

        Returns:
            BatchFrameTransform: A batch of transformations with the same
                parent and child frames.
        """
        parent_impl = super().repeat(batch_size, timestamps=timestamps)
        return BatchFrameTransform(
            xyz=parent_impl.xyz,
            quat=parent_impl.quat,
            timestamps=parent_impl.timestamps,
            parent_frame_id=self.parent_frame_id,
            child_frame_id=self.child_frame_id,
        )

    def compose(self, *others: Self) -> Self:
        cur_child_frame_id = self.child_frame_id
        for other in others:
            if other.parent_frame_id != cur_child_frame_id:
                raise ValueError(
                    f"Parent frame ID of {other.parent_frame_id} does not match "  # noqa: E501
                    f"the previous child frame ID {cur_child_frame_id}."
                )
            cur_child_frame_id = other.child_frame_id
        super_ret = super().compose(*others)
        return type(self)(
            parent_frame_id=self.parent_frame_id,
            child_frame_id=cur_child_frame_id,
            xyz=super_ret.xyz,
            quat=super_ret.quat,
        )

    def as_BatchPose(self) -> BatchPose:
        """Convert the BatchFrameTransform to BatchPose.

        Returns:
            BatchPose: A BatchPose object with the same
                position and orientation as the BatchFrameTransform.
        """
        return BatchPose(
            xyz=self.xyz,
            quat=self.quat,
            frame_id=self.parent_frame_id,
        )

    def inverse(self) -> Self:
        """Get the inverse of the transformations.

        Returns:
            Self: A new object with the inverse transformations.
        """
        p = super().inverse()
        return type(self)(
            parent_frame_id=self.child_frame_id,
            child_frame_id=self.parent_frame_id,
            **p.__dict__,
        )

    def concat(
        self, others: Sequence[BatchFrameTransform]
    ) -> BatchFrameTransform:
        """Concatenate two BatchFrameTransform objects along batch dimension.

        Args:
            other (BatchFrameTransform): The other BatchFrameTransform
                to concatenate.

        Returns:
            BatchFrameTransform: A new BatchFrameTransform with concatenated
                data.
        """
        # check that frame id is all None or all the same
        for frame_id in [other.parent_frame_id for other in others]:
            if frame_id != self.parent_frame_id:
                raise ValueError(
                    "All BatchFrameTransform objects must have the same "
                    "parent_frame_id."
                )
        for frame_id in [other.child_frame_id for other in others]:
            if frame_id != self.child_frame_id:
                raise ValueError(
                    "All BatchFrameTransform objects must have the same "
                    "child_frame_id."
                )
        super_ret = super().concat(others)
        return BatchFrameTransform(
            parent_frame_id=self.parent_frame_id,
            child_frame_id=self.child_frame_id,
            **super_ret.__dict__,
        )


# for deprecated naming compatibility
Pose6D = Pose
BatchPose6D = BatchPose
