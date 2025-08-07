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
from typing import List, Literal, Optional, Sequence, Union

import torch

from robo_orchard_core.utils.math.math_utils import _axis_angle_rotation
from robo_orchard_core.utils.math.transform.transform3d import _broadcast_bmm
from robo_orchard_core.utils.torch_utils import Device, make_device

__all__ = [
    "Transform2D_M",
    "Translate2D",
    "Scale2D",
    "Rotate2D",
    "Shear2D",
]


class Transform2D_M:
    """A class for 2D transformations, including rotation and translation.

    This class assumes that transformations are applied on inputs which
    are column vectors. The internal representation
    of the Nx3x3 transformation matrix is of the form:

    .. code-block:: python

        M = [
            [Rxx, Rxy, Tx],
            [Ryx, Ryy, Ty],
            [0, 0, 1],
        ]

    Args:
        dtype (torch.dtype, optional): The data type of the transformation
            matrix. Defaults to torch.float32.
        device (Device, optional): The device on which the transformation
            matrix is stored. Defaults to "cpu".
        matrix (torch.Tensor, optional): A 2D or 3D tensor representing the
            transformation matrix. If not provided, an identity matrix is used.
            The shape should be (3, 3) or (N, 3, 3) for a batch of N matrices.
            If not provided, an identity matrix of shape (1, 3, 3) is created.

    """

    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        device: Device = "cpu",
        matrix: torch.Tensor | None = None,
    ):
        if matrix is None:
            self._matrix = torch.eye(
                3, dtype=dtype, device=make_device(device)
            ).view(1, 3, 3)
        else:
            if matrix.ndim not in (2, 3):
                raise ValueError(
                    "Matrix must be 2D or 3D tensor, got shape: "
                    f"{matrix.shape}"
                )
            if matrix.shape[-2] != 3 or matrix.shape[-1] != 3:
                raise ValueError(
                    '"matrix" has to be a tensor of shape '
                    "(minibatch, 3, 3) or (3, 3)."
                )
            dtype = matrix.dtype
            device = matrix.device
            self._matrix = matrix.view(-1, 3, 3)
        self._transforms: list[
            Transform2D_M
        ] = []  # store transforms to compose
        self.device = make_device(device)
        self.dtype = dtype

    def __repr__(self) -> str:
        return f"Transform2D_M<id={id(self)}>({self.get_matrix()})"

    def get_matrix(self) -> torch.Tensor:
        """Returns a 3x3 matrix corresponding to each transform in the batch.

        If the transform was composed from others, the matrix for the composite
        transform will be returned.
        For example, if self.transforms contains transforms t1, t2, and t3, and
        given a set of points x, the following should be true:

        .. code-block:: python

            y1 = t1.compose(t2, t3).transform(x)
            y2 = t3.transform(t2.transform(t1.transform(x)))
            y1.get_matrix() == y2.get_matrix()

        Where necessary, those transforms are broadcast against each other.

        Returns:
            torch.Tensor: A (N, 3, 3) batch of transformation matrices representing
                the stored transforms. See the class documentation for the conventions.
        """  # noqa: E501
        composed_matrix = self._matrix.clone()
        if len(self._transforms) > 0:
            for other in self._transforms:
                other_matrix = other.get_matrix()
                # [points, 1] @ M1.transpose(-1,-2) @ M2.transpose(-1,-2)
                # is equivalent to
                # [points, 1] @ (M2 @ M1).transpose(-1,-2)
                # so next matrix should be applied left to the previous one
                composed_matrix = _broadcast_bmm(other_matrix, composed_matrix)
        return composed_matrix

    def __len__(self) -> int:
        """Return the number of transformations."""
        return self.get_matrix().shape[0]

    def __getitem__(
        self,
        index: Union[
            int, List[int], slice, torch.BoolTensor, torch.LongTensor
        ],
    ) -> Transform2D_M:
        """Get item.

        Args:
            index: Specifying the index of the transform to retrieve.
                Can be an int, slice, list of ints, boolean, long tensor.
                Supports negative indices.

        Returns:
            Transform3D_M: Transform3d object with selected transforms.
                The tensors are not cloned.
        """
        if isinstance(index, int):
            index = [index]
        return self.__class__(matrix=self.get_matrix()[index])

    def compose(self, *others: Transform2D_M) -> Transform2D_M:
        """Return a new Transform2D_M representing the composition of self with the given other transforms, which will be stored as an internal list.

        Args:
            *others: Any number of Transform2D_M objects

        Returns:
            Transform2D_M: A new Transform2D_M with the stored transforms
        """  # noqa: E501
        out = Transform2D_M(dtype=self.dtype, device=self.device)
        out._matrix = self._matrix.clone()
        for other in others:
            if not isinstance(other, Transform2D_M):
                msg = "Only possible to compose Transform2D_M objects; got %s"
                raise ValueError(msg % type(other))
        out._transforms = self._transforms + list(others)
        return out

    def __matmul__(self, other: Transform2D_M) -> Transform2D_M:
        """Overload the @ operator to compose transformations.

        Args:
            other: Another Transform2D_M object to compose with self.

        Returns:
            Transform2D_M: A new Transform2D_M object representing the
                composition.
        """
        if not isinstance(other, Transform2D_M):
            msg = "Only possible to compose Transform2D_M objects; got %s"
            raise ValueError(msg % type(other))
        return self.compose(other)

    def get_translation(self) -> torch.Tensor:
        """Returns the translation component of the transformation.

        Returns:
            A (N, 2) tensor representing the translation component
            of the stored transforms.
        """
        return self.get_matrix()[:, :2, 2]

    def _get_matrix_inverse(self) -> torch.Tensor:
        """Return the inverse of self._matrix."""
        return torch.inverse(self._matrix)

    def inverse(self, invert_composed: bool = False) -> Transform2D_M:
        """Returns a new Transform2D_M object that represents an inverse of the current transformation.

        Args:
            invert_composed:
                - True: First compose the list of stored transformations
                  and then apply inverse to the result. This is
                  potentially slower for classes of transformations
                  with inverses that can be computed efficiently
                  (e.g. rotations and translations).
                - False: Invert the individual stored transformations
                  independently without composing them.

        Returns:
            Transform3D_M: A new Transform3D_M object containing the inverse of the original transformation.
        """  # noqa: E501

        tinv = Transform2D_M(dtype=self.dtype, device=self.device)

        if invert_composed:
            # first compose then invert
            tinv._matrix = torch.inverse(self.get_matrix())
        else:
            # self._get_matrix_inverse() implements efficient inverse
            # of self._matrix
            i_matrix = self._get_matrix_inverse()

            # 2 cases:
            if len(self._transforms) > 0:
                # a) Either we have a non-empty list of transforms:
                # Here we take self._matrix and append its inverse at the
                # end of the reverted _transforms list. After composing
                # the transformations with get_matrix(), this correctly
                # right-multiplies by the inverse of self._matrix
                # at the end of the composition.
                tinv._transforms = [
                    t.inverse() for t in reversed(self._transforms)
                ]
                last = Transform2D_M(dtype=self.dtype, device=self.device)
                last._matrix = i_matrix
                tinv._transforms.append(last)
            else:
                # b) Or there are no stored transformations
                # we just set inverted matrix
                tinv._matrix = i_matrix
        return tinv

    def stack(self, *others: Transform2D_M) -> Transform2D_M:
        """Return a new batched Transform representing the batch elements from self and all the given other transforms all batched together.

        Args:
            *others: Any number of Transform objects

        Returns:
            Transform2D_M: A new Transform.
        """  # noqa: E501
        transforms = [self] + list(others)
        matrix = torch.cat([t.get_matrix() for t in transforms], dim=0)
        out = Transform2D_M(dtype=self.dtype, device=self.device)
        out._matrix = matrix
        return out

    def transform_points(
        self, points: torch.Tensor, eps: Optional[float] = None
    ) -> torch.Tensor:
        """Use this transform to transform a set of 2D points.

        Assumes row major ordering of the input points.

        Args:
            points: Tensor of shape (P, 2) or (N, P, 2)
            eps: If eps!=None, the argument is used to clamp the
                last coordinate before performing the final division.
                The clamping corresponds to:
                last_coord := (last_coord.sign() + (last_coord==0)) *
                torch.clamp(last_coord.abs(), eps),
                i.e. the last coordinates that are exactly 0 will
                be clamped to +eps.

        Returns:
            points_out: points of shape (N, P, 2) or (P, 2) depending
            on the dimensions of the transform
        """  # noqa: E501
        old_shape = points.shape
        old_dim = points.dim()

        # points_batch = points.clone()
        points_batch = points

        if points_batch.dim() == 2:
            points_batch = points_batch[None]  # (P, 2) -> (1, P, 2)
        if points_batch.dim() != 3:
            msg = "Expected points to have dim = 2 or dim = 3: got shape %r"
            raise ValueError(msg % repr(old_shape))

        N, P, _2 = points_batch.shape
        ones = torch.ones(N, P, 1, dtype=points.dtype, device=points.device)
        points_batch = torch.cat([points_batch, ones], dim=2)

        # transform composed matrix and then apply
        composed_matrix = self.get_matrix().transpose(-1, -2)
        points_out = _broadcast_bmm(points_batch, composed_matrix)
        denom = points_out[..., 2:]  # denominator
        if eps is not None:
            denom_sign = denom.sign() + (denom == 0.0).type_as(denom)
            denom = denom_sign * torch.clamp(denom.abs(), eps)
        points_out = points_out[..., :2] / denom

        # When transform is (1, 3, 3) and points is (P, 2) return
        # points_out of shape (P, 2)
        if points_out.shape[0] == 1 and old_dim == 2:
            points_out = points_out.reshape(old_shape)

        return points_out

    def clone(self) -> Transform2D_M:
        """Return a clone of the current Transform2D_M object."""
        other = Transform2D_M(dtype=self.dtype, device=self.device)
        other._matrix = self._matrix.clone()
        other._transforms = [t.clone() for t in self._transforms]
        return other

    def to(
        self,
        device: Device,
        copy: bool = False,
        dtype: Optional[torch.dtype] = None,
    ) -> "Transform2D_M":
        """Match functionality of torch.Tensor.to().

        If copy = True or the self Tensor is on a different device, the
        returned tensor is a copy of self with the desired torch.device.
        If copy = False and the self Tensor already has the correct torch.device,
        then self is returned.

        Args:
            device: Device (as str or torch.device) for the new tensor.
            copy: Boolean indicator whether or not to clone self. Default False.
            dtype: If not None, casts the internal tensor variables
                to a given torch.dtype.

        Returns:
            Transform2D_M: New Transforms object move to device.
        """  # noqa: E501
        device_ = make_device(device)
        dtype_ = self.dtype if dtype is None else dtype
        skip_to = self.device == device_ and self.dtype == dtype_

        if not copy and skip_to:
            return self

        other = self.clone()

        if skip_to:
            return other

        other.device = device_
        other.dtype = dtype_
        other._matrix = other._matrix.to(device=device_, dtype=dtype_)
        other._transforms = [
            t.to(device_, copy=copy, dtype=dtype_) for t in other._transforms
        ]
        return other

    def cpu(self) -> "Transform2D_M":
        return self.to("cpu")

    def cuda(self) -> "Transform2D_M":
        return self.to("cuda")


def _make_input_2d(
    data: Sequence | torch.Tensor,
    name: str,
    dtype: torch.dtype = torch.float32,
    device: Optional[Device] = None,
):
    """Handle input data and return tensor with shape (N, 2)."""
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)
    if data.dtype != dtype or (
        device is not None and data.device != make_device(device)
    ):
        data = data.to(dtype=dtype, device=device)

    shape_err_msg = f"{name} vector must be of shape (2,) or (N, 2)."
    # handle shape: Should be (2,), or (N, 2)
    if data.dim() == 1:
        if data.shape[0] != 2:
            raise ValueError(shape_err_msg)
        data = data.view(1, 2)
    else:
        if data.dim() != 2:
            raise ValueError(shape_err_msg)
        if data.shape[-1] != 2:
            raise ValueError(shape_err_msg)
    return data


class Translate2D(Transform2D_M):
    """A class for 2D translation transformations.

    This class represents a translation in 2D space, defined by a translation
    vector (Tx, Ty). The transformation matrix is of the form:

    .. code-block:: python

        M = [
            [1, 0, Tx],
            [0, 1, Ty],
            [0, 0, 1],
        ]

    Args:
        trans: A list or tensor of shape (2,) or (N, 2) representing the
            translation vector.

    """

    def __init__(
        self,
        trans: list | torch.Tensor | tuple,
        dtype: torch.dtype = torch.float32,
        device: Optional[Device] = None,
    ):
        trans = _make_input_2d(trans, name="trans", dtype=dtype, device=device)
        batch_n = trans.shape[0]
        matrix = (
            torch.eye(3, dtype=dtype, device=device)
            .view(1, 3, 3)
            .repeat(batch_n, 1, 1)
        )
        matrix[:, :2, 2] = trans

        super().__init__(dtype=dtype, device=matrix.device, matrix=matrix)

    def _get_matrix_inverse(self) -> torch.Tensor:
        """Return the inverse of self._matrix."""
        inv_mask = self._matrix.new_ones([1, 3, 3])
        inv_mask[0, :2, 2] = -1.0
        i_matrix = self._matrix * inv_mask
        return i_matrix


class Scale2D(Transform2D_M):
    """A class for 2D scaling transformations.

    This class represents a scaling transformation in 2D space, defined by a
    scaling vector (Sx, Sy). The transformation matrix is of the form:

    .. code-block:: python

        M = [
            [Sx, 0, 0],
            [0, Sy, 0],
            [0, 0, 1],
        ]
    """

    def __init__(
        self,
        scale: Sequence | torch.Tensor,
        dtype: torch.dtype = torch.float32,
        device: Optional[Device] = None,
    ):
        scale = _make_input_2d(scale, name="scale", dtype=dtype, device=device)
        batch_n = scale.shape[0]
        matrix = (
            torch.eye(3, dtype=dtype, device=device)
            .view(1, 3, 3)
            .repeat(batch_n, 1, 1)
        )
        matrix[:, 0, 0] = scale[:, 0]
        matrix[:, 1, 1] = scale[:, 1]

        super().__init__(dtype=dtype, device=matrix.device, matrix=matrix)

    def _get_matrix_inverse(self) -> torch.Tensor:
        """Return the inverse of self._matrix."""
        xy = torch.stack([self._matrix[:, i, i] for i in range(3)], dim=1)
        # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.  # noqa: E501
        ixy = 1.0 / xy
        # pyre-fixme[6]: For 1st param expected `Tensor` but got `float`.
        imat = torch.diag_embed(ixy, dim1=1, dim2=2)
        return imat


class Rotate2D(Transform2D_M):
    """A class for 2D rotation transformations.

    This class represents a rotation transformation in 2D space, defined by an
    angle in radians. The transformation matrix is of the form (when
    axis is Z):

    .. code-block:: python

        M = [
            [cos(angle), -sin(angle), 0],
            [sin(angle), cos(angle), 0],
            [0, 0, 1],
        ]

    Args:
        angle: A float, list, or tensor of shape (N,) representing the rotation
            angle in radians. If a list or tensor is provided, it should be of
            shape (N,). The angle value should be in radians.

    """

    def __init__(
        self,
        angle: float | Sequence[float] | torch.Tensor,
        axis: Literal["Z", "-Z"] = "Z",
        dtype: torch.dtype = torch.float32,
        device: Optional[Device] = None,
    ):
        if isinstance(angle, (int, float)):
            angle_arr = [angle]
        else:
            angle_arr = angle
        angle_arr = torch.asarray(angle_arr, dtype=dtype, device=device)
        if angle_arr.ndim != 1:
            raise ValueError(
                "Angle must be a 1D tensor, a single float or list of floats, "
                f"got shape: {angle_arr.shape}"
            )

        if axis == "-Z":
            angle_arr = -angle_arr

        mat = _axis_angle_rotation(angle=angle_arr, axis="Z")

        super().__init__(dtype=dtype, device=mat.device, matrix=mat)

    def _get_matrix_inverse(self) -> torch.Tensor:
        """Return the inverse of self._matrix."""
        # The inverse of a rotation matrix is its transpose
        return self._matrix.transpose(-1, -2)


class Shear2D(Transform2D_M):
    """A class for 2D shear transformations.

    This class represents a shear transformation in 2D space, defined by
    shear factors (Sxy, Syx). The transformation matrix is of the form:

    .. code-block:: python
        M = [
            [1, Sxy, 0],
            [Syx, 1, 0],
            [0, 0, 1],
        ]

    Args:
        shear: A list or tensor of shape (2,) or (N, 2) representing the
            shear factors.
    """

    def __init__(
        self,
        shear: Sequence | torch.Tensor,
        dtype: torch.dtype = torch.float32,
        device: Optional[Device] = None,
    ):
        shear = _make_input_2d(shear, name="shear", dtype=dtype, device=device)
        batch_n = shear.shape[0]
        matrix = (
            torch.eye(3, dtype=dtype, device=device)
            .view(1, 3, 3)
            .repeat(batch_n, 1, 1)
        )
        matrix[:, 0, 1] = shear[:, 0]
        matrix[:, 1, 0] = shear[:, 1]

        super().__init__(dtype=dtype, device=matrix.device, matrix=matrix)

    def _get_matrix_inverse(self) -> torch.Tensor:
        """Calculate the inverse of the shear matrix."""
        # use the formula for the inverse of a shear matrix
        shear = self._matrix[:, :2, :2]
        det = 1 - shear[:, 0, 1] * shear[:, 1, 0]
        if torch.any(det == 0):
            raise ValueError(
                "Shear matrix is singular and cannot be inverted."
            )
        inv_shear = torch.zeros_like(shear)
        inv_shear[:, 0, 0] = 1 / det
        inv_shear[:, 0, 1] = -shear[:, 0, 1] / det
        inv_shear[:, 1, 0] = -shear[:, 1, 0] / det
        inv_shear[:, 1, 1] = 1 / det
        i_matrix = (
            torch.eye(3, dtype=self.dtype, device=self.device)
            .view(1, 3, 3)
            .repeat(shear.shape[0], 1, 1)
        )
        i_matrix[:, :2, :2] = inv_shear
        return i_matrix
