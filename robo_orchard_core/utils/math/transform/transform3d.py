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
# ----------------(Copyright from Torch3d)----------------------------
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""3D transformations.

This file is modified from pytorch3d.transforms.transform3d.py to follow
the column-major convention of rotation matrix.
"""

from __future__ import annotations
import math
import os
import warnings
from typing import List, Optional, Union

import torch

from robo_orchard_core.utils.math.math_utils import (
    _axis_angle_rotation,
    matrix_to_axis_angle,
    matrix_to_quaternion,
)
from robo_orchard_core.utils.math.transform.se3 import se3_log_map
from robo_orchard_core.utils.torch_utils import Device, get_device, make_device

__all__ = [
    "Transform3D_M",
    "Translate",
    "Scale",
    "Rotate",
    "RotateAxisAngle",
]


def _safe_det_3x3(t: torch.Tensor):
    """Fast determinant calculation for a batch of 3x3 matrices.

    Note, result of this function might not be the same as `torch.det()`.
    The differences might be in the last significant digit.

    Args:
        t: Tensor of shape (N, 3, 3).

    Returns:
        Tensor of shape (N) with determinants.
    """

    det = (
        t[..., 0, 0]
        * (t[..., 1, 1] * t[..., 2, 2] - t[..., 1, 2] * t[..., 2, 1])
        - t[..., 0, 1]
        * (t[..., 1, 0] * t[..., 2, 2] - t[..., 2, 0] * t[..., 1, 2])
        + t[..., 0, 2]
        * (t[..., 1, 0] * t[..., 2, 1] - t[..., 2, 0] * t[..., 1, 1])
    )

    return det


class Transform3D_M:
    """The transform3d using Matrix representation.

    A Transform3d object encapsulates a batch of N 3D transformations, and knows
    how to transform points and normal vectors. Suppose that t is a Transform3d;
    then we can do the following:

    .. code-block:: python

        N = len(t)
        points = torch.randn(N, P, 3)
        normals = torch.randn(N, P, 3)
        points_transformed = t.transform_points(points)  # => (N, P, 3)
        normals_transformed = t.transform_normals(normals)  # => (N, P, 3)


    BROADCASTING
    Transform3d objects supports broadcasting. Suppose that t1 and tN are
    Transform3d objects with len(t1) == 1 and len(tN) == N respectively. Then we
    can broadcast transforms like this:

    .. code-block:: python

        t1.transform_points(torch.randn(P, 3))  # => (P, 3)
        t1.transform_points(torch.randn(1, P, 3))  # => (1, P, 3)
        t1.transform_points(torch.randn(M, P, 3))  # => (M, P, 3)
        tN.transform_points(torch.randn(P, 3))  # => (N, P, 3)
        tN.transform_points(torch.randn(1, P, 3))  # => (N, P, 3)


    COMBINING TRANSFORMS
    Transform3D_M objects can be combined in two ways: composing and stacking.
    Composing is function composition. Given Transform3D_M objects t1, t2, t3,
    the following all compute the same thing:

    .. code-block:: python

        y1 = t3.transform_points(t2.transform_points(t1.transform_points(x)))
        y2 = t1.compose(t2).compose(t3).transform_points(x)
        y3 = t1.compose(t2, t3).transform_points(x)


    Composing transforms should broadcast.

    if len(t1) == 1 and len(t2) == N, then len(t1.compose(t2)) == N.
    We can also stack a sequence of Transform3D_M objects, which represents
    composition along the batch dimension; then the following should compute
    the same thing.

    .. code-block:: python

        N, M = len(tN), len(tM)
        xN = torch.randn(N, P, 3)
        xM = torch.randn(M, P, 3)
        y1 = torch.cat(
            [tN.transform_points(xN), tM.transform_points(xM)], dim=0
        )
        y2 = tN.stack(tM).transform_points(torch.cat([xN, xM], dim=0))

    BUILDING TRANSFORMS
    We provide convenience methods for easily building Transform3D_M objects
    as compositions of basic transforms.

    .. code-block:: python

        # Scale by 0.5, then translate by (1, 2, 3)
        t1 = Transform3D_M().scale(0.5).translate(1, 2, 3)

        # Scale each axis by a different amount, then translate, then scale
        t2 = Transform3D_M().scale(1, 3, 3).translate(2, 3, 1).scale(2.0)

        t3 = t1.compose(t2)
        tN = t1.stack(t3, t3)


    BACKPROP THROUGH TRANSFORMS
    When building transforms, we can also parameterize them by Torch tensors;
    in this case we can backprop through the construction and application of
    Transform objects, so they could be learned via gradient descent or
    predicted by a neural network.

    .. code-block:: python

        s1_params = torch.randn(N, requires_grad=True)
        t_params = torch.randn(N, 3, requires_grad=True)
        s2_params = torch.randn(N, 3, requires_grad=True)

        t = (
            Transform3D_M()
            .scale(s1_params)
            .translate(t_params)
            .scale(s2_params)
        )
        x = torch.randn(N, 3)
        y = t.transform_points(x)
        loss = compute_loss(y)
        loss.backward()

        with torch.no_grad():
            s1_params -= lr * s1_params.grad
            t_params -= lr * t_params.grad
            s2_params -= lr * s2_params.grad

    CONVENTIONS
    We adopt a right-hand coordinate system, meaning that rotation about an axis
    with a positive angle results in a counter clockwise rotation.

    This class assumes that transformations are applied on inputs which
    are column vectors (different from pytorch3d!). The internal representation
    of the Nx4x4 transformation matrix is of the form:

    .. code-block:: python

        M = [
            [Rxx, Rxy, Rxz, Tx],
            [Ryx, Ryy, Ryz, Ty],
            [Rzx, Rzy, Rzz, Tz],
            [0, 0, 0, 1],
        ]

    To apply the transformation to points, which are row vectors, the latter are
    converted to homogeneous (4D) coordinates and right-multiplied by the
    transposed M matrix:

    .. code-block::

        points = [[0, 1, 2]]  # (1 x 3) xyz coordinates of a point
        [transformed_points, 1] ∝  M @ [points, 1].transpose(-1,-2)
        [transformed_points, 1] ∝  [points, 1] @ M.transpose(-1,-2)


    Note:
        Be careful that this class uses the column-major convention for SE(3)
        matrices, which is different from the row-major convention used in
        PyTorch3D!

    """  # noqa: E501

    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        device: Device = "cpu",
        matrix: Optional[torch.Tensor] = None,
    ) -> None:
        """Constructor.

        Args:
            dtype: The data type of the transformation matrix.
                to be used if `matrix = None`.
            device: The device for storing the implemented transformation.
                If `matrix != None`, uses the device of input `matrix`.
            matrix: A tensor of shape (4, 4) or of shape (minibatch, 4, 4)
                representing the 4x4 3D transformation matrix.
                If `None`, initializes with identity using
                the specified `device` and `dtype`.
        """

        if matrix is None:
            self._matrix = torch.eye(4, dtype=dtype, device=device).view(
                1, 4, 4
            )
        else:
            if matrix.ndim not in (2, 3):
                raise ValueError(
                    '"matrix" has to be a 2- or a 3-dimensional tensor.'
                )
            if matrix.shape[-2] != 4 or matrix.shape[-1] != 4:
                raise ValueError(
                    '"matrix" has to be a tensor of shape (minibatch, 4, 4) or (4, 4).'  # noqa: E501
                )
            # set dtype and device from matrix
            dtype = matrix.dtype
            device = matrix.device
            self._matrix = matrix.view(-1, 4, 4)

        self._transforms = []  # store transforms to compose
        self._lu = None
        self.device = make_device(device)
        self.dtype = dtype

    def __repr__(self):
        return f"Transform3D_M<id={id(self)}>({self._matrix})"

    @classmethod
    def from_rot_trans(cls, R: torch.Tensor, T: torch.Tensor) -> Transform3D_M:
        """Create a Transform3d object from rotation and translation tensors.

        Args:
            R: A tensor of shape (N, 3, 3) representing the rotation.
            T: A tensor of shape (N, 3) representing the translation.

        Returns:
            A Transform3d object with the rotation and translation.
        """
        if R.dim() == 2:
            R = R[None]
        if T.dim() == 1:
            T = T[None]
        if R.shape[0] != T.shape[0]:
            raise ValueError("R and T must have the same batch dimension.")
        if R.shape[1:] != (3, 3):
            raise ValueError("R must have shape (N, 3, 3).")
        if T.shape[1:] != (3,):
            raise ValueError("T must have shape (N, 3).")

        N = R.shape[0]
        matrix = (
            torch.eye(4, dtype=R.dtype, device=R.device)
            .unsqueeze(0)
            .repeat(N, 1, 1)
        )
        matrix[:, :3, :3] = R
        matrix[:, :3, 3] = T
        return cls(matrix=matrix)

    def __len__(self) -> int:
        return self.get_matrix().shape[0]

    def __getitem__(
        self,
        index: Union[
            int, List[int], slice, torch.BoolTensor, torch.LongTensor
        ],
    ) -> Transform3D_M:
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

    def compose(self, *others: Transform3D_M) -> Transform3D_M:
        """Return a new Transform3D_M representing the composition of self with the given other transforms, which will be stored as an internal list.

        Args:
            *others: Any number of Transform3D_M objects

        Returns:
            Transform3D_M: A new Transform3D_M with the stored transforms
        """  # noqa: E501
        out = Transform3D_M(dtype=self.dtype, device=self.device)
        out._matrix = self._matrix.clone()
        for other in others:
            if not isinstance(other, Transform3D_M):
                msg = "Only possible to compose Transform3D_M objects; got %s"
                raise ValueError(msg % type(other))
        out._transforms = self._transforms + list(others)
        return out

    def get_matrix(self) -> torch.Tensor:
        """Returns a 4x4 matrix corresponding to each transform in the batch.

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
            torch.Tensor: A (N, 4, 4) batch of transformation matrices representing
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

    def get_translation(self) -> torch.Tensor:
        """Returns the translation component of the transformation.

        Returns:
            A (N, 3) tensor representing the translation component
            of the stored transforms.
        """
        return self.get_matrix()[:, :3, 3]

    def get_rotation_quaternion(self, normalize: bool = False) -> torch.Tensor:
        """Returns the rotation component of the transformation as quaternions.

        Args:
            normalize: If True, the output quaternions will be normalized
                to unit length. This is useful if the input rotation matrices
                are not guaranteed to be valid rotation matrices.

        Returns:
            A (N, 4) tensor representing the rotation component
            of the stored transforms.
        """
        R = self.get_matrix()[:, :3, :3]
        if os.environ.get("PYTORCH3D_CHECK_ROTATION_MATRICES", "0") == "1":
            # Note: aten::all_close in the check is computationally slow, so we
            # only run the check when PYTORCH3D_CHECK_ROTATION_MATRICES is on.
            _check_valid_rotation_matrix(R, tol=1e-5)

        return matrix_to_quaternion(R, normalize_output=normalize)

    def get_rotation_axis_angle(self) -> torch.Tensor:
        """Returns the rotation component of the transformation as axis-angle.

        Returns:
            A (N, 3) tensor representing the rotation component
            of the stored transforms.
        """
        R = self.get_matrix()[:, :3, :3]
        if os.environ.get("PYTORCH3D_CHECK_ROTATION_MATRICES", "0") == "1":
            # Note: aten::all_close in the check is computationally slow, so we
            # only run the check when PYTORCH3D_CHECK_ROTATION_MATRICES is on.
            _check_valid_rotation_matrix(R, tol=1e-5)

        return matrix_to_axis_angle(R)

    def get_se3_log(
        self, eps: float = 1e-4, cos_bound: float = 1e-4
    ) -> torch.Tensor:
        """Returns a 6D SE(3) log vector corresponding to each transform in the batch.

        In the SE(3) logarithmic representation SE(3) matrices are
        represented as 6-dimensional vectors `[log_translation | log_rotation]`,
        i.e. a concatenation of two 3D vectors `log_translation` and `log_rotation`.

        The conversion from the 4x4 SE(3) matrix `transform` to the
        6D representation `log_transform = [log_translation | log_rotation]`
        is done as follows::

            log_transform = log(transform.get_matrix())
            log_translation = log_transform[:3, 3]
            log_rotation = inv_hat(log_transform[:3, :3])

        where `log` is the matrix logarithm
        and `inv_hat` is the inverse of the Hat operator [2].

        See the docstring for `se3.se3_log_map` and [1], Sec 9.4.2. for more
        detailed description.

        Args:
            eps: A threshold for clipping the squared norm of the rotation logarithm
                to avoid division by zero in the singular case.
            cos_bound: Clamps the cosine of the rotation angle to
                [-1 + cos_bound, 3 - cos_bound] to avoid non-finite outputs.
                The non-finite outputs can be caused by passing small rotation angles
                to the `acos` function in `so3_rotation_angle` of `so3_log_map`.

        Returns:
            A (N, 6) tensor, rows of which represent the individual transforms
            stored in the object as SE(3) logarithms.

        Raises:
            ValueError if the stored transform is not Euclidean (e.g. R is not a rotation
                matrix or the last column has non-zeros in the first three places).

        [1] https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf
        [2] https://en.wikipedia.org/wiki/Hat_operator
        """  # noqa: E501
        return se3_log_map(self.get_matrix(), eps, cos_bound)

    def _get_matrix_inverse(self) -> torch.Tensor:
        """Return the inverse of self._matrix."""
        return torch.inverse(self._matrix)

    def inverse(self, invert_composed: bool = False) -> Transform3D_M:
        """Returns a new Transform3D_M object that represents an inverse of the current transformation.

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

        tinv = Transform3D_M(dtype=self.dtype, device=self.device)

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
                last = Transform3D_M(dtype=self.dtype, device=self.device)
                last._matrix = i_matrix
                tinv._transforms.append(last)
            else:
                # b) Or there are no stored transformations
                # we just set inverted matrix
                tinv._matrix = i_matrix

        return tinv

    def stack(self, *others: Transform3D_M) -> Transform3D_M:
        """Return a new batched Transform3d representing the batch elements from self and all the given other transforms all batched together.

        Args:
            *others: Any number of Transform3d objects

        Returns:
            Transform3D_M: A new Transform3d.
        """  # noqa: E501
        transforms = [self] + list(others)
        matrix = torch.cat([t.get_matrix() for t in transforms], dim=0)
        out = Transform3D_M(dtype=self.dtype, device=self.device)
        out._matrix = matrix
        return out

    def transform_points(
        self, points, eps: Optional[float] = None
    ) -> torch.Tensor:
        """Use this transform to transform a set of 3D points.

        Assumes row major ordering of the input points.

        Args:
            points: Tensor of shape (P, 3) or (N, P, 3)
            eps: If eps!=None, the argument is used to clamp the
                last coordinate before performing the final division.
                The clamping corresponds to:
                last_coord := (last_coord.sign() + (last_coord==0)) *
                torch.clamp(last_coord.abs(), eps),
                i.e. the last coordinates that are exactly 0 will
                be clamped to +eps.

        Returns:
            points_out: points of shape (N, P, 3) or (P, 3) depending
            on the dimensions of the transform
        """  # noqa: E501
        points_batch = points.clone()
        if points_batch.dim() == 2:
            points_batch = points_batch[None]  # (P, 3) -> (1, P, 3)
        if points_batch.dim() != 3:
            msg = "Expected points to have dim = 2 or dim = 3: got shape %r"
            raise ValueError(msg % repr(points.shape))

        N, P, _3 = points_batch.shape
        ones = torch.ones(N, P, 1, dtype=points.dtype, device=points.device)
        points_batch = torch.cat([points_batch, ones], dim=2)

        # transform composed matrix and then apply
        composed_matrix = self.get_matrix().transpose(-1, -2)
        points_out = _broadcast_bmm(points_batch, composed_matrix)
        denom = points_out[..., 3:]  # denominator
        if eps is not None:
            denom_sign = denom.sign() + (denom == 0.0).type_as(denom)
            denom = denom_sign * torch.clamp(denom.abs(), eps)
        points_out = points_out[..., :3] / denom

        # When transform is (1, 4, 4) and points is (P, 3) return
        # points_out of shape (P, 3)
        if points_out.shape[0] == 1 and points.dim() == 2:
            points_out = points_out.reshape(points.shape)

        return points_out

    def transform_normals(self, normals) -> torch.Tensor:
        """Use this transform to transform a set of normal vectors.

        Args:
            normals: Tensor of shape (P, 3) or (N, P, 3)

        Returns:
            normals_out: Tensor of shape (P, 3) or (N, P, 3) depending
            on the dimensions of the transform
        """
        if normals.dim() not in [2, 3]:
            msg = "Expected normals to have dim = 2 or dim = 3: got shape %r"
            raise ValueError(msg % (normals.shape,))
        composed_matrix = self.get_matrix()

        # TODO: inverse is bad! Solve a linear system instead
        mat = composed_matrix[:, :3, :3]
        normals_out = _broadcast_bmm(normals, mat.inverse())

        # This doesn't pass unit tests. TODO investigate further
        # if self._lu is None:
        #     self._lu = self._matrix[:, :3, :3].transpose(1, 2).lu()
        # normals_out = normals.lu_solve(*self._lu)

        # When transform is (1, 4, 4) and normals is (P, 3) return
        # normals_out of shape (P, 3)
        if normals_out.shape[0] == 1 and normals.dim() == 2:
            normals_out = normals_out.reshape(normals.shape)

        return normals_out

    def translate(self, *args, **kwargs) -> "Transform3D_M":
        return self.compose(
            Translate(*args, device=self.device, dtype=self.dtype, **kwargs)
        )

    def scale(self, *args, **kwargs) -> "Transform3D_M":
        return self.compose(
            Scale(*args, device=self.device, dtype=self.dtype, **kwargs)
        )

    def rotate(self, *args, **kwargs) -> "Transform3D_M":
        return self.compose(
            Rotate(*args, device=self.device, dtype=self.dtype, **kwargs)
        )

    def rotate_axis_angle(self, *args, **kwargs) -> "Transform3D_M":
        return self.compose(
            RotateAxisAngle(
                *args, device=self.device, dtype=self.dtype, **kwargs
            )
        )

    def clone(self) -> "Transform3D_M":
        """Deep copy of Transforms object.

        All internal tensors are cloned individually.

        Returns:
            Transform3D_M: New Transforms object.
        """
        other = Transform3D_M(dtype=self.dtype, device=self.device)
        if self._lu is not None:
            other._lu = [elem.clone() for elem in self._lu]  # type: ignore
        other._matrix = self._matrix.clone()
        other._transforms = [t.clone() for t in self._transforms]
        return other

    def to(
        self,
        device: Device,
        copy: bool = False,
        dtype: Optional[torch.dtype] = None,
    ) -> "Transform3D_M":
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
            Transform3D_M: New Transforms object move to device.
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

    def cpu(self) -> "Transform3D_M":
        return self.to("cpu")

    def cuda(self) -> "Transform3D_M":
        return self.to("cuda")


class Translate(Transform3D_M):
    """Create a new Transform3D_M representing 3D translations.

    Option I: Translate(xyz, dtype=torch.float32, device='cpu')
        xyz should be a tensor of shape (N, 3)

    Option II: Translate(x, y, z, dtype=torch.float32, device='cpu')
        Here x, y, and z will be broadcast against each other and
        concatenated to form the translation. Each can be:
        - A python scalar
        - A torch scalar
        - A 1D torch tensor
    """

    def __init__(
        self,
        x,
        y=None,
        z=None,
        dtype: torch.dtype = torch.float32,
        device: Optional[Device] = None,
    ) -> None:
        xyz = _handle_input(x, y, z, dtype, device, "Translate")
        super().__init__(device=xyz.device, dtype=dtype)
        N = xyz.shape[0]

        mat = torch.eye(4, dtype=dtype, device=self.device)
        mat = mat.view(1, 4, 4).repeat(N, 1, 1)
        mat[:, :3, 3] = xyz
        self._matrix = mat

    def _get_matrix_inverse(self) -> torch.Tensor:
        """Return the inverse of self._matrix."""
        inv_mask = self._matrix.new_ones([1, 4, 4])
        inv_mask[0, :3, 3] = -1.0
        i_matrix = self._matrix * inv_mask
        return i_matrix

    def __getitem__(
        self,
        index: Union[
            int, List[int], slice, torch.BoolTensor, torch.LongTensor
        ],
    ) -> "Transform3D_M":
        """Get item.

        Args:
            index: Specifying the index of the transform to retrieve.
                Can be an int, slice, list of ints, boolean, long tensor.
                Supports negative indices.

        Returns:
            Transform3D_M object with selected transforms. The tensors are not cloned.
        """  # noqa: E501
        if isinstance(index, int):
            index = [index]
        return self.__class__(self.get_matrix()[index, :3, 3])


class Scale(Transform3D_M):
    """A Transform3D_M representing a scaling operation, with different scale factors along each coordinate axis.

    Option I: Scale(s, dtype=torch.float32, device='cpu')
        s can be one of
            - Python scalar or torch scalar: Single uniform scale
            - 1D torch tensor of shape (N,): A batch of uniform scale
            - 2D torch tensor of shape (N, 3): Scale differently along each axis

    Option II: Scale(x, y, z, dtype=torch.float32, device='cpu')
        Each of x, y, and z can be one of
            - python scalar
            - torch scalar
            - 1D torch tensor
    """  # noqa: E501

    def __init__(
        self,
        x,
        y=None,
        z=None,
        dtype: torch.dtype = torch.float32,
        device: Optional[Device] = None,
    ) -> None:
        xyz = _handle_input(
            x, y, z, dtype, device, "scale", allow_singleton=True
        )
        super().__init__(device=xyz.device, dtype=dtype)
        N = xyz.shape[0]

        # TODO: Can we do this all in one go somehow?
        mat = torch.eye(4, dtype=dtype, device=self.device)
        mat = mat.view(1, 4, 4).repeat(N, 1, 1)
        mat[:, 0, 0] = xyz[:, 0]
        mat[:, 1, 1] = xyz[:, 1]
        mat[:, 2, 2] = xyz[:, 2]
        self._matrix = mat

    def _get_matrix_inverse(self) -> torch.Tensor:
        """Return the inverse of self._matrix."""
        xyz = torch.stack([self._matrix[:, i, i] for i in range(4)], dim=1)
        # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.  # noqa: E501
        ixyz = 1.0 / xyz
        # pyre-fixme[6]: For 1st param expected `Tensor` but got `float`.
        imat = torch.diag_embed(ixyz, dim1=1, dim2=2)
        return imat

    def __getitem__(
        self,
        index: Union[
            int, List[int], slice, torch.BoolTensor, torch.LongTensor
        ],
    ) -> "Transform3D_M":
        """Get item.

        Args:
            index: Specifying the index of the transform to retrieve.
                Can be an int, slice, list of ints, boolean, long tensor.
                Supports negative indices.

        Returns:
            Transform3D_M: Object with selected transforms. The tensors are not cloned.
        """  # noqa: E501
        if isinstance(index, int):
            index = [index]
        mat = self.get_matrix()[index]
        x = mat[:, 0, 0]
        y = mat[:, 1, 1]
        z = mat[:, 2, 2]
        return self.__class__(x, y, z)


class Rotate(Transform3D_M):
    """Create a new Transform3D_M representing 3D rotation using a rotation matrix as the input.

    Args:
        R: a tensor of shape (3, 3) or (N, 3, 3)
        orthogonal_tol: tolerance for the test of the orthogonality of R

    """  # noqa: E501

    def __init__(
        self,
        R: torch.Tensor,
        dtype: torch.dtype = torch.float32,
        device: Optional[Device] = None,
        orthogonal_tol: float = 1e-5,
    ) -> None:
        device_ = get_device(R, device)
        super().__init__(device=device_, dtype=dtype)
        if R.dim() == 2:
            R = R[None]
        if R.shape[-2:] != (3, 3):
            msg = "R must have shape (3, 3) or (N, 3, 3); got %s"
            raise ValueError(msg % repr(R.shape))
        R = R.to(device=device_, dtype=dtype)
        if os.environ.get("PYTORCH3D_CHECK_ROTATION_MATRICES", "0") == "1":
            # Note: aten::all_close in the check is computationally slow, so we
            # only run the check when PYTORCH3D_CHECK_ROTATION_MATRICES is on.
            _check_valid_rotation_matrix(R, tol=orthogonal_tol)
        N = R.shape[0]
        mat = torch.eye(4, dtype=dtype, device=device_)
        mat = mat.view(1, 4, 4).repeat(N, 1, 1)
        mat[:, :3, :3] = R
        self._matrix = mat

    def _get_matrix_inverse(self) -> torch.Tensor:
        """Return the inverse of self._matrix."""
        return self._matrix.permute(0, 2, 1).contiguous()

    def __getitem__(
        self,
        index: Union[
            int, List[int], slice, torch.BoolTensor, torch.LongTensor
        ],
    ) -> "Transform3D_M":
        """Get item.

        Args:
            index: Specifying the index of the transform to retrieve.
                Can be an int, slice, list of ints, boolean, long tensor.
                Supports negative indices.

        Returns:
            Transform3D_M object with selected transforms. The tensors are not cloned.
        """  # noqa: E501
        if isinstance(index, int):
            index = [index]
        return self.__class__(self.get_matrix()[index, :3, :3])


class RotateAxisAngle(Rotate):
    """Create a new Transform3D_M representing 3D rotation about an axis by an angle.

    Assuming a right-hand coordinate system, positive rotation angles result
    in a counter clockwise rotation.

    Args:
        angle:
            - A torch tensor of shape (N,)
            - A python scalar
            - A torch scalar
        axis:
            string: one of ["X", "Y", "Z"] indicating the axis about which
            to rotate.
            NOTE: All batch elements are rotated about the same axis.
    """  # noqa: E501

    def __init__(
        self,
        angle,
        axis: str = "X",
        degrees: bool = True,
        dtype: torch.dtype = torch.float32,
        device: Optional[Device] = None,
    ) -> None:
        axis = axis.upper()
        if axis not in ["X", "Y", "Z"]:
            msg = "Expected axis to be one of ['X', 'Y', 'Z']; got %s"
            raise ValueError(msg % axis)
        angle = _handle_angle_input(angle, dtype, device, "RotateAxisAngle")
        angle = (angle / 180.0 * math.pi) if degrees else angle
        # We assume the points on which this transformation will be applied
        # are row vectors. The rotation matrix returned from
        # _axis_angle_rotation is for transforming column vectors.
        R = _axis_angle_rotation(axis, angle)
        super().__init__(device=angle.device, R=R, dtype=dtype)


def _handle_coord(c, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Helper function for _handle_input.

    Args:
        c: Python scalar, torch scalar, or 1D torch tensor

    Returns:
        c_vec: 1D torch tensor
    """
    if not torch.is_tensor(c):
        c = torch.tensor(c, dtype=dtype, device=device)
    if c.dim() == 0:
        c = c.view(1)
    if c.device != device or c.dtype != dtype:
        c = c.to(device=device, dtype=dtype)
    return c


def _handle_input(
    x,
    y,
    z,
    dtype: torch.dtype,
    device: Optional[Device],
    name: str,
    allow_singleton: bool = False,
) -> torch.Tensor:
    """Helper function to handle parsing logic for building transforms.

    The output is always a tensor of shape (N, 3), but there are several
    types of allowed input.

    Case I: Single Matrix
        In this case x is a tensor of shape (N, 3), and y and z are None. Here
        just return x.

    Case II: Vectors and Scalars
        In this case each of x, y, and z can be one of the following
            - Python scalar
            - Torch scalar
            - Torch tensor of shape (N, 1) or (1, 1)
        In this case x, y and z are broadcast to tensors of shape (N, 1)
        and concatenated to a tensor of shape (N, 3)

    Case III: Singleton (only if allow_singleton=True)
        In this case y and z are None, and x can be one of the following:
            - Python scalar
            - Torch scalar
            - Torch tensor of shape (N, 1) or (1, 1)
        Here x will be duplicated 3 times, and we return a tensor of shape (N, 3)

    Returns:
        xyz: Tensor of shape (N, 3)
    """  # noqa: E501
    device_ = get_device(x, device)
    # If x is actually a tensor of shape (N, 3) then just return it
    if torch.is_tensor(x) and x.dim() == 2:
        if x.shape[1] != 3:
            msg = "Expected tensor of shape (N, 3); got %r (in %s)"
            raise ValueError(msg % (x.shape, name))
        if y is not None or z is not None:
            msg = "Expected y and z to be None (in %s)" % name
            raise ValueError(msg)
        return x.to(device=device_, dtype=dtype)

    if allow_singleton and y is None and z is None:
        y = x
        z = x

    # Convert all to 1D tensors
    xyz = [_handle_coord(c, dtype, device_) for c in [x, y, z]]

    # Broadcast and concatenate
    sizes = [c.shape[0] for c in xyz]
    N = max(sizes)
    for c in xyz:
        if c.shape[0] != 1 and c.shape[0] != N:
            msg = "Got non-broadcastable sizes %r (in %s)" % (sizes, name)
            raise ValueError(msg)
    xyz = [c.expand(N) for c in xyz]
    xyz = torch.stack(xyz, dim=1)
    return xyz


def _handle_angle_input(
    x, dtype: torch.dtype, device: Optional[Device], name: str
) -> torch.Tensor:
    """Helper function for building a rotation function using angles.

    The output is always of shape (N,).

    The input can be one of:
        - Torch tensor of shape (N,)
        - Python scalar
        - Torch scalar
    """
    device_ = get_device(x, device)
    if torch.is_tensor(x) and x.dim() > 1:
        msg = "Expected tensor of shape (N,); got %r (in %s)"
        raise ValueError(msg % (x.shape, name))
    else:
        return _handle_coord(x, dtype, device_)


def _broadcast_bmm(a, b) -> torch.Tensor:
    """Batch multiply two matrices and broadcast if necessary.

    Args:
        a: torch tensor of shape (P, K) or (M, P, K)
        b: torch tensor of shape (N, K, K)

    Returns:
        a and b broadcast multiplied. The output batch dimension is max(N, M).

    To broadcast transforms across a batch dimension if M != N then
    expect that either M = 1 or N = 1. The tensor with batch dimension 1 is
    expanded to have shape N or M.
    """
    if a.dim() == 2:
        a = a[None]
    if len(a) != len(b):
        if not ((len(a) == 1) or (len(b) == 1)):
            msg = "Expected batch dim for bmm to be equal or 1; got %r, %r"
            raise ValueError(msg % (a.shape, b.shape))
        if len(a) == 1:
            a = a.expand(len(b), -1, -1)
        if len(b) == 1:
            b = b.expand(len(a), -1, -1)
    return a.bmm(b)


@torch.no_grad()
def _check_valid_rotation_matrix(R, tol: float = 1e-7) -> None:
    """Determine if R is a valid rotation matrix by checking it satisfies the following conditions.

    ``RR^T = I and det(R) = 1``

    Args:
        R: an (N, 3, 3) matrix

    Returns:
        None

    Emits a warning if R is an invalid rotation matrix.
    """  # noqa: E501
    N = R.shape[0]
    eye = torch.eye(3, dtype=R.dtype, device=R.device)
    eye = eye.view(1, 3, 3).expand(N, -1, -1)
    orthogonal = torch.allclose(R.bmm(R.transpose(-2, -1)), eye, atol=tol)
    det_R = _safe_det_3x3(R)
    no_distortion = torch.allclose(det_R, torch.ones_like(det_R))
    if not (orthogonal and no_distortion):
        msg = "R is not a valid rotation matrix"
        warnings.warn(msg)
    return
