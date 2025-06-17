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
#
# -----------------------------------------------------------------------------
# Portions of this file are derived from Pytorch3D
# (https://github.com/facebookresearch/pytorch3d).
# The original PyTorch3D code is licensed under the BSD-3 license.
#
# Original PyTorch3D Copyright Notice:
#
# BSD License
#
# For PyTorch3D software
#
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
#  * Neither the name Meta nor the names of its contributors may be used to
#    endorse or promote products derived from this software without specific
#   prior written permission.
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
# You can find the original PyTorch3D source code at:
# https://github.com/facebookresearch/pytorch3d
#
# Modifications to the derived code, if any, are Copyright (c) 2025
# Horizon Robotics and are licensed under the Apache License, Version 2.0.
# The combined work in this file is distributed under the Apache License,
# Version 2.0, subject to the conditions of the BSD-3 license for the portions
# derived from PyTorch3D.

"""Math utilities for transformations and conversions."""

import functools
import math
import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from robo_orchard_core.utils.math.coord_convention import CoordConventionType
from robo_orchard_core.utils.torch_utils import (
    Device,
    convert_to_tensors_and_broadcast,
)


def torch_jit_compile(f):
    """Decorator to compile a function with torch.jit.script.

    This decorator will keep the function signature and docstring intact.
    """
    if not TYPE_CHECKING:
        return torch.jit.script(f)
    else:
        return functools.wraps(f)(torch.jit.script(f))


@torch_jit_compile
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


@torch.no_grad()
def check_valid_rotation_matrix(R, tol: float = 1e-7) -> bool:
    """Determine if R is a valid rotation matrix by checking it satisfies the following conditions.

    ``RR^T = I and det(R) = 1``


    This is a copy of the function from PyTorch3D.

    Args:
        R: an (N, 3, 3) matrix

    Returns:
        bool

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
        return False
    return True


def normalize(
    x: torch.Tensor, dim: int = -1, eps: float = 1e-9
) -> torch.Tensor:
    """Normalize a tensor along a given dimension.

    Args:
        x: The tensor to normalize, of shape (N, dims).
        dim: The dimension along which to normalize. Defaults to -1.
        eps: A small value to avoid division by zero. Defaults to 1e-8.

    Returns:
        The normalized tensor.

    """
    return x / (torch.linalg.norm(x, dim=dim, keepdim=True).clamp(min=eps))


def quaternion_standardize(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert a unit quaternion to a standard form.

    The standard form is the one in which the real part is non negative.

    Note:
        Quaternion representations have a singularity since ``q`` and ``-q``
        represent the same rotation. This function ensures the real part of
        the quaternion is non-negative.

    Args:
        quaternions: The quaternion orientation in (w, x, y, z).
            Shape is (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L373C1-L385C77

    """  # noqa: E501
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


@torch_jit_compile
def quaternion_raw_multiply(
    a: torch.Tensor, b: torch.Tensor, batch_mode: bool = False
) -> torch.Tensor:
    """Multiply two quaternions.

    Usual torch rules for broadcasting apply. If the input tensors are in
    batch mode, the first dimension is the batch size.

    Args:
        a: Quaternions as tensor of shape (..., 4) or (B, ..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4) or (B, ..., 4), real part first.
        batch_mode: If True, the input tensors are in batch mode, which means the
            first dimension is the batch size. Defaults to False.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4) or or (B, ..., 4).

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py

    """  # noqa: E501
    if batch_mode:
        if a.shape[0] != b.shape[0]:
            raise ValueError("Input tensors must have the same batch size.")

        if a.dim() < b.dim():
            a = a[:, None, ...]
        elif b.dim() < a.dim():
            b = b[:, None, ...]

    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    ret = torch.stack((ow, ox, oy, oz), -1)
    return ret


@torch_jit_compile
def quaternion_multiply(
    a: torch.Tensor, b: torch.Tensor, batch_mode: bool = False
) -> torch.Tensor:
    """Multiply two quaternions.

    Multiply two quaternions representing rotations, returning the quaternion
    representing their composition, i.e. the versor with nonnegative real part.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4) or (B, ..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4) or (B, ..., 4), real part first.
        batch_mode: If True, the input tensors are in batch mode, which means the
            first dimension is the batch size. Defaults to False.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4) or or (B, ..., 4).

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    """  # noqa: E501
    ab = quaternion_raw_multiply(a, b, batch_mode=batch_mode)
    return quaternion_standardize(ab)


@torch_jit_compile
def quaternion_invert(quaternion: torch.Tensor) -> torch.Tensor:
    """Compute the inverse of a quaternion.

    Given a quaternion representing rotation, get the quaternion representing
    its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    """

    scaling = torch.tensor([1, -1, -1, -1], device=quaternion.device)
    return quaternion * scaling


@torch_jit_compile
def quaternion_apply_point(
    quaternion: torch.Tensor, point: torch.Tensor, batch_mode: bool = False
) -> torch.Tensor:
    r"""Rotates a point or vector relative to a fixed coordinate frame.

    This function applies :math:`p' = q \times p \times q^{-1}`
    where p is the point and q is the quaternion.

    Args:
        quaternion: Tensor of quaternions, real part first, of shape (..., 4)
            or (B, ..., 4).
        point: Tensor of 3D points of shape (..., 3) or (B, ..., 3).

    Returns:
        Tensor of rotated points or vectors of shape (..., 3) or (B, ..., 3).

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    """
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((real_parts, point), -1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(
            quaternion, point_as_quaternion, batch_mode=batch_mode
        ),
        quaternion_invert(quaternion),
        batch_mode=batch_mode,
    )
    return out[..., 1:]


@torch_jit_compile
def quaternion_apply_frame(
    quaternion: torch.Tensor, point: torch.Tensor, batch_mode: bool = False
):
    r"""Rotates the coordinate frame while keeping the vector/point fixed.

    This function applies :math:`p' = q^{-1} \times p \times q`
    where p is the point and q is the quaternion.

    Args:
        quaternion: Tensor of quaternions, real part first, of shape (..., 4)
            or (B, ..., 4).
        point: Tensor of 3D points of shape (..., 3) or (B, ..., 3).

    Returns:
        Tensor of rotated points or vectors of shape (..., 3) or (B, ..., 3).

    """

    return quaternion_apply_point(
        quaternion_invert(quaternion), point, batch_mode=batch_mode
    )


@torch_jit_compile
def quaternion_left_division(
    qa: torch.Tensor, qb: torch.Tensor
) -> torch.Tensor:
    r"""Compute the difference between two quaternions.

    It performs the operation :math:`q = q_a^{-1} \times q_b`, which means
    :math:`q_a \times q = q_b`. The result is an identity quaternion if
    the two quaternions are the same. This operator is also known as the
    left division of q_b by q_a.

    Args:
        qa (torch.Tensor): The first quaternion, unit quaternions.
        qb (torch.Tensor): The second quaternion, unit quaternions.

    Returns:
        torch.Tensor: The relative difference from q_a to q_b.
    """
    return quaternion_multiply(quaternion_invert(qa), qb)


@torch_jit_compile
def quaternion_right_division(
    qa: torch.Tensor, qb: torch.Tensor
) -> torch.Tensor:
    r"""Compute the difference between two quaternions.

    It performs the operation :math:`q = q_a \times q_b^{-1}`, which means
    :math:`q \times q_b = q_a`. The result is an identity quaternion if
    the two quaternions are the same. This operator is also known as the
    right division of q_a by q_b.

    Args:
        qa (torch.Tensor): The first quaternion, unit quaternions.
        qb (torch.Tensor): The second quaternion, unit quaternions.

    Returns:
        torch.Tensor: The relative difference from q_a to q_b.
    """
    return quaternion_multiply(qa, quaternion_invert(qb))


@torch_jit_compile
def quaternion_to_axis_angle(
    quaternions: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first, as tensor of
            shape (..., 4).
        eps: The tolerance for Taylor approximation. Defaults to 1.0e-6.

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles

    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # Apply Taylor approximation for small angles
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


@torch_jit_compile
def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as quaternions to rotation matrices.

    The rotation matrix is expected of the form:

    .. code-block:: python

        R = [[Rxx, Rxy, Rxz], [Ryx, Ryy, Ryz], [Rzx, Rzy, Rzz]]

    You can construct SE(3) matrices by concatenating the rotation
    matrix with the translation vector like:

    .. code-block:: text

            [R, T]
            [0, 1],
        or
            [inv(R), 0]
            [T,      1]

    Args:
        quaternions: quaternions in (w, x, y, z),
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).


    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types
    # `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """torch.sqrt(torch.max(0, x)) but with a zero subgradient where x is 0.

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L93C1-L104C15

    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """Return the rotation matrices for one of the rotations.

    The rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z", for  intrinsic rotations.
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L169

    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3). Each matrix is
        expected of the form:

        .. code-block:: python

            M = [
                [Rxx, Rxy, Rxz],
                [Ryx, Ryy, Ryz],
                [Rzx, Rzy, Rzz],
            ]


    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    """

    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


@torch_jit_compile
def matrix_to_quaternion(
    matrix: torch.Tensor, normalize_output: bool = False
) -> torch.Tensor:
    """Convert rotations given as rotation matrices to quaternions.

    This function is copied from PyTorch3D to avoid dependency:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L107


    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        normalize_output: If True, the output quaternions will be normalized
            to unit quaternions. Defaults to False.

    Returns:
        quaternions in (w, x, y, z), as tensor of shape (..., 4).

    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack(
                [q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1
            ),
            torch.stack(
                [m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1
            ),
            torch.stack(
                [m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1
            ),
            torch.stack(
                [m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1
            ),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important;
    # if q_abs is small, the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be
    # same (up to a sign), forall i; we pick the best-conditioned one
    # (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    if not normalize_output:
        return quaternion_standardize(out)
    else:
        return quaternion_standardize(normalize(out))


@torch_jit_compile
def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as rotation matrices to axis/angle.

    The rotation matrix is expected of the form:

    .. code-block:: python

        R = [[Rxx, Rxy, Rxz], [Ryx, Ryy, Ryz], [Rzx, Rzy, Rzz]]

    You can construct SE(3) matrices by concatenating the rotation
    matrix with the translation vector like:

    .. code-block:: text

            [R, T]
            [0, 1],
        or
            [inv(R), 0]
            [T,      1]

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def make_axis_angle(
    axis: torch.Tensor, angle: torch.Tensor | float
) -> torch.Tensor:
    """Create rotations given as axis/angle.

    Args:
        axis: The axis of rotation as a tensor of shape (..., 3).
        angle: The angle of rotation in radians as a tensor of shape (...,).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.

    """

    if isinstance(angle, float):
        return normalize(axis) * angle
    else:
        return normalize(axis) * angle.unsqueeze(-1)  # type: ignore


def get_axis_and_angle_from_axis_angle(
    axis_angle: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get the axis and angle from axis/angle representation.

    Args:
        axis_angle: The axis/angle representation of the rotation as a
            tensor of shape (..., 3).

    Returns:
        A tuple containing the axis and angle as tensors of shape (..., 3)
            and (...,).

    """
    axis = normalize(axis_angle)
    angle = torch.norm(axis_angle, p=2, dim=-1)
    return axis, angle


@torch_jit_compile
def axis_angle_to_quaternion(
    axis_angle: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.
        eps: The tolerance for Taylor approximation. Defaults to 1.0e-6.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5

    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # Apply Taylor approximation for small angles
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles],
        dim=-1,
    )
    return quaternions


@torch_jit_compile
def euler_angles_to_matrix(
    euler_angles: torch.Tensor, convention: str
) -> torch.Tensor:
    """Convert rotations given as Euler angles in radians to rotation matrices.

    The rotation matrix is expected of the form:

    .. code-block:: python

        R = [
            [Rxx, Rxy, Rxz],
            [Ryx, Ryy, Ryz],
            [Rzx, Rzy, Rzz],
        ]

    You can construct SE(3) matrices by concatenating the rotation
    matrix with the translation vector like:

    .. code-block:: text

        R =[[R, T],
            [0, 1]]

        or

        R =[[inv(R), 0],
            [T,      1]]

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}, for intrinsic rotations.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L199

    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(
            convention, torch.unbind(euler_angles, -1), strict=False
        )
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


def convert_orientation_convention(
    orientation: torch.Tensor,
    origin: CoordConventionType = "opengl",
    target: CoordConventionType = "cam",
) -> torch.Tensor:
    """Convert orientation from one convention to another.

    The orientation conventions are defined as follows:

    - ``"opengl"`` - forward axis: ``-Z`` - up axis: ``+Y`` - Applied
      in the OpenGL (Usd.Camera) convention. OpenGL uses a right-handed
      coordinate system.

    - ``"cam"``    - forward axis: ``+Z`` - up axis: ``-Y`` - Applied
      in the cam convention. It is a right-handed coordinate system.

    - ``"world"``  - forward axis: ``+X`` - up axis: ``+Z`` - Applied
      in the World Frame convention.


    Args:
        orientation: The orientation as a quaternion in (w, x, y, z).
            Shape is (..., 4).
        origin: The origin convention. One of "opengl", "cam", "world".
            Defaults to "opengl".
        target: The target convention. One of "opengl", "cam", "world".
            Defaults to "cam".

    Returns:
        The orientation in the target convention as a quaternion
            in (w, x, y, z). Shape is (..., 4).

    Reference:
        https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/utils/math.py

    """
    if origin == target:
        return orientation.clone()

    # unify all conventions to opengl
    if origin == "cam":
        rot_mat = quaternion_to_matrix(orientation)
        rot_mat[:, :, 2] = -rot_mat[:, :, 2]
        rot_mat[:, :, 1] = -rot_mat[:, :, 1]
        quat_gl = matrix_to_quaternion(rot_mat)
    elif origin == "world":
        rot_mat = quaternion_to_matrix(orientation)
        rot_mat[:, :, 0] = -rot_mat[:, :, 0]
        rot_mat[:, :, 2] = -rot_mat[:, :, 2]
        rot_mat = torch.matmul(
            rot_mat,
            euler_angles_to_matrix(
                torch.tensor(
                    [math.pi / 2, -math.pi / 2, 0], device=orientation.device
                ),
                "XYZ",
            ),
        )
        quat_gl = matrix_to_quaternion(rot_mat)
    else:
        quat_gl = orientation
        if target != "opengl":
            rot_mat = quaternion_to_matrix(orientation)
        else:
            return quat_gl.clone()

    # convert from opengl to target
    if target == "opengl":
        return quat_gl.clone()
    elif target == "cam":
        rot_mat[:, :, 2] = -rot_mat[:, :, 2]
        rot_mat[:, :, 1] = -rot_mat[:, :, 1]
        return matrix_to_quaternion(rot_mat)
    elif target == "world":
        rot_mat = torch.matmul(
            rot_mat,
            euler_angles_to_matrix(
                torch.tensor(
                    [math.pi / 2, -math.pi / 2, 0], device=orientation.device
                ),
                "XYZ",
            ).T,
        )
        return matrix_to_quaternion(rot_mat)
    else:
        raise ValueError(f"Unsupported conversion from {origin} to {target}.")


@torch_jit_compile
def _rotation_matrix_from_view_impl(
    camera_position: torch.Tensor,
    at: torch.Tensor,
    up: torch.Tensor,
    view_convention: str = "opengl",
):
    for t, n in zip(
        [camera_position, at, up],
        ["camera_position", "at", "up"],
        strict=False,
    ):
        if t.shape[-1] != 3:
            msg = "Expected arg %s to have shape (N, 3); got %r"
            raise ValueError(msg % (n, t.shape))

    backward_axis = F.normalize(camera_position - at, eps=1e-5)
    right_axis = F.normalize(torch.cross(up, backward_axis, dim=1), eps=1e-5)
    up_axis = F.normalize(
        torch.cross(backward_axis, right_axis, dim=1), eps=1e-5
    )
    is_close = torch.isclose(right_axis, torch.tensor(0.0), atol=5e-3).all(
        dim=1, keepdim=True
    )
    if is_close.any():
        replacement = F.normalize(
            torch.cross(up_axis, backward_axis, dim=1), eps=1e-5
        )
        right_axis = torch.where(is_close, replacement, right_axis)

    if view_convention == "opengl":
        R = torch.cat(
            (
                right_axis[:, None, :],
                up_axis[:, None, :],
                backward_axis[:, None, :],
            ),
            dim=1,
        )
    elif view_convention == "cam":
        R = torch.cat(
            (
                right_axis[:, None, :],
                -up_axis[:, None, :],
                -backward_axis[:, None, :],
            ),
            dim=1,
        )
    elif view_convention == "world":
        R = torch.cat(
            (
                -backward_axis[:, None, :],
                -right_axis[:, None, :],
                up_axis[:, None, :],
            ),
            dim=1,
        )
    else:
        raise ValueError(f"Unsupported view_convention {view_convention}.")

    return R.transpose(1, 2)


def rotation_matrix_from_view(
    camera_position,
    at=((0, 0, 0),),
    up=((0, 0, 1),),
    device: Device = "cpu",
    view_convention: CoordConventionType = "opengl",
) -> torch.Tensor:
    """Create a rotation matrix from a camera position and view direction.

    This function takes a vector 'camera_position' which specifies the location
    of the camera in local coordinates and two vectors `at` and `up` which
    indicate the position of the object and the up directions of the local
    coordinate system respectively.

    The output is a rotation matrix representing the rotation matix of view
    coordinates (such as openGL convension, -Z forward, +Y up, +X right) to
    world coordinates.

    Warning:
        When the camera are facing down or up along the world Up axis, the
        right axis is not well defined.

    Args:
        camera_position: position of the camera in local coordinates.
        at: position of the object in local coordinates
        up: vector specifying the up direction in the world coordinate frame.
            default is +Z axis.
        view_convention: The convention of the view. One of "opengl", "cam",
            "world". Defaults to "opengl".

    The inputs camera_position, at and up can each be a
        - 3 element tuple/list
        - torch tensor of shape (1, 3)
        - torch tensor of shape (N, 3)

    The vectors are broadcast against each other so they all have shape (N, 3).

    Returns:
        R: (N, 3, 3) batched rotation matrices

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/renderer/cameras.py#L1637

    """
    # Format input and broadcast
    broadcasted_args = convert_to_tensors_and_broadcast(
        camera_position, at, up, device=device
    )
    camera_position, at, up = broadcasted_args

    return _rotation_matrix_from_view_impl(
        camera_position, at, up, view_convention=view_convention
    )


@torch_jit_compile
def skew_symmetric_matrix(vec: torch.Tensor) -> torch.Tensor:
    """Computes the skew-symmetric matrix of a vector.

    This is also known as the Hat operator of 3D vectors.

    Args:
        vec: The input vector. Shape is (3,) or (N, 3).

    Returns:
        The skew-symmetric matrix. Shape is (1, 3, 3) or (N, 3, 3).

    Raises:
        ValueError: If input tensor is not of shape (..., 3).

    Reference:
        https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/utils/math.py

    """
    # check input is correct
    if vec.shape[-1] != 3:
        raise ValueError(
            f"Expected input vector shape mismatch: {vec.shape} != (..., 3)."
        )
    # unsqueeze the last dimension
    if vec.ndim == 1:
        vec = vec.unsqueeze(0)
    # create a skew-symmetric matrix
    skew_sym_mat = torch.zeros(
        vec.shape[0], 3, 3, device=vec.device, dtype=vec.dtype
    )
    skew_sym_mat[:, 0, 1] = -vec[:, 2]  # -z
    skew_sym_mat[:, 0, 2] = vec[:, 1]  # y
    skew_sym_mat[:, 1, 0] = vec[:, 2]  # z
    skew_sym_mat[:, 1, 2] = -vec[:, 0]  # -x
    skew_sym_mat[:, 2, 0] = -vec[:, 1]  # -y
    skew_sym_mat[:, 2, 1] = vec[:, 0]  # x

    return skew_sym_mat


@torch_jit_compile
def frame_transform_combine(
    t01: torch.Tensor,
    q01: torch.Tensor,
    t12: torch.Tensor | None = None,
    q12: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Combine two frame transformations of coordinates.

    Frame transform rotates the coordinate frame while keeping the vector
    or point fixed.

    It performs the following transformation operation:
    :math:`T_{02} = T_{01} \times T_{12}`, where :math:`T_{AB}` is the
    homogeneous transformation matrix from frame A to B.


    Args:
        t01: The translation of frame 1 w.r.t. frame 0, of shape (..., 3).
        q01: The rotation of frame 1 w.r.t. frame 0, of shape (..., 4).
        t12: The translation 2 w.r.t. frame 1.
            Defaults to None, in which case it is set to zeros.
        q12: The rotation 2 w.r.t. frame 1.
            Defaults to None, in which case it is set to zeros.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The combined translation and
            rotation of frame 2 w.r.t. frame 0. The translation is of
            shape (..., 3) and the rotation is of shape (..., 4). The return
            order is (t02, q02).

    """
    if t12 is None:
        t12 = torch.zeros_like(t01, device=t01.device)
    if q12 is None:
        q12 = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=q01.device
        ).broadcast_to(t12.shape[:-1] + (4,))

    q02 = quaternion_multiply(q01, q12)
    t02 = t01 + quaternion_apply_point(q01, t12)
    return t02, q02


@torch_jit_compile
def frame_transform_subtract(
    t01: torch.Tensor,
    q01: torch.Tensor,
    t02: torch.Tensor | None = None,
    q02: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Subtract two frame transformations of coordinates.

    It performs the following transformation operation:
    :math:`T_{12} = T_{01}^{-1} \times T_{02}`, where :math:`T_{AB}` is the
    homogeneous transformation matrix of frame B w.r.t. A.


    Args:
        t01: The translation of frame 1 w.r.t. frame 0, of shape (..., 3).
        q01: The rotation of frame 1 w.r.t. frame 0, of shape (..., 4).
        t02: The translation frame 2 w.r.t. frame 0.
            Defaults to None, in which case it is set to identity.
        q02: The rotation frame 2 w.r.t. frame 0.
            Defaults to None, in which case it is set to identity.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The combined translation and rotation
            of frame 2 w.r.t. frame 1. The translation is of shape (..., 3) and the
            rotation is of shape (..., 4). The return order is (t12, q12).

    """  # noqa: E501
    if t02 is None:
        t02 = torch.zeros_like(t01, device=t01.device)
    if q02 is None:
        q02 = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=q01.device
        ).broadcast_to(t01.shape[:-1] + (4,))

    # same to quaternion_left_division(q01, q02)
    q01_inv = quaternion_invert(q01)
    q12 = quaternion_multiply(q01_inv, q02)
    t12 = quaternion_apply_point(q01_inv, t02 - t01)

    return t12, q12


@torch_jit_compile
def pose_diff(
    ta: torch.Tensor,
    qa: torch.Tensor,
    tb: torch.Tensor,
    qb: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Compute the difference between two poses in the same frame.

    It performs the following transformation operation:
    :math:`t_{diff} = t_a - t_b`, and :math:`q_{diff} = q_a \times q_b^{-1}`,
    so that :math:`q_{diff} \times q_b = q_a`, and :math:`t_{diff} + t_b = t_a`.

    Args:
        ta: The translation of the second pose, of shape (..., 3).
        qa: The rotation of the second pose, of shape (..., 4).
        tb: The translation of the first pose, of shape (..., 3).
        qb: The rotation of the first pose, of shape (..., 4).

    Returns:
        The difference in translation and rotation between the two poses.
        The translation is of shape (..., 3) and the rotation is of
        shape (..., 4).
    """  # noqa: E501

    t_diff = ta - tb
    q_diff = quaternion_right_division(qa, qb)
    return t_diff, q_diff


DEFAULT_ACOS_BOUND: float = 1.0 - 1e-4


def acos_linear_extrapolation(
    x: torch.Tensor,
    bounds: tuple[float, float] = (-DEFAULT_ACOS_BOUND, DEFAULT_ACOS_BOUND),
) -> torch.Tensor:
    """`arccos(x)` with stable backpropagation.

    Implements `arccos(x)` which is linearly extrapolated outside `x`'s
    original domain of `(-1, 1)`. This allows for stable backpropagation
    in case `x` is not guaranteed to be strictly within `(-1, 1)`.

    More specifically::

        bounds=(lower_bound, upper_bound)
        if lower_bound <= x <= upper_bound:
            acos_linear_extrapolation(x) = acos(x)
        elif x <= lower_bound: # 1st order Taylor approximation
            acos_linear_extrapolation(x)
                = acos(lower_bound) + dacos/dx(lower_bound) * (x - lower_bound)
        else:  # x >= upper_bound
            acos_linear_extrapolation(x)
                = acos(upper_bound) + dacos/dx(upper_bound) * (x - upper_bound)

    Args:
        x: Input `Tensor`.
        bounds: A float 2-tuple defining the region for the
            linear extrapolation of `acos`.
            The first/second element of `bound`
            describes the lower/upper bound that defines the lower/upper
            extrapolation region, i.e. the region where
            `x <= bound[0]`/`bound[1] <= x`.
            Note that all elements of `bound` have to be within (-1, 1).

    Returns:
        acos_linear_extrapolation: `Tensor` containing the extrapolated
            `arccos(x)`.

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/math.py

    """

    def _acos_linear_approximation(x: torch.Tensor, x0: float) -> torch.Tensor:
        """Calculates the 1st order Taylor expansion of `arccos(x)` around `x0`."""  # noqa: E501
        return (x - x0) * _dacos_dx(x0) + math.acos(x0)

    def _dacos_dx(x: float) -> float:
        """Calculates the derivative of `arccos(x)` w.r.t. `x`."""
        return (-1.0) / math.sqrt(1.0 - x * x)

    lower_bound, upper_bound = bounds

    if lower_bound > upper_bound:
        raise ValueError(
            "lower bound has to be smaller or equal to upper bound."
        )

    if lower_bound <= -1.0 or upper_bound >= 1.0:
        raise ValueError(
            "Both lower bound and upper bound have to be within (-1, 1)."
        )

    # init an empty tensor and define the domain sets
    acos_extrap = torch.empty_like(x)
    x_upper = x >= upper_bound
    x_lower = x <= lower_bound
    x_mid = (~x_upper) & (~x_lower)

    # acos calculation for upper_bound < x < lower_bound
    acos_extrap[x_mid] = torch.acos(x[x_mid])
    # the linear extrapolation for x >= upper_bound
    acos_extrap[x_upper] = _acos_linear_approximation(x[x_upper], upper_bound)
    # the linear extrapolation for x <= lower_bound
    acos_extrap[x_lower] = _acos_linear_approximation(x[x_lower], lower_bound)

    return acos_extrap
