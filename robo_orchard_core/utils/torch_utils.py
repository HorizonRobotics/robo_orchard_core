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


"""Helper functions for working with PyTorch tensors."""

import functools

import torch

Device = str | torch.device


def get_device(x, device: Device | None = None) -> torch.device:
    """Get the device of the specified variable x if it is a tensor.

    Gets the device of the specified variable x if it is a tensor, or
    falls back to a default CPU device otherwise. Allows overriding by
    providing an explicit device.

    Args:
        x: a torch.Tensor to get the device from or another type
        device: Device (as str or torch.device) to fall back to

    Returns:
        A matching torch.device object

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/common/datatypes.py
    """

    # User overrides device
    if device is not None:
        return make_device(device)

    # Set device based on input tensor
    if torch.is_tensor(x):
        return x.device

    # Default device is cpu
    return torch.device("cpu")


def make_device(device: Device) -> torch.device:
    """Make a torch.device object from a string or torch.device object.

    Makes an actual torch.device object from the device specified as
    either a string or torch.device object. If the device is `cuda` without
    a specific index, the index of the current device is assigned.

    Args:
        device: Device (as str or torch.device)

    Returns:
        A matching torch.device object

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/common/datatypes.py#L17
    """
    device = torch.device(device) if isinstance(device, str) else device
    if device.type == "cuda" and device.index is None:
        # If cuda but with no index, then the current cuda device is indicated.
        # In that case, we fix to that device
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    return device


def format_tensor(
    input,
    dtype: torch.dtype = torch.float32,
    device: Device = "cpu",
) -> torch.Tensor:
    """Helper function for converting a scalar value to a tensor.

    Args:
        input: Python scalar, Python list/tuple, torch scalar, 1D torch tensor
        dtype: data type for the input
        device: Device (as str or torch.device) on which the tensor should
            be placed.

    Returns:
        input_vec: torch tensor with optional added batch dimension.

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/renderer/utils.py#L284

    """
    device_ = make_device(device)
    if not torch.is_tensor(input):
        input = torch.tensor(input, dtype=dtype, device=device_)

    if input.dim() == 0:
        input = input.view(1)

    if input.device == device_:
        return input

    input = input.to(device=device)
    return input


def convert_to_tensors_and_broadcast(
    *args,
    dtype: torch.dtype = torch.float32,
    device: Device = "cpu",
):
    """Convert inputs to tensors and broadcast them.

    Helper function to handle parsing an arbitrary number of inputs (*args)
    which all need to have the same batch dimension.
    The output is a list of tensors.

    Args:
        *args: an arbitrary number of inputs
            Each of the values in `args` can be one of the following
                - Python scalar
                - Torch scalar
                - Torch tensor of shape (N, K_i) or (1, K_i) where K_i are
                  an arbitrary number of dimensions which can vary for each
                  value in args. In this case each input is broadcast to a
                  tensor of shape (N, K_i)
        dtype: data type to use when creating new tensors.
        device: torch device on which the tensors should be placed.

    Output:
        args: A list of tensors of shape (N, K_i)

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/renderer/utils.py#L314
    """
    # Convert all inputs to tensors with a batch dimension
    args_1d = [format_tensor(c, dtype, device) for c in args]

    # Find broadcast size
    sizes = [c.shape[0] for c in args_1d]
    N = max(sizes)

    args_Nd = []
    for c in args_1d:
        if c.shape[0] != 1 and c.shape[0] != N:
            msg = "Got non-broadcastable sizes %r" % sizes
            raise ValueError(msg)

        # Expand broadcast dim and keep non broadcast dims the same size
        expand_sizes = (N,) + (-1,) * len(c.shape[1:])
        args_Nd.append(c.expand(*expand_sizes))

    return args_Nd


@functools.lru_cache
def dtype_str2torch(dtype_str: str) -> torch.dtype:
    """Convert a string representation of a dtype to a torch.dtype."""

    if dtype_str == "float32":
        return torch.float32
    elif dtype_str == "float64":
        return torch.float64
    elif dtype_str == "int32":
        return torch.int32
    elif dtype_str == "int64":
        return torch.int64
    elif dtype_str == "uint8":
        return torch.uint8
    elif dtype_str == "int8":
        return torch.int8
    elif dtype_str == "int16":
        return torch.int16
    elif dtype_str == "bool":
        return torch.bool
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
