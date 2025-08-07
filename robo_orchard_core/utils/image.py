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

from enum import Enum
from typing import Literal

import torch

from robo_orchard_core.utils.config import TorchTensor

__all__ = [
    "ImageChannelLayout",
    "guess_channel_layout",
    "get_image_shape",
    "affine_mat2non_align_corners",
    "convert_grid_sample_theta",
    "wrapAffine",
]


class ImageChannelLayout(str, Enum):
    """Enum for channel layout in camera data."""

    HWC = "HWC"
    """Height, Width, Channel format."""
    CHW = "CHW"
    """Channel, Height, Width format."""


def guess_channel_layout(data: TorchTensor) -> ImageChannelLayout:
    """Guess the channel dimension of the camera data."""

    possible_channels = (1, 3, 4)
    if data.ndim == 2:
        raise ValueError(
            "Data must be at least 3D, got 2D tensor. "
            "If you want to use 2D data, please use BatchCameraDataEncoded."
        )
    match_hwc = False
    match_chw = False
    if data.shape[-1] in possible_channels:
        match_hwc = True
    if data.shape[-3] in possible_channels:
        match_chw = True

    # Ambiguous case: both HWC and CHW are possible
    if match_hwc and match_chw:
        # If both HWC and CHW are possible, we cannot guess the channel layout
        raise ValueError(
            "Data has ambiguous channel dimension, "
            "please specify the channel layout explicitly."
        )
    if match_hwc:
        return ImageChannelLayout.HWC
    elif match_chw:
        return ImageChannelLayout.CHW

    raise ValueError(
        "Data has no channel dimension, "
        "please specify the channel layout explicitly."
    )


def get_image_shape(
    data: TorchTensor,
    channel_layout: ImageChannelLayout | None = None,
) -> tuple[int, int]:
    """Get the image shape(h,w) from the camera data."""
    if channel_layout is None:
        channel_layout = guess_channel_layout(data)
    if channel_layout == ImageChannelLayout.HWC:
        return data.shape[-3], data.shape[-2]
    elif channel_layout == ImageChannelLayout.CHW:
        return data.shape[-2], data.shape[-1]
    else:
        raise ValueError(f"Unknown channel layout: {channel_layout}. ")


def affine_mat2non_align_corners(
    mat: TorchTensor,
    src_hw: tuple[int, int],
    dst_hw: tuple[int, int],
) -> TorchTensor:
    """Convert affine matrix to non-align-corners affine matrix."""

    theta = convert_grid_sample_theta(
        mat=mat,
        src_hw=src_hw,
        dst_hw=dst_hw,
        to_theta=True,
        align_corners=False,
    )
    return convert_grid_sample_theta(
        mat=theta,
        src_hw=src_hw,
        dst_hw=dst_hw,
        to_theta=False,
        align_corners=True,
    )


def convert_grid_sample_theta(
    mat: TorchTensor,
    src_hw: tuple[int, int],
    dst_hw: tuple[int, int],
    to_theta: bool,
    align_corners: bool = False,
) -> TorchTensor:
    """Convert a affine matrix to theta for grid_sample, or inverse.

    Args:
        mat (TorchTensor): The affine matrix, or grid sample theta.
        src_hw (tuple[int, int]): The source height and width.
        dst_hw (tuple[int, int]): The destination height and width.
        to_theta (bool): If True, convert mat to theta. If False,
            convert theta to mat.
        align_corners (bool, optional): If True, align corners of the image.
            Defaults to False.

    Returns:
        TorchTensor: The converted theta or affine matrix.


    """
    if mat.shape[-2:] != (3, 3):
        raise ValueError(
            f"mat must be a 3x3 matrix, got shape {mat.shape[-2:]}"
        )
    dtype = mat.dtype

    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    if align_corners:
        left = torch.asarray(
            [
                [2.0 / (src_w - 1), 0, -1],
                [0, 2.0 / (src_h - 1), -1],
                [0, 0, 1],
            ],
            dtype=dtype,
            device=mat.device,
        )
        right = torch.asarray(
            [
                [2.0 / (dst_w - 1), 0, -1],
                [0, 2.0 / (dst_h - 1), -1],
                [0, 0, 1],
            ],
            dtype=dtype,
            device=mat.device,
        )
    else:
        # # If align_corners is False, we use the following transformation
        left = torch.asarray(
            [
                [(2 - 2.0 / src_w) / (src_w - 1), 0, -1 + 1.0 / src_w],
                [0, (2 - 2.0 / src_h) / (src_h - 1), -1 + 1.0 / src_h],
                [0, 0, 1],
            ],
            dtype=dtype,
            device=mat.device,
        )
        right = torch.asarray(
            [
                [(2 - 2.0 / dst_w) / (dst_w - 1), 0, -1 + 1.0 / dst_w],
                [0, (2 - 2.0 / dst_h) / (dst_h - 1), -1 + 1.0 / dst_h],
                [0, 0, 1],
            ],
            dtype=dtype,
            device=mat.device,
        )

    if to_theta:
        return left @ mat.inverse() @ right.inverse()
    else:
        return right.inverse() @ mat.inverse() @ left


def wrapAffine(
    src: TorchTensor,
    M: TorchTensor,
    target_hw: tuple[int, int],
    interpolation: Literal["bilinear", "nearest", "bicubic"] = "bilinear",
    padding_mode: Literal["zeros", "border", "reflection"] = "zeros",
    align_corners: bool = False,
) -> TorchTensor:
    """Wrap the affine transformation to the image.

    This function is similar to cv2.warpAffine, but uses PyTorch's grid_sample
    for the transformation.


    Args:
        src (TorchTensor): The source image tensor of shape (H, W, C) or
            (B, H, W, C), where B is the batch size, H is the height,
            W is the width, and C is the number of channels.
        M (TorchTensor): The affine transformation matrix of shape (2, 3) or
            (3, 3). If it is (2, 3), it will be converted to (3, 3) by adding
            a row [0, 0, 1].
        target_hw (tuple[int, int]): The desired output size (height, width).
        interpolation (str, optional): The interpolation method to use.
            Defaults to "bilinear".
        padding_mode (str, optional): The padding mode to use.
            Defaults to "zeros".
        align_corners (bool, optional): If True, align corners of the image.
            Defaults to False.

    """

    if M.shape[-1] != 3 or M.shape[-2] not in [2, 3]:
        raise ValueError(
            f"M must be a 2x3 or 3x3 matrix, got shape {M.shape[-2:]}"
        )

    if M.shape[-2] != 3:
        mat = torch.zeros(
            src.shape[:-2] + (3, 3), dtype=src.dtype, device=src.device
        )
        mat[..., :2, :3] = M[..., :2, :3]  # Copy the affine part
    else:
        mat = M

    if src.ndim not in (3, 4):
        raise ValueError(
            f"src must be a tensor of shape (H, W, C) or (B, H, W, C), "
            f"got {src.ndim} dimensions."
        )

    if src.ndim == 3:
        src = src.unsqueeze(0)  # Add batch dimension if not present

    src_layout = guess_channel_layout(src)
    if src_layout != ImageChannelLayout.CHW:
        raise ValueError(
            f"src must be in CHW format, got {src_layout}. "
            "Please convert it to CHW format before using wrapAffine."
        )
    src_hw = get_image_shape(src, src_layout)
    theta = convert_grid_sample_theta(
        mat=mat,
        src_hw=src_hw,
        dst_hw=target_hw,
        to_theta=True,
        align_corners=True,
    )[..., :2, :3]
    grid = torch.nn.functional.affine_grid(
        theta=theta,
        size=list(src.shape[:-2]) + [target_hw[0], target_hw[1]],
        align_corners=align_corners,
    )
    print(
        "grid target size: ",
        list(src.shape[:-2]) + [target_hw[0], target_hw[1]],
    )
    print("src.shape: ", src.shape)
    print("grid.shape: ", grid.shape)

    return torch.nn.functional.grid_sample(
        input=src,
        grid=grid,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
