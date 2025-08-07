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

import os

import cv2
import numpy as np
import pytest
import torch

from robo_orchard_core.utils.image import (
    wrapAffine,
)
from robo_orchard_core.utils.math.transform import (
    Rotate2D,
    Scale2D,
    Transform2D_M,
    Translate2D,
)


@pytest.fixture(scope="session")
def img_lenna(workspace: str) -> torch.Tensor:
    """Fixture to load the Lenna image."""
    img_path = os.path.join(
        workspace, "robo_orchard_workspace", "imgs", "Lenna.png"
    )

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    assert isinstance(img, (np.ndarray,)), "Image not loaded correctly"
    return torch.asarray(img)


def test_lena_read(img_lenna: torch.Tensor):
    print(img_lenna.shape)  # torch.Size([500, 500, 3])


def get_affine_transform(
    center: tuple[float, float], angle: float, scale: float
) -> Transform2D_M:
    t = Translate2D([-center[0], -center[1]])
    r = Rotate2D(angle)
    s = Scale2D([scale, scale])
    return t @ r @ s @ t.inverse()


def cv_wrapAffine(
    src: torch.Tensor, mat: torch.Tensor, target_hw: tuple[int, int]
):
    assert src.shape[-3] in [1, 3], (
        "src must be a single-channel or 3-channel image"
    )
    dst = torch.zeros(size=src.shape[:-2] + target_hw, dtype=src.dtype)
    flattened_src = src.reshape(-1, src.shape[-2], src.shape[-1])
    flattened_dst = dst.reshape(-1, target_hw[0], target_hw[1])
    if mat.shape[0] != 1:
        raise KeyError(
            "mat must be a single matrix, got shape: "
            f"{mat.shape}, expected (1, 2, 3) or (1, 3, 3)"
        )

    for i in range(flattened_src.shape[0]):
        img_after_warp = cv2.warpAffine(
            src=flattened_src[i].numpy(),
            M=mat.numpy()[0, :2, :],
            dsize=(target_hw[1], target_hw[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0,),
        )
        flattened_dst[i] = torch.from_numpy(img_after_warp)

    return dst


class TestWrapAffine:
    def test_wrapAffine_cv2_consistency(self, img_lenna: torch.Tensor):
        src_img = img_lenna[:100, :200, :]
        src_hw = src_img.shape[:2]
        target_hw = (100, 200)

        # convert hwc to chw
        src_img = src_img.permute(2, 0, 1)  # torch.Size([3, 100, 200])

        ts = get_affine_transform(
            center=(src_hw[1] / 2 + 4, src_hw[0] / 2 - 10),
            angle=np.deg2rad(90),
            scale=1.5,
        )
        src_img = src_img.unsqueeze(0).to(
            dtype=torch.float32,
        )  # torch.Size([1, 3, 100, 200])

        dst_img = wrapAffine(
            src=src_img,
            M=ts.get_matrix().to(),
            target_hw=target_hw,
            align_corners=True,  # need align corners to match implementation
        )

        cv2_dst_img = cv_wrapAffine(
            src=src_img,
            mat=ts.get_matrix().to(),
            target_hw=target_hw,
        )
        assert torch.allclose(dst_img, cv2_dst_img, atol=3, rtol=0.1)
