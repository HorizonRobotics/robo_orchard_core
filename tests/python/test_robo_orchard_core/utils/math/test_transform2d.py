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

from typing import Literal

import cv2
import numpy as np
import pytest
import torch

from robo_orchard_core.utils.math import transform2d


class TestRotate2D:
    @pytest.mark.parametrize("axis", ["Z", "-Z"])
    def test_rotate_2d_consistency_with_cv2(self, axis: Literal["Z", "-Z"]):
        # Create a rotation transform using the custom Rotate2D class
        batch = 10
        angle = (
            (torch.randn(batch) - 0.5) * 3.14 * 2
        )  # Random angle in radians
        rotate_transform = (
            transform2d.Rotate2D(angle, axis=axis).get_matrix().numpy()
        )

        for i, m in enumerate(rotate_transform):
            a = np.rad2deg(float(angle[i]))
            if axis == "Z":
                a *= -1
            cv_m = cv2.getRotationMatrix2D(
                center=(0, 0),
                angle=a,
                scale=1.0,
            )
            assert np.allclose(m[:2], cv_m, atol=1e-6), (
                f"Rotation matrix mismatch at index {i}"
            )

    @pytest.mark.parametrize("axis", ["Z", "-Z"])
    def test_inv(self, axis: Literal["Z", "-Z"]):
        # Create a rotation transform using the custom Rotate2D class
        batch = 10
        angle = (
            (torch.randn(batch) - 0.5) * 3.14 * 2
        )  # Random angle in radians
        rotate_transform = transform2d.Rotate2D(angle, axis=axis)
        inv_transform = rotate_transform.inverse()
        inv_matrix = inv_transform.get_matrix().numpy()
        rot_transform_inv = np.linalg.inv(
            rotate_transform.get_matrix().numpy()
        )
        assert np.allclose(inv_matrix, rot_transform_inv, atol=1e-6), (
            "Inverse rotation matrix does not match the expected inverse."
        )


class TestShear2D:
    @pytest.mark.parametrize(
        "shear_factors",
        [
            (torch.randn(10, 2) - 0.5) * 10,
        ],
    )
    def test_inv(self, shear_factors: torch.Tensor):
        # Create a shear transform using the custom Shear2D class
        shear_transform = transform2d.Shear2D(shear_factors)
        inv_transform = shear_transform.inverse()
        inv_matrix = inv_transform.get_matrix().numpy()
        shear_transform_inv = np.linalg.inv(
            shear_transform.get_matrix().numpy()
        )
        assert np.allclose(inv_matrix, shear_transform_inv, atol=1e-6), (
            "Inverse shear matrix does not match the expected inverse."
        )
