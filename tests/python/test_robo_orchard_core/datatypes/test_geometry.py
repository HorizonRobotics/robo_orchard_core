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

import pytest
import torch

from robo_orchard_core.datatypes.geometry import (
    BatchTransform3D,
    FrameTransform,
    Pose,
    Transform3D,
)
from robo_orchard_core.utils.math import math_utils


class TestTensorToMixin:
    @pytest.mark.parametrize(
        "device",
        [
            pytest.param("cpu"),
            pytest.param(
                "cuda:0",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA NOT AVAILABLE"
                ),
            ),
        ],
    )
    def test_tensor_to_mixin(self, device):
        batch_size = 100
        q = math_utils.normalize(
            torch.rand(size=(batch_size, 4), device=device) - 0.5, dim=-1
        )
        q = math_utils.quaternion_standardize(q)
        t = torch.rand(size=(batch_size, 3), device=device) - 0.5
        batch_transform = BatchTransform3D(xyz=t, quat=q)

        target_devices = [torch.device("cpu"), torch.device("cuda:0")]
        target_dtypes = [torch.float32, torch.float64]
        for target_device in target_devices:
            for target_dtype in target_dtypes:
                # Convert to target device and dtype
                batch_transform = batch_transform.to(
                    device=target_device, dtype=target_dtype
                )
                assert batch_transform.xyz.device == target_device
                assert batch_transform.quat.device == target_device
                assert batch_transform.xyz.dtype == target_dtype
                assert batch_transform.quat.dtype == target_dtype


class TestTransform3D:
    @pytest.mark.parametrize(
        "device",
        [
            pytest.param("cpu"),
            pytest.param(
                "cuda:0",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA NOT AVAILABLE"
                ),
            ),
        ],
    )
    def test_repeat(self, device):
        q = math_utils.normalize(
            torch.rand(size=(4,), device=device) - 0.5, dim=-1
        )
        q = math_utils.quaternion_standardize(q)
        t = torch.rand(size=(3,), device=device) - 0.5
        transform = Transform3D(xyz=t, quat=q)
        # Test repeat

        target_devices = [torch.device("cpu"), torch.device("cuda:0")]
        for target_device in target_devices:
            batch_size = 3
            repeated_transform = transform.repeat(
                batch_size, device=target_device
            )
            assert repeated_transform.xyz.shape == (batch_size, 3)
            assert repeated_transform.quat.shape == (batch_size, 4)
            assert repeated_transform.xyz.device == target_device
            assert repeated_transform.quat.device == target_device

            # Check the values
            xyz_target_device = torch.tensor(
                transform.xyz, device=target_device
            )
            quat_target_device = torch.tensor(
                transform.quat, device=target_device
            )

            for i in range(batch_size):
                assert torch.allclose(
                    repeated_transform.xyz[i],
                    xyz_target_device,
                    atol=1e-5,
                )
                assert torch.allclose(
                    repeated_transform.quat[i],
                    quat_target_device,
                    atol=1e-5,
                )


class TestFrameTransform:
    @pytest.mark.parametrize(
        "device",
        [
            pytest.param("cpu"),
            pytest.param(
                "cuda:0",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA NOT AVAILABLE"
                ),
            ),
        ],
    )
    def test_repeat(self, device):
        q = math_utils.normalize(
            torch.rand(size=(4,), device=device) - 0.5, dim=-1
        )
        q = math_utils.quaternion_standardize(q)
        t = torch.rand(size=(3,), device=device) - 0.5
        transform = FrameTransform(
            xyz=t, quat=q, parent_frame_id="parent", child_frame_id="child"
        )
        target_devices = [torch.device("cpu"), torch.device("cuda:0")]
        for target_device in target_devices:
            batch_size = 3
            repeated_transform = transform.repeat(
                batch_size, device=target_device
            )
            assert repeated_transform.xyz.shape == (batch_size, 3)
            assert repeated_transform.quat.shape == (batch_size, 4)
            assert repeated_transform.xyz.device == target_device
            assert repeated_transform.quat.device == target_device
            assert (
                repeated_transform.parent_frame_id == transform.parent_frame_id
            )
            assert (
                repeated_transform.child_frame_id == transform.child_frame_id
            )

            # Check the values
            xyz_target_device = torch.tensor(
                transform.xyz, device=target_device
            )
            quat_target_device = torch.tensor(
                transform.quat, device=target_device
            )

            for i in range(batch_size):
                assert torch.allclose(
                    repeated_transform.xyz[i],
                    xyz_target_device,
                    atol=1e-5,
                )
                assert torch.allclose(
                    repeated_transform.quat[i],
                    quat_target_device,
                    atol=1e-5,
                )


class TestPose:
    @pytest.mark.parametrize(
        "device",
        [
            pytest.param("cpu"),
            pytest.param(
                "cuda:0",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA NOT AVAILABLE"
                ),
            ),
        ],
    )
    def test_repeat(self, device):
        q = math_utils.normalize(
            torch.rand(size=(4,), device=device) - 0.5, dim=-1
        )
        q = math_utils.quaternion_standardize(q)
        t = torch.rand(size=(3,), device=device) - 0.5
        pose = Pose(xyz=t, quat=q, frame_id="pose_frame")
        target_devices = [torch.device("cpu"), torch.device("cuda:0")]
        for target_device in target_devices:
            batch_size = 3
            repeated_pose = pose.repeat(batch_size, device=target_device)
            assert repeated_pose.xyz.shape == (batch_size, 3)
            assert repeated_pose.quat.shape == (batch_size, 4)
            assert repeated_pose.xyz.device == target_device
            assert repeated_pose.quat.device == target_device
            assert repeated_pose.frame_id == pose.frame_id

            # Check the values
            xyz_target_device = torch.tensor(pose.xyz, device=target_device)
            quat_target_device = torch.tensor(pose.quat, device=target_device)

            for i in range(batch_size):
                assert torch.allclose(
                    repeated_pose.xyz[i],
                    xyz_target_device,
                    atol=1e-5,
                )
                assert torch.allclose(
                    repeated_pose.quat[i],
                    quat_target_device,
                    atol=1e-5,
                )


class TestBatchTransform3D:
    @pytest.mark.parametrize(
        "device",
        [
            pytest.param("cpu"),
            pytest.param(
                "cuda:0",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA NOT AVAILABLE"
                ),
            ),
        ],
    )
    def test_apply_point_consistency(self, device):
        batch_size = 100
        q = math_utils.normalize(
            torch.rand(size=(batch_size, 4), device=device) - 0.5, dim=-1
        )
        q = math_utils.quaternion_standardize(q)
        t = torch.rand(size=(batch_size, 3), device=device) - 0.5
        p = torch.rand(size=(batch_size, 1000, 3), device=device) - 0.5
        batch_transform = BatchTransform3D(xyz=t, quat=q)
        batch_transform_M = batch_transform.as_Transform3D_M()
        transformed_p = batch_transform.transform_points(p)
        transformed_p_ = batch_transform_M.transform_points(p)
        assert torch.allclose(transformed_p, transformed_p_, atol=1e-5)

    @pytest.mark.parametrize(
        "device",
        [
            pytest.param("cpu"),
            pytest.param(
                "cuda:0",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA NOT AVAILABLE"
                ),
            ),
        ],
    )
    def test_compose_order_consistency(self, device):
        q = math_utils.normalize(
            torch.rand(size=(3, 4), device=device) - 0.5, dim=-1
        )
        q_23, q_12, q_01 = math_utils.quaternion_standardize(q).unbind(0)
        t = torch.rand(size=(3, 3), device=device) - 0.5
        t_23, t_12, t_01 = t.unbind(0)
        batch_transform_23 = BatchTransform3D(xyz=t_23, quat=q_23)
        batch_transform_12 = BatchTransform3D(xyz=t_12, quat=q_12)
        batch_transform_01 = BatchTransform3D(xyz=t_01, quat=q_01)
        batch_transform_03_1 = batch_transform_23.compose(
            batch_transform_12, batch_transform_01
        )
        batch_transform_03_2 = batch_transform_23.compose(
            batch_transform_12
        ).compose(batch_transform_01)
        batch_transform_03_3 = batch_transform_23.compose(
            batch_transform_12.compose(batch_transform_01)
        )

        assert torch.allclose(
            batch_transform_03_1.quat, batch_transform_03_2.quat, atol=1e-5
        )
        assert torch.allclose(
            batch_transform_03_1.xyz, batch_transform_03_2.xyz, atol=1e-5
        )
        assert torch.allclose(
            batch_transform_03_1.quat, batch_transform_03_3.quat, atol=1e-5
        )
        assert torch.allclose(
            batch_transform_03_1.xyz, batch_transform_03_3.xyz, atol=1e-5
        )

    @pytest.mark.parametrize(
        "device",
        [
            pytest.param("cpu"),
            pytest.param(
                "cuda:0",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA NOT AVAILABLE"
                ),
            ),
        ],
    )
    def test_compose_consistency_with_matrix(self, device):
        q = math_utils.normalize(
            torch.rand(size=(3, 4), device=device) - 0.5, dim=-1
        )
        q_23, q_12, q_01 = math_utils.quaternion_standardize(q).unbind(0)
        t = torch.rand(size=(3, 3), device=device) - 0.5
        t_23, t_12, t_01 = t.unbind(0)
        batch_transform_23 = BatchTransform3D(xyz=t_23, quat=q_23)
        batch_transform_12 = BatchTransform3D(xyz=t_12, quat=q_12)
        batch_transform_01 = BatchTransform3D(xyz=t_01, quat=q_01)
        batch_transform_03_1 = batch_transform_23.compose(
            batch_transform_12, batch_transform_01
        )
        batch_transform_23_M = batch_transform_23.as_Transform3D_M()
        batch_transform_12_M = batch_transform_12.as_Transform3D_M()
        batch_transform_01_M = batch_transform_01.as_Transform3D_M()
        batch_transform_03_1_M = batch_transform_23_M.compose(
            batch_transform_12_M, batch_transform_01_M
        )
        assert torch.allclose(
            batch_transform_03_1.quat,
            batch_transform_03_1_M.get_rotation_quaternion(),
            atol=1e-5,
        )
        assert torch.allclose(
            batch_transform_03_1.xyz,
            batch_transform_03_1_M.get_translation(),
            atol=1e-5,
        )

    @pytest.mark.parametrize(
        "device",
        [
            pytest.param("cpu"),
            pytest.param(
                "cuda:0",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA NOT AVAILABLE"
                ),
            ),
        ],
    )
    def test_substract_consistency(self, device):
        q = math_utils.normalize(
            torch.rand(size=(2, 4), device=device) - 0.5, dim=-1
        )
        q_02, q_01 = math_utils.quaternion_standardize(q).unbind(0)
        t = torch.rand(size=(2, 3), device=device) - 0.5
        t_02, t_01 = t.unbind(0)
        batch_transform_02 = BatchTransform3D(xyz=t_02, quat=q_02)
        batch_transform_01 = BatchTransform3D(xyz=t_01, quat=q_01)
        batch_transform_12 = batch_transform_02.subtract(batch_transform_01)
        batch_transform_12_ = batch_transform_02.compose(
            batch_transform_01.inverse()
        )

        assert torch.allclose(
            batch_transform_12.quat, batch_transform_12_.quat, atol=1e-5
        )
        assert torch.allclose(
            batch_transform_12.xyz, batch_transform_12_.xyz, atol=1e-5
        )

        batch_transform_02_ = batch_transform_12.compose(batch_transform_01)

        assert torch.allclose(
            batch_transform_02.quat, batch_transform_02_.quat, atol=1e-5
        )
        assert torch.allclose(
            batch_transform_02.xyz, batch_transform_02_.xyz, atol=1e-5
        )

    @pytest.mark.parametrize(
        "device",
        [
            pytest.param("cpu"),
            pytest.param(
                "cuda:0",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA NOT AVAILABLE"
                ),
            ),
        ],
    )
    def test_inverse(self, device):
        batch_size = 100
        q = math_utils.normalize(
            torch.rand(size=(batch_size, 4), device=device) - 0.5, dim=-1
        )
        q = math_utils.quaternion_standardize(q)
        t = torch.rand(size=(batch_size, 3), device=device) - 0.5
        batch_transform = BatchTransform3D(xyz=t, quat=q)
        batch_transform_m = batch_transform.as_Transform3D_M()
        inv_batch_transform_m = batch_transform_m.inverse(True)
        inv_batch_transform = batch_transform.inverse()
        assert torch.allclose(
            inv_batch_transform.xyz,
            inv_batch_transform_m.get_translation(),
            atol=1e-5,
        )
        assert torch.allclose(
            inv_batch_transform.quat,
            inv_batch_transform_m.get_rotation_quaternion(),
            atol=1e-5,
        )


if __name__ == "__main__":
    pytest.main(["-s", __file__])
