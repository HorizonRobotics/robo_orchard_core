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

"""Data classes for camera sensor data."""

from typing import Literal

from robo_orchard_core.datatypes.dataclass import DataClass
from robo_orchard_core.datatypes.geometry import BatchPose6D, Pose6D
from robo_orchard_core.utils.config import TorchTensor
from robo_orchard_core.utils.torch_utils import Device

__all___ = [
    "CameraData",
    "BatchCameraData",
]


class CameraData(DataClass):
    """Data class for camera sensor data."""

    topic: str | None = None
    """The topic of the camera sensor."""

    pose: Pose6D | None = None
    """The pose of the camera sensor."""

    image_shape: tuple[int, int] | None = None
    """A tuple containing (height, width) of the camera sensor."""

    intrinsic_matrix: TorchTensor | None
    """The intrinsic matrix for the camera.

    Shape is (3, 3).
    """

    sensor_data: TorchTensor
    """The sensor data from the camera.

    Shape is (H, W, C) for raw data, where C is the number of channels,
    H is the height of the image, and W is the width of the image.

    For compressed data, the shape is (N, ) where N is the number of bytes.
    """

    distortion_model: (
        Literal["plumb_bob", "rational_polynomial", "equidistant"] | None
    ) = None
    """The distortion model of the camera.

    If None, no distortion model is applied. The distortion model follows ROS2
    convention,  see:
    - http://docs.ros.org/en/api/image_geometry/html/c++/pinhole__camera__model_8cpp.html
    - http://docs.ros.org/en/rolling/p/camera_calibration/doc/index.html

    """

    distorsion_coefficients: TorchTensor | None = None
    """Distortion coefficients of the camera.

    It should be 1D tensor with 4, 5, or 8 elements depending on the
    distortion model.
    """

    pix_fmt: Literal["rgb", "bgr"] | None = None
    """Pixel format."""

    def __post_init__(self):
        if self.image_shape is None:
            if self.sensor_data.dim() == 3:
                data_shape = self.sensor_data.shape
                self.image_shape = (data_shape[0], data_shape[1])
            else:
                raise ValueError(
                    "image_shape must be provided if sensor_data is not 3D."
                )

    def get_extrinsic_matrix(self, device: Device = "cpu") -> TorchTensor:
        """Get the extrinsic matrix of the camera.

        Pose6D describes the transformation from the camera frame to the
        world frame, while the extrinsic matrix describes the transformation
        from the world frame to the camera frame, which is the inverse of
        the pose transformation (cam2world).

        The extrinsic matrix is a 4x4 matrix:

        .. code-block:: text

            [[R, t],
             [0, 1]]

        Where R is a 3x3 rotation matrix and t is a 3x1 translation vector.
        """
        assert self.pose is not None
        return (
            self.pose.as_BatchTransform3D(device=device)
            .inverse()
            .as_Transform3D_M()
            .get_matrix()[0]
        )


class BatchCameraData(DataClass):
    """Data class for batched camera sensor data."""

    topic: str | None = None
    """The topic of the camera sensor."""

    pose: BatchPose6D | None = None
    """The pose of the camera sensor."""

    image_shape: tuple[int, int] | None = None
    """A tuple containing (height, width) of the camera sensor."""

    intrinsic_matrices: TorchTensor | None = None
    """The intrinsic matrices for all camera.

    Shape is (B, 3, 3), where B is the batch size.
    """

    distortion_model: (
        Literal["plumb_bob", "rational_polynomial", "equidistant"] | None
    ) = None
    """The distortion model of the camera.

    If None, no distortion model is applied. The distortion model follows ROS2
    convention,  see:
    - http://docs.ros.org/en/api/image_geometry/html/c++/pinhole__camera__model_8cpp.html
    - http://docs.ros.org/en/rolling/p/camera_calibration/doc/index.html

    """

    distorsion_coefficients: TorchTensor | None = None
    """Distortion coefficients of the camera.

    It should be 1D tensor with 4, 5, or 8 elements depending on the
    distortion model.

    All data in the batch should have the same distortion model and
    coefficients.
    """

    sensor_data: TorchTensor
    """The sensor data from all cameras.

    Shape is (B, H, W, C) for raw data, where B is the batch size, C is the
    number of channels, H is the height of the image, and W is the width
    of the image.
    """

    pix_fmt: Literal["rgb", "bgr"] | None = None
    """Pixel format."""

    @property
    def batch_size(self) -> int:
        """Get the batch size.

        The batch size is the number of cameras in the batch.

        Returns:
            int: The batch size.
        """
        return self.sensor_data.shape[0]

    def get_extrinsic_matrix(self) -> TorchTensor:
        """Get the extrinsic matrix of the cameras.

        Pose6D describes the transformation from the camera frame to the
        world frame, while the extrinsic matrix describes the transformation
        from the world frame to the camera frame, which is the inverse of
        the pose transformation (cam2world).

        The extrinsic matrix is a Bx4x4 matrix:

        .. code-block:: text

            [[[R, t],
              [0, 1]],
              ...
             [[R, t],
              [0, 1]]]
            ]

        Where R is a 3x3 rotation matrix and t is a 3x1 translation vector.
        """

        assert self.pose is not None
        return self.pose.inverse().as_Transform3D_M().get_matrix()
