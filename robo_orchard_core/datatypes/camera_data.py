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

import deprecated

from robo_orchard_core.datatypes.dataclass import DataClass, TensorToMixin
from robo_orchard_core.datatypes.geometry import (
    BatchFrameTransform,
    FrameTransform,
)
from robo_orchard_core.utils.config import TorchTensor
from robo_orchard_core.utils.torch_utils import Device

__all___ = [
    "Distortion",
    "CameraData",
    "BatchCameraData",
]


class Distortion(DataClass, TensorToMixin):
    model: (
        Literal["plumb_bob", "rational_polynomial", "equidistant"] | None
    ) = None
    """The distortion model of the camera.

    If None, no distortion model is applied. The distortion model follows ROS2
    convention,  see:
    - http://docs.ros.org/en/api/image_geometry/html/c++/pinhole__camera__model_8cpp.html
    - http://docs.ros.org/en/rolling/p/camera_calibration/doc/index.html

    """

    coefficients: TorchTensor | None = None
    """Distortion coefficients of the camera.

    It should be 1D tensor with 4, 5, or 8 elements depending on the
    distortion model.
    """


@deprecated.deprecated(
    version="0.2.0",
    reason="CameraData will be replaced by BatchCameraData for "
    "simplicity and efficiency. ",
)
class CameraData(DataClass, TensorToMixin):
    """Data class for camera sensor data."""

    topic: str | None = None
    """The topic of the camera sensor."""

    frame_id: str | None = None
    """Coordinate frame ID for the camera sensor.

    This is NOT the frame number or index of the image frame!

    frame_id is typically used to identify the coordinate frame in which the
    camera data is expressed. It can be useful for visualization or
    transformation purposes.
    If not provided, it defaults to None.
    """

    pose: FrameTransform | None = None
    """The pose of the camera sensor.

    This is also known as the extrinsic matrix of the camera.
    """

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

    distortion: Distortion | None = None

    pix_fmt: Literal["rgb", "bgr", "gray", "depth"] | None = None
    """Pixel format."""

    @property
    def distortion_coefficients(self) -> TorchTensor | None:
        """Get the distortion coefficients of the camera.

        Returns:
            TorchTensor | None: The distortion coefficients of the camera.
            If no distortion model is applied, returns None.
        """
        if self.distortion is not None:
            return self.distortion.coefficients
        return None

    @property
    def distortion_model(
        self,
    ) -> Literal["plumb_bob", "rational_polynomial", "equidistant"] | None:
        """Get the distortion model of the camera.

        Returns:
            Literal["plumb_bob", "rational_polynomial", "equidistant"] | None:
            The distortion model of the camera. If no distortion model is
            applied, returns None.
        """
        if self.distortion is not None:
            return self.distortion.model
        return None

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


class BatchCameraData(DataClass, TensorToMixin):
    """Data class for batched camera sensor data.

    A batch of camera data shares the same image shape, distortion model.
    The intrinsic matrices and extrinsic matrices (pose) of the cameras
    can be different.
    """

    topic: str | None = None
    """The topic of the camera sensor."""

    frame_id: str | None = None
    """Coordinate frame ID for the camera sensor.

    This is NOT the frame number or index of the image frame!

    frame_id is typically used to identify the coordinate frame in which the
    camera data is expressed. It can be useful for visualization or
    transformation purposes.
    If not provided, it defaults to None.
    """

    image_shape: tuple[int, int] | None = None
    """A tuple containing (height, width) of the camera sensor."""

    intrinsic_matrices: TorchTensor | None = None
    """The intrinsic matrices for all camera.

    Shape is (B, 3, 3), where B is the batch size.
    """

    distortion: Distortion | None = None

    # pose: BatchPose6D | None = None
    # """The pose of the camera sensor."""

    pose: BatchFrameTransform | None = None
    """Frame transform of the camera sensor.

    This is also known as the extrinsic matrix of the camera.
    """

    sensor_data: TorchTensor
    """The sensor data from all cameras.

    Shape is (B, H, W, C) for raw data, where B is the batch size, C is the
    number of channels, H is the height of the image, and W is the width
    of the image.
    """

    pix_fmt: Literal["rgb", "bgr", "gray", "depth"] | None = None
    """Pixel format."""

    @property
    def distorsion_coefficients(self) -> TorchTensor | None:
        """Get the distortion coefficients of the camera.

        Returns:
            TorchTensor | None: The distortion coefficients of the camera.
            If no distortion model is applied, returns None.
        """
        if self.distortion is not None:
            return self.distortion.coefficients
        return None

    @property
    def distortion_model(
        self,
    ) -> Literal["plumb_bob", "rational_polynomial", "equidistant"] | None:
        """Get the distortion model of the camera.

        Returns:
            Literal["plumb_bob", "rational_polynomial", "equidistant"] | None:
            The distortion model of the camera. If no distortion model is
            applied, returns None.
        """
        if self.distortion is not None:
            return self.distortion.model
        return None

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
