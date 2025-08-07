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

from __future__ import annotations
import copy
from typing import Callable, Literal, Sequence

import deprecated
import torch
from typing_extensions import Self

from robo_orchard_core.datatypes.dataclass import DataClass, TensorToMixin
from robo_orchard_core.datatypes.geometry import (
    BatchFrameTransform,
    FrameTransform,
)
from robo_orchard_core.datatypes.timestamps import concat_timestamps
from robo_orchard_core.utils.config import TorchTensor
from robo_orchard_core.utils.image import (
    ImageChannelLayout,
    affine_mat2non_align_corners,
    get_image_shape,
    guess_channel_layout,
)
from robo_orchard_core.utils.math.transform import Transform2D_M
from robo_orchard_core.utils.torch_utils import Device

__all___ = [
    "ImageChannelLayout",
    "Distortion",
    "CameraData",
    "BatchCameraInfo",
    "BatchCameraData",
    "BatchCameraDataEncoded",
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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Distortion):
            return NotImplemented
        # if coefficients is None for one of the distortion models, we cannot
        # compare the coefficients
        ret = self.model == other.model
        # case: only one of the distortion models has coefficients
        if [self.coefficients, other.coefficients].count(None) == 1:
            return False
        # case: both distortion models have coefficients
        if self.coefficients is not None and other.coefficients is not None:
            ret &= torch.equal(self.coefficients, other.coefficients)
        # case: both distortion models have no coefficients just ignore
        return ret

    def copy(self) -> Distortion:
        """Copy the distortion model.

        Returns:
            Distortion: A copy of the distortion model.
        """
        return Distortion(
            model=self.model,
            coefficients=self.coefficients.clone()
            if self.coefficients is not None
            else None,
        )


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
        the pose transformation (cam w.r.t world).

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


class BatchCameraInfo(DataClass, TensorToMixin):
    """Data class for batched camera sensor data information.

    A batch of camera data shares the same image shape, distortion model.
    The intrinsic matrices and TransformFrame(Pose, also known as the inverse
    extrinsic matrix) of the cameras can be different.
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

    pose: BatchFrameTransform | None = None
    """Frame transform of the camera sensor.

    Usually the frame transform are camera framew.r.t. the world frame,
    or known as the cam2World transformation, they are the same.

    The inverse of the pose transformation is the extrinsic matrix
    (world w.r.t cam), which are widely used in computer vision.
    """

    @property
    def extrinsics(self) -> TorchTensor | None:
        """Get the extrinsic matrices of the cameras.

        Returns:
            TorchTensor | None: The extrinsic matrices of the cameras.
            If no pose is provided, returns None.
        """
        if self.pose is not None:
            return self.pose.inverse().as_Transform3D_M().get_matrix()
        return None

    def __post_init__(self):
        if self.intrinsic_matrices is not None and self.pose is not None:
            if self.intrinsic_matrices.shape[0] != self.pose.batch_size:
                raise ValueError(
                    "The batch size of intrinsic matrices must match the "
                    "batch size of pose. "
                    f"Expected {self.pose.batch_size}, got "
                    f"{self.intrinsic_matrices.shape[0]}."
                )

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

    def get_extrinsic_matrix(self) -> TorchTensor:
        """Get the extrinsic matrix of the cameras.

        Pose6D describes the transformation from the camera frame to the
        world frame, while the extrinsic matrix describes the transformation
        from the world frame to the camera frame, which is the inverse of
        the pose transformation (cam w.r.t world).

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

    def concat(self, others: Sequence[Self]) -> Self:
        """Concatenate two BatchCameraInfo objects.

        Args:
            others: Sequence[BatchCameraInfo]: The other BatchCameraInfo
                objects to concatenate with.

        Returns:
            BatchCameraInfo: The concatenated BatchCameraInfo object.
        """
        for topic in [other.topic for other in others]:
            if topic != self.topic:
                raise ValueError(
                    "All BatchCameraInfo objects must have the same topic."
                )
        for frame_id in [other.frame_id for other in others]:
            if frame_id != self.frame_id:
                raise ValueError(
                    "All BatchCameraInfo objects must have the same frame_id."
                )
        for image_shape in [other.image_shape for other in others]:
            if image_shape != self.image_shape:
                raise ValueError(
                    "All BatchCameraInfo objects must have the same image shape."  # noqa: E501
                )
        for distortion in [other.distortion for other in others]:
            if distortion != self.distortion:
                raise ValueError(
                    "All BatchCameraInfo objects must have the same distortion."  # noqa: E501
                )

        for intrinsic_matrix in [other.intrinsic_matrices for other in others]:
            if [intrinsic_matrix, self.intrinsic_matrices].count(None) == 1:
                raise ValueError(
                    "All BatchCameraInfo objects must have the same "
                    "intrinsic matrix type. "
                )
        intrinsic_matrices = (
            torch.cat(
                [self.intrinsic_matrices]
                + [other.intrinsic_matrices for other in others],  # type: ignore
                dim=0,
            )
            if self.intrinsic_matrices is not None
            else None
        )
        pose = (
            self.pose.concat([other.pose for other in others])  # type: ignore
            if self.pose is not None
            else None
        )

        return type(self)(
            topic=self.topic,
            frame_id=self.frame_id,
            image_shape=copy.copy(self.image_shape),
            intrinsic_matrices=intrinsic_matrices,
            distortion=self.distortion.copy() if self.distortion else None,
            pose=pose,
        )


class BatchCameraData(BatchCameraInfo):
    """Data class for batched camera sensor data.

    BatchCameraData extends BatchCameraDataInfo to include the actual
    sensor data in tensor format.

    A batch of camera data shares the same image shape, distortion model.
    The intrinsic matrices and extrinsic matrices (pose) of the cameras
    can be different.

    """

    sensor_data: TorchTensor
    """The sensor data from all cameras.

    Shape is (B, H, W, C) for raw data, where B is the batch size, C is the
    number of channels, H is the height of the image, and W is the width
    of the image. For traning, the shape is usually (B, C, H, W).

    Note that sensor_data does not restrict the channel layout, it can be
    in HWC or CHW format.


    """

    pix_fmt: Literal["rgb", "bgr", "gray", "depth"] | None = None
    """Pixel format."""

    timestamps: list[int] | None = None
    """Timestamps of the camera data in nanoseconds(1e-9 seconds)."""

    def __post_init__(self):
        super().__post_init__()
        if (
            self.timestamps is not None
            and len(self.timestamps) != self.batch_size
        ):
            raise ValueError(
                "The length of timestamps must match the batch size. "
                f"Expected {self.batch_size}, got {len(self.timestamps)}."
            )
        if self.pose is not None and self.pose.batch_size != self.batch_size:
            raise ValueError(
                "The batch size of pose must match the batch size of "
                "sensor data. "
                f"Expected {self.batch_size}, got {self.pose.batch_size}."
            )
        if (
            self.intrinsic_matrices is not None
            and self.intrinsic_matrices.shape[0] != self.batch_size
        ):
            raise ValueError(
                "The batch size of intrinsic matrices must match the "
                "batch size of sensor data. "
                f"Expected {self.batch_size}, got "
                f"{self.intrinsic_matrices.shape[0]}."
            )

        data_hw = get_image_shape(self.sensor_data)
        if self.image_shape is None:
            self.image_shape = data_hw
        else:
            if self.image_shape != data_hw:
                raise ValueError(
                    "The image shape of the sensor data does not match "
                    "the expected image shape. "
                    f"Expected {self.image_shape}, got {data_hw}."
                )

    @property
    def batch_size(self) -> int:
        """Get the batch size.

        The batch size is the number of cameras in the batch.

        Returns:
            int: The batch size.
        """
        return self.sensor_data.shape[0]

    def concat(self, others: Sequence[Self]) -> Self:
        # check pix_fmt
        for pix_fmt in [other.pix_fmt for other in others]:
            if pix_fmt != self.pix_fmt:
                raise ValueError(
                    "All BatchCameraData objects must have the same pix_fmt."
                )
        # concat sensor_data:
        super_ret = super().concat(others)

        return type(self)(
            sensor_data=torch.cat(
                [self.sensor_data] + [other.sensor_data for other in others],  # type: ignore
                dim=0,
            ),
            pix_fmt=copy.copy(self.pix_fmt),
            timestamps=concat_timestamps(
                [self.timestamps] + [other.timestamps for other in others],
            ),
            **super_ret.__dict__,
        )

    @torch.no_grad()
    def apply_transform2d(
        self,
        transform: Transform2D_M,
        target_hw: tuple[int, int] | None = None,
        inter_mode: Literal["bilinear", "nearest", "bicubic"] = "bicubic",
        padding_mode: Literal["zeros"] = "zeros",
    ) -> BatchCameraData:
        """Apply a 2D transformation to the camera data.

        Args:
            transform (Transform2D_M): The transformation to apply.

        Returns:
            BatchCameraData: The transformed camera data.
        """
        import cv2

        cv_inter_mode = None
        if inter_mode == "bilinear":
            cv_inter_mode = cv2.INTER_LINEAR
        elif inter_mode == "nearest":
            cv_inter_mode = cv2.INTER_NEAREST
        elif inter_mode == "bicubic":
            cv_inter_mode = cv2.INTER_CUBIC
        else:
            raise ValueError(
                f"Unsupported interpolation mode: {inter_mode}. "
                "Supported modes are: 'bilinear', 'nearest', 'bicubic'."
            )

        layout = guess_channel_layout(self.sensor_data)
        if layout != ImageChannelLayout.HWC:
            raise NotImplementedError(
                "Only HWC channel layout is supported for applying "
                "2D transformations."
            )

        if self.image_shape is None:
            src_hw = get_image_shape(self.sensor_data)
        else:
            src_hw = self.image_shape

        if target_hw is None:
            target_hw = src_hw

        src = self.sensor_data

        trans = transform.get_matrix()
        if trans.shape[0] != src.shape[0] and trans.shape[0] != 1:
            raise ValueError(
                "The transformation matrix must have the same batch size as "
                "the camera data. "
                f"Expected {src.shape[0]}, got {trans.shape[0]}."
            )
        if trans.shape[0] == 1 and src.shape[0] > 1:
            # repeat the transformation matrix for each camera
            trans = trans.repeat(src.shape[0], 1, 1)
        dst = torch.zeros(
            size=(
                src.shape[0],
                target_hw[0],
                target_hw[1],
                src.shape[-1],
            ),
            dtype=src.dtype,
        )
        dst_intrinsic = (
            self.intrinsic_matrices.clone()
            if self.intrinsic_matrices is not None
            else None
        )
        if dst_intrinsic is not None:
            dst_intrinsic = trans @ dst_intrinsic

        non_align_corner_mat = affine_mat2non_align_corners(
            mat=trans, src_hw=src_hw, dst_hw=target_hw
        )

        for i in range(src.shape[0]):
            cur_img = src[i].numpy()
            cur_dst = dst[i].numpy()
            cur_mat = non_align_corner_mat[i].numpy()
            cv2.warpAffine(
                src=cur_img,
                M=cur_mat[:2, :],
                dsize=(target_hw[1], target_hw[0]),
                flags=cv_inter_mode,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0,),
                dst=cur_dst,
            )
        ret_dict = self.__dict__.copy()
        ret_dict["sensor_data"] = dst
        ret_dict["intrinsic_matrices"] = dst_intrinsic
        ret_dict["image_shape"] = target_hw
        return BatchCameraData(**ret_dict)


class BatchCameraDataEncoded(BatchCameraInfo):
    """Data class for batched compressed camera sensor data.

    BatchCameraCompressedData extends BatchCameraDataInfo to include the
    actual compressed sensor data in tensor format.

    A batch of camera data shares the same image shape, distortion model.
    The intrinsic matrices and extrinsic matrices (pose) of the cameras
    can be different.
    """

    sensor_data: list[bytes]
    """The sensor data in compressed bytes."""

    format: Literal["jpeg", "png"]
    """The format of the compressed sensor data."""

    timestamps: list[int] | None = None
    """Timestamps of the camera data in nanoseconds(1e-9 seconds)."""

    def __post_init__(self):
        super().__post_init__()
        if (
            self.timestamps is not None
            and len(self.timestamps) != self.batch_size
        ):
            raise ValueError(
                "The length of timestamps must match the batch size. "
                f"Expected {self.batch_size}, got {len(self.timestamps)}."
            )
        if self.pose is not None and self.pose.batch_size != self.batch_size:
            raise ValueError(
                "The batch size of pose must match the batch size of "
                "sensor data. "
                f"Expected {self.batch_size}, got {self.pose.batch_size}."
            )
        if (
            self.intrinsic_matrices is not None
            and self.intrinsic_matrices.shape[0] != self.batch_size
        ):
            raise ValueError(
                "The batch size of intrinsic matrices must match the "
                "batch size of sensor data. "
                f"Expected {self.batch_size}, got "
                f"{self.intrinsic_matrices.shape[0]}."
            )

    @property
    def batch_size(self) -> int:
        """Get the batch size.

        The batch size is the number of cameras in the batch.

        Returns:
            int: The batch size.
        """
        return len(self.sensor_data)

    def decode(
        self,
        decoder: Callable[[bytes, str], TorchTensor],
        pix_fmt: Literal["rgb", "bgr", "gray", "depth"] | None = None,
        device: Device = "cpu",
    ) -> BatchCameraData:
        """Decode the compressed sensor data to a BatchCameraData.

        Args:
            decoder (Callable[[bytes, str], TorchTensor]): The decoder
                function to decode the compressed data. It should take
                a byte string and the format as input and return a tensor.
            pix_fmt (Literal["rgb", "bgr", "gray", "depth"] | None, optional):
                The pixel format of the decoded data. Defaults to None.
            device (Device, optional): The device to put the decoded data on.
                Defaults to "cpu".

        Returns:
            BatchCameraData: The decoded camera data.
        """
        decoded_data = [
            decoder(data, self.format) for data in self.sensor_data
        ]
        sensor_data = torch.stack(decoded_data, dim=0).to(device)
        image_shape = self.image_shape
        if image_shape is None:
            image_shape = (sensor_data.shape[1], sensor_data.shape[2])
        else:
            if image_shape != (sensor_data.shape[1], sensor_data.shape[2]):
                raise ValueError(
                    "The image shape of the decoded data does not match "
                    "the expected image shape. "
                    f"Expected {self.image_shape}, got "
                    f"({sensor_data.shape[1]}, {sensor_data.shape[2]})"
                )

        return BatchCameraData(
            topic=self.topic,
            frame_id=self.frame_id,
            image_shape=image_shape,
            intrinsic_matrices=self.intrinsic_matrices,
            distortion=self.distortion,
            pose=self.pose,
            sensor_data=sensor_data,
            pix_fmt=pix_fmt,
            timestamps=self.timestamps,
        )

    def concat(self, others: Sequence[Self]) -> Self:
        # check pix_fmt
        for format in [other.format for other in others]:
            if format != self.format:
                raise ValueError(
                    "All BatchCameraDataEncoded objects must have the same format."  # noqa: E501
                )
        # concat sensor_data:
        super_ret = super().concat(others)

        return type(Self)(
            sensor_data=torch.cat(
                [self.sensor_data] + [other.sensor_data for other in others],  # type: ignore
                dim=0,
            ),
            format=copy.copy(self.format),
            timestamps=concat_timestamps(
                [self.timestamps] + [other.timestamps for other in others],
            ),
            **super_ret.__dict__,
        )
