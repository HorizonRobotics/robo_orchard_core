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

from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import Callable

import torch
from typing_extensions import Generic, TypeVar

from robo_orchard_core.datatypes.camera_data import (
    BatchCameraData,
    CameraData,
    Distortion,
)
from robo_orchard_core.datatypes.geometry import (
    BatchFrameTransform,
    FrameTransform,
)
from robo_orchard_core.utils.config import (
    ClassConfig,
    ClassInitFromConfigMixin,
    ClassType_co,
)
from robo_orchard_core.utils.hook import HookHandler

CameraBaseType_co = TypeVar(
    "CameraBaseType_co", bound="CameraBase|BatchCameraBase", covariant=True
)
CameraBaseCfgType_co = TypeVar(
    "CameraBaseCfgType_co", bound="CameraBaseCfg", covariant=True
)


class CameraBase(
    ClassInitFromConfigMixin, Generic[CameraBaseCfgType_co], metaclass=ABCMeta
):
    """The base class of a single camera."""

    def __init__(self, cfg: CameraBaseCfgType_co):
        self.cfg = cfg

        self.after_capture_hook_handler: HookHandler[
            Callable[[CameraBase[CameraBaseCfgType_co]], None]
        ] = HookHandler(name="after_capture_hook_handler")

    @property
    @abstractmethod
    def image_shape(self) -> tuple[int, int]:
        """Get the shape(height, width) of the image.

        Returns:
            tuple[int, int]: A tuple containing (height, width) of the camera.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def intrinsic_matrix(self) -> torch.Tensor:
        """Get the intrinsic matrix of the camera.

        Returns:
            torch.Tensor: The intrinsic matrix of the camera.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def pose_global(self) -> FrameTransform:
        """Get the pose of the camera in the global frame.

        Global frame usually refers to the world frame, or the frame that is
        shared by all sensors and objects in the scene.

        Returns:
            FrameTransform: The pose of the camera in the global frame.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def sensor_data(self) -> torch.Tensor:
        """Get the camera data of the camera.

        Shape is (H, W, C) for raw data, where C is the number of channels,
        H is the height of the image, and W is the width of the image.

        For compressed data, the shape is (N, ) where N is the number of bytes.

        Returns:
            torch.Tensor: The camera data.
        """
        raise NotImplementedError

    @property
    def distortion(self) -> Distortion | None:
        """Get the distortion model of the camera.

        User should inherit this method to return the distortion model
        if it is applied to the camera.

        Returns:
            Distortion | None: The distortion model of the camera.
            If no distortion model is applied, returns None.
        """
        return None

    @property
    def tf_parent(self) -> FrameTransform | None:
        """Get the parent frame transform of the camera.

        User should inherit this method to return the parent frame transform
        if it is applied to the camera.

        Returns:
            FrameTransform | None: The parent frame transform of the camera.
            If no parent frame transform is applied, returns None.
        """
        return None

    def get_camera_data(self, tf_global: bool = True) -> CameraData:
        """Get the camera data of the camera.

        Args:
            tf_global (bool, optional): If True, the pose will be transformed
                to the global frame. If False, the pose will be in the parent
                frame. Defaults to True.

        Returns:
            CameraData: The camera data.
        """
        pose = self.pose_global if tf_global else self.tf_parent
        return CameraData(
            intrinsic_matrix=self.intrinsic_matrix,
            pose=pose,
            sensor_data=self.sensor_data,
            image_shape=self.image_shape,
            distortion=self.distortion,
        )


class CameraBaseCfg(ClassConfig[CameraBaseType_co]):
    """The base configuration for all cameras."""

    class_type: ClassType_co[CameraBaseType_co]


CameraBaseCfgType_co = TypeVar(
    "CameraBaseCfgType_co", bound=CameraBaseCfg, covariant=True
)


class BatchCameraBase(
    ClassInitFromConfigMixin, Generic[CameraBaseCfgType_co], metaclass=ABCMeta
):
    """The base class of a batch of cameras.

    A batch of cameras is a collection of cameras that share the image shape
    and distortion model. The intrinsic matrices and poses of the cameras
    can be different.

    """

    def __init__(self, cfg: CameraBaseCfgType_co, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.after_capture_hook_handler: HookHandler[
            Callable[[BatchCameraBase[CameraBaseCfgType_co]], None]
        ] = HookHandler(name="after_capture_hook_handler")

    @property
    @abstractmethod
    def image_shape(self) -> tuple[int, int]:
        """Get the shape(height, width) of the image.

        Returns:
            tuple[int, int]: A tuple containing (height, width) of the camera.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def intrinsic_matrices(self) -> torch.Tensor:
        """Get the intrinsic matrix of the camera.

        Returns:
            CameraData: The intrinsic matrix of the camera.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def pose_global(self) -> BatchFrameTransform:
        """Get the pose of the camera in the global frame.

        Global frame usually refers to the world frame, or the frame that is
        shared by all sensors and objects in the scene.

        Returns:
            BatchFrameTransform: The pose of the camera in the global frame.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def sensor_data(self) -> torch.Tensor:
        """Get the camera data of the camera.

        Shape is (N, H, W, C) for raw data, where B is the batch size,
        C is the number of channels, H is the height of the image,
        and W is the width of the image.

        Returns:
            torch.Tensor: The camera data.
        """
        raise NotImplementedError

    @property
    def distortion(self) -> Distortion | None:
        """Get the distortion model of the camera.

        User should inherit this method to return the distortion model
        if it is applied to the camera.

        Returns:
            Distortion | None: The distortion model of the camera.
            If no distortion model is applied, returns None.
        """
        return None

    @property
    def tf_parent(self) -> BatchFrameTransform | None:
        """Get the parent frame transform of the camera.

        User should inherit this method to return the parent frame transform
        if it is applied to the camera.

        Returns:
            BatchFrameTransform | None: The parent frame transform of
                the camera. If no parent frame transform is applied,
                returns None.
        """
        return None

    def get_camera_data(self, tf_global: bool = True) -> BatchCameraData:
        """Get the camera data of the camera.

        Args:
            tf_global (bool, optional): If True, the pose will be transformed
                to the global frame. If False, the pose will be in the parent
                frame. Defaults to True.

        Returns:
            CameraData: The camera data.
        """

        pose = self.pose_global if tf_global else self.tf_parent

        return BatchCameraData(
            intrinsic_matrices=self.intrinsic_matrices,
            pose=pose,
            distortion=self.distortion,
            sensor_data=self.sensor_data,
            image_shape=self.image_shape,
        )
