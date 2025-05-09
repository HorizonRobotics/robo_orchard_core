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

from abc import ABCMeta, abstractmethod
from typing import Literal

import torch

from robo_orchard_core.utils.torch_utils import Device

__all__ = [
    "CoordAxis",
    "CoordAxisWorld",
    "CoordAxisCamera",
    "CoordAxisOpenGL",
    "CoordConventionType",
]

CoordConventionType = Literal["opengl", "cam", "world"]


class CoordAxis(metaclass=ABCMeta):
    """Abstract class for coordinate axis."""

    def __init__(
        self, dtype: torch.dtype = torch.float32, device: Device = "cpu"
    ):
        self.device = device
        self.dtype = dtype

    @property
    @abstractmethod
    def convention(self) -> CoordConventionType:
        """Get the coordinate convention."""
        raise NotImplementedError

    @property
    @abstractmethod
    def forward(self) -> torch.Tensor:
        """Get the forward direction."""
        raise NotImplementedError

    @property
    def backward(self) -> torch.Tensor:
        """Get the backward direction."""
        return self.forward * -1.0

    @property
    @abstractmethod
    def left(self) -> torch.Tensor:
        """Get the left direction."""
        raise NotImplementedError

    @property
    def right(self) -> torch.Tensor:
        """Get the right direction."""
        return self.left * -1.0

    @property
    @abstractmethod
    def up(self) -> torch.Tensor:
        """Get the up direction."""
        raise NotImplementedError

    @property
    def down(self) -> torch.Tensor:
        """Get the down direction."""
        return self.up * -1.0


class CoordAxisWorld(CoordAxis):
    """Coordinate axis for world frame."""

    def __init__(
        self, dtype: torch.dtype = torch.float32, device: Device = "cpu"
    ):
        super().__init__(dtype=dtype, device=device)

    @property
    def convention(self) -> CoordConventionType:
        """Get the coordinate convention."""
        return "world"

    @property
    def forward(self) -> torch.Tensor:
        """Get the forward direction (+z-axis)."""
        return torch.tensor(
            [1.0, 0.0, 0.0],
            device=self.device,
            dtype=self.dtype,
        )

    @property
    def left(self) -> torch.Tensor:
        """Get the left direction (+y-axis)."""
        return torch.tensor(
            [0.0, 1.0, 0.0], device=self.device, dtype=self.dtype
        )

    @property
    def up(self) -> torch.Tensor:
        """Get the up direction (+z-axis)."""
        return torch.tensor(
            [0.0, 0.0, 1.0], device=self.device, dtype=self.dtype
        )


class CoordAxisCamera(CoordAxis):
    """Coordinate axis for Camera frame."""

    def __init__(
        self, dtype: torch.dtype = torch.float32, device: Device = "cpu"
    ):
        super().__init__(dtype=dtype, device=device)

    @property
    def convention(self) -> CoordConventionType:
        """Get the coordinate convention."""
        return "cam"

    @property
    def forward(self) -> torch.Tensor:
        """Get the forward direction (+Z-axis)."""
        return torch.tensor(
            [0.0, 0.0, 1.0],
            device=self.device,
            dtype=self.dtype,
        )

    @property
    def left(self) -> torch.Tensor:
        """Get the left direction (-X-axis)."""
        return torch.tensor(
            [-1.0, 0.0, 0.0], device=self.device, dtype=self.dtype
        )

    @property
    def up(self) -> torch.Tensor:
        """Get the up direction (-Y-axis)."""
        return torch.tensor(
            [0.0, -1.0, 0.0], device=self.device, dtype=self.dtype
        )


class CoordAxisOpenGL(CoordAxis):
    """Coordinate axis for OpenGL frame."""

    def __init__(
        self, dtype: torch.dtype = torch.float32, device: Device = "cpu"
    ):
        super().__init__(dtype=dtype, device=device)

    @property
    def convention(self) -> CoordConventionType:
        """Get the coordinate convention."""
        return "opengl"

    @property
    def forward(self) -> torch.Tensor:
        """Get the forward direction (-Z-axis)."""
        return torch.tensor(
            [0.0, 0.0, -1.0],
            device=self.device,
            dtype=self.dtype,
        )

    @property
    def left(self) -> torch.Tensor:
        """Get the left direction (-X-axis)."""
        return torch.tensor(
            [-1.0, 0.0, 0.0], device=self.device, dtype=self.dtype
        )

    @property
    def up(self) -> torch.Tensor:
        """Get the up direction (+Y-axis)."""
        return torch.tensor(
            [0.0, 1.0, 0.0], device=self.device, dtype=self.dtype
        )
