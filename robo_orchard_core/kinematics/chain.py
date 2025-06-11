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

"""Kinematic chains.

A kinematic chain is a structure that represents a robot's
kinematics. It consists of links and joints.

"""

from __future__ import annotations
import os
from typing import Literal

import pytorch_kinematics as pk
import torch
from pytorch_kinematics.frame import Frame

from robo_orchard_core.utils.math.transform.transform3d import Transform3D_M


class KinematicChain:
    """A chain of links and joints that represent a kinematic chain."""

    def __init__(self, chain: pk.Chain) -> None:
        self._chain = chain
        self._device = torch.device(chain.device)

    def get_chain_rep_doc(self) -> str:
        """Get the chain representation docstring."""
        return self._chain.print_tree(do_print=False)

    @staticmethod
    def from_content(
        data: str,
        format: Literal["urdf", "sdf", "mjcf"],
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> KinematicChain:
        """Create a kinematic chain from the content of a file.

        Args:
            data (str): The content of the file.
            format (Literal["urdf", "sdf", "mjcf"]): The format of the file.
            device (str, optional): The device to use. Defaults to "cpu".
        """

        torch_device = torch.device(device)
        if format == "urdf":
            chain = pk.build_chain_from_urdf(data)
        elif format == "sdf":
            chain = pk.build_chain_from_sdf(data)
        elif format == "mjcf":
            chain = pk.build_chain_from_mjcf(data)
        else:
            raise ValueError(f"unsupported format {format}")
        chain = chain.to(device=torch_device, dtype=dtype)
        return KinematicChain(chain)

    @staticmethod
    def from_file(
        path: str,
        format: Literal["urdf", "sdf", "mjcf"],
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """Create a kinematic chain from a file.

        Args:
            path (str): The path to the file.
            format (Literal["urdf", "sdf", "mjcf"]): The format of the file.
            device (str, optional): The device to use. Defaults to "cpu".
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"file not found at {path}")

        with open(path, "r") as f:
            data = f.read()
        return KinematicChain.from_content(
            data.encode("utf-8"),  # type: ignore
            format,
            device=device,
            dtype=dtype,
        )

    def find_link(self, name: str) -> pk.Link | None:
        """Find a link in the chain by name.

        If the link is not found, None is returned.
        """

        return self._chain.find_link(name)

    def find_joint(self, name: str) -> pk.Joint | None:
        """Find a joint in the chain by name.

        If the joint is not found, None is returned.
        """
        return self._chain.find_joint(name)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._chain.dtype

    @property
    def joint_parameter_names(self) -> list[str]:
        """The names of the joint parameters in the chain.

        Fixed joints are excluded from this list.
        """
        return self._chain.get_joint_parameter_names(exclude_fixed=True)

    @property
    def dof(self) -> int:
        """The number of degrees of freedom in the chain."""
        return self._chain.n_joints

    @property
    def frame_names(self) -> list[str]:
        """The names of the frames in the chain."""
        return self._chain.get_frame_names(exclude_fixed=False)

    def find_frame(self, name: str) -> Frame | None:
        r"""Find a frame in the chain by name.

        A Frame is defined as follows:

        .. code-block:: text

                            ||--------Frame0--------||
                                                    ||----------Frame0 Children-----||
                                                    ||----------Frame1--------------||
            [Parent_link0]                          joint1 ->  [link1]
                        \                        /
                            joint0  -->  [link0]
                                                \
                                                    joint2 ->  [link2]
                                                    ||----------Frame2--------------||

        """  # noqa: E501

        return self._chain.find_frame(name)

    def forward_kinematics(
        self,
        joint_positions: torch.Tensor,
    ) -> dict[str, Transform3D_M]:
        """Compute forward kinematics for the chain.

        Args:
            joint_positions (torch.Tensor): The joint positions tensor.
                The tensor should be of shape (N, DOF) where N is the batch
                size and DOF is the number of degrees of freedom in the chain.
                The joint_positions tensor should follow the same order as the
                chain's joint order of `self.joint_parameter_names`.

        Returns:
            dict[str, Transform3D_M]: A dictionary containing the forward
                kinematics of the chain. The keys of the dictionary are the
                names of the frames in the chain and the values are the
                corresponding transformation matrices.
        """

        # joint_positions should be a tensor of shape (N, DOF)
        # where DOF is the number of degrees of freedom in the chain
        # and N is the batch size.

        # The joint_positions tensor should follow the same order as the
        # chain's joint order of self.joint_parameter_names

        fk_dict = self._chain.forward_kinematics(
            joint_positions, frame_indices=None
        )
        fk_dict = {
            k: Transform3D_M(
                dtype=v.dtype, device=v.device, matrix=v.get_matrix()
            )
            for k, v in fk_dict.items()
        }
        return fk_dict


class KinematicSerialChain(KinematicChain):
    """A serial chain of links and joints that represent a kinematic chain.

    A serial chain is a special type of kinematic chain that has no branching.
    """

    def __init__(
        self,
        chain: KinematicChain,
        end_frame_name: str,
        root_frame_name: str = "",
    ):
        self._chain = pk.SerialChain(
            chain._chain,
            end_frame_name=end_frame_name,
            root_frame_name=root_frame_name,
            device=chain._device,
            dtype=chain.dtype,
        )
        self._device = chain.device

    @staticmethod
    def from_content(
        data: str,
        format: Literal["urdf", "sdf", "mjcf"],
        end_frame_name: str,
        root_frame_name: str = "",
        device: str = "cpu",
    ) -> KinematicSerialChain:
        chain = KinematicChain.from_content(data, format, device)
        return KinematicSerialChain(chain, end_frame_name, root_frame_name)

    @staticmethod
    def from_file(
        path: str,
        format: Literal["urdf", "sdf", "mjcf"],
        end_frame_name: str,
        root_frame_name: str = "",
        device: str = "cpu",
    ) -> KinematicSerialChain:
        chain = KinematicChain.from_file(path, format, device)
        return KinematicSerialChain(chain, end_frame_name, root_frame_name)

    def forward_kinematics(
        self, joint_positions: torch.Tensor
    ) -> dict[str, Transform3D_M]:
        fk_dict = self._chain.forward_kinematics(
            joint_positions, end_only=False
        )
        assert isinstance(fk_dict, dict)
        fk_dict = {
            k: Transform3D_M(
                dtype=v.dtype, device=v.device, matrix=v.get_matrix()
            )
            for k, v in fk_dict.items()
        }
        return fk_dict

    def jacobian(self, joint_positions: torch.Tensor) -> torch.Tensor:
        """Compute the geometric Jacobian of the chain.

        Args:
            joint_positions (torch.Tensor): The joint positions tensor.
                The tensor should be of shape (N, DOF) where N is the batch
                size and DOF is the number of degrees of freedom in the chain.
                The joint_positions tensor should follow the same order as the
                chain's joint order of `self.joint_parameter_names`.

        Returns:
            torch.Tensor: The geometric Jacobian tensor of shape (N, 6, DOF).

        """
        ret = self._chain.jacobian(
            joint_positions,
            ret_eef_pose=False,
        )
        assert isinstance(ret, torch.Tensor)
        return ret
