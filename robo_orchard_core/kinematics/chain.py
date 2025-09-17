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
import functools
import os
import warnings
from typing import Literal

import pytorch_kinematics as pk
import torch
from pytorch_kinematics.frame import Frame as _Frame

from robo_orchard_core.datatypes.geometry import BatchFrameTransform
from robo_orchard_core.datatypes.joint_state import BatchJointsState
from robo_orchard_core.utils.math.transform.transform3d import Transform3D_M


def _get_timestamps(
    timestamps: list[int] | None, joints: BatchJointsState | torch.Tensor
) -> list[int] | None:
    if isinstance(joints, BatchFrameTransform):
        if isinstance(timestamps, list) and joints.timestamps is not None:
            warnings.warn(
                "Passing timestamps when joints is a "
                "BatchFrameTransform will be ignored.",
            )
            timestamps = joints.timestamps
    return timestamps


class Frame(_Frame):
    """A frame in a kinematic chain.

    A Frame is defined as follows:

    .. code-block:: text

                        ||--------Frame0--------||
                                                ||----------Frame0 Children-----||
                                                ||----------Frame1--------------||
        [Parent_link0]                          joint1 ->  [link1]
                    \\                        /
                        joint0  -->  [link0]
                                             \
                                                joint2 ->  [link2]
                                                ||----------Frame2--------------||

    The frame name is usually the same as the link name. For root frame (the frame
    that has no parent), it is attached to a virtual fixed joint with empty name
    and offset.
    """  # noqa: E501

    @property
    def joint_type(self) -> Literal["fixed", "revolute", "prismatic"]:
        return self.joint.joint_type  # type: ignore

    @classmethod
    def from_frame(cls, frame: _Frame) -> Frame:
        """Create a Frame from a pytorch_kinematics Frame."""
        ret = cls.__new__(cls)
        ret.__dict__.update(frame.__dict__)
        return ret

    def get_transform(
        self, joint_positions: torch.Tensor | BatchJointsState
    ) -> Transform3D_M:
        """Get the transform of this frame w.r.t. the parent frame.

        Args:
            joint_positions (torch.Tensor|BatchJointsState): The joint
                positions tensor. The tensor should be of shape (N,)
                where N is the batch size. If a BatchJointsState is
                provided, the joint positions for this frame will be
                extracted based on the joint name if available, otherwise
                the first joint position will be used.

        """
        if isinstance(joint_positions, BatchJointsState):
            if joint_positions.position is None:
                raise ValueError("joint_positions.position is None")
            # find joint indices
            if joint_positions.names is None:
                theta = joint_positions.position[:, 0]
            else:
                joint_idx = None
                for i, j_name in enumerate(joint_positions.names):
                    if j_name == self.joint.name:
                        joint_idx = i
                        break
                if joint_idx is None:
                    raise ValueError(
                        f"joint {self.joint.name} not found in "
                        "joint_positions.names"
                    )
                theta = joint_positions.position[:, joint_idx]
        else:
            theta = joint_positions

        mat = super().get_transform(theta)
        return Transform3D_M(
            dtype=mat.dtype, device=mat.device, matrix=mat.get_matrix()
        )

    def get_frame_transform(
        self,
        joint_positions: torch.Tensor | BatchJointsState,
        parent_link_name: str,
        timestamps: list[int] | None = None,
    ) -> BatchFrameTransform:
        """Get the transform of this frame w.r.t. the parent link frame.

        Args:
            joint_positions (torch.Tensor|BatchJointsState): The joint
                positions tensor. The tensor should be of shape (N,)
                where N is the batch size. If a BatchJointsState is
                provided, the joint positions for this frame will be
                extracted based on the joint name if available, otherwise
                the first joint position will be used.
            parent_link_name (str): The name of the parent link frame.

        Returns:
            BatchFrameTransform: The transform of this frame w.r.t. the
                parent link frame.
        """
        timestamps = _get_timestamps(timestamps, joint_positions)

        mat = self.get_transform(joint_positions)
        return BatchFrameTransform(
            xyz=mat.get_translation(),
            quat=mat.get_rotation_quaternion(normalize=True),
            timestamps=timestamps,
            parent_frame_id=parent_link_name,
            child_frame_id=self.name,
        )


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

        The frame name is usually the same as the link name. For root frame (the frame
        that has no parent), it is attached to a virtual fixed joint with empty name
        and offset.
        """  # noqa: E501
        ret = self._chain.find_frame(name)
        if ret is not None:
            return Frame.from_frame(ret)
        else:
            return None

    def forward_kinematics_tf(
        self,
        joint_positions: torch.Tensor | BatchJointsState,
        frame_names: list[str] | None = None,
        timestamps: list[int] | None = None,
    ) -> dict[str, BatchFrameTransform]:
        """Compute forward kinematics and return as BatchFrameTransform."""

        if isinstance(joint_positions, BatchJointsState):
            if joint_positions.position is None:
                raise ValueError("joint_positions.position is None")
            joint_positions = joint_positions.position

        timestamps = _get_timestamps(timestamps, joint_positions)

        ret = self.forward_kinematics(joint_positions, frame_names=frame_names)
        root_frame_name = self._chain._root.name
        return {
            k: BatchFrameTransform(
                xyz=v.get_translation(),
                quat=v.get_rotation_quaternion(normalize=True),
                parent_frame_id=root_frame_name,
                child_frame_id=k,
                timestamps=timestamps,
            )
            for k, v in ret.items()
        }

    def forward_kinematics(
        self,
        joint_positions: torch.Tensor,
        frame_names: list[str] | None = None,
    ) -> dict[str, Transform3D_M]:
        """Compute forward kinematics for the chain.

        The forward kinematics is computed as the pose of each frame in
        the chain w.r.t. the root frame.

        Args:
            joint_positions (torch.Tensor): The joint positions tensor.
                The tensor should be of shape (N, DOF) where N is the batch
                size and DOF is the number of degrees of freedom in the chain.
                The joint_positions tensor should follow the same order as the
                chain's joint order of `self.joint_parameter_names`.
            frame_names: A list of frame name to compute transforms for.
                If None, all frames are computed.

        Returns:
            dict[str, Transform3D_M]: A dictionary containing the forward
                kinematics of the chain. The keys of the dictionary are the
                names of the frames in the chain and the values are the
                corresponding pose matrices.
        """

        frame_indices = None
        if frame_names is not None and frame_names != []:
            frame_indices = torch.tensor(
                [self._chain.frame_to_idx[n] for n in frame_names],
                dtype=torch.long,
            )

        fk_dict = self._chain.forward_kinematics(
            joint_positions, frame_indices=frame_indices
        )
        fk_dict = {
            k: Transform3D_M(
                dtype=v.dtype, device=v.device, matrix=v.get_matrix()
            )
            for k, v in fk_dict.items()
        }
        return fk_dict

    @functools.cached_property
    def parent_map(self) -> dict[str, str]:
        """A map from frame name to its parent frame name."""
        parent_map = {}
        for p_name in self._chain.get_frame_names(exclude_fixed=False):
            p_frame = self._chain.find_frame(p_name)
            assert p_frame is not None
            for c in p_frame.children:
                if c.name not in parent_map:
                    parent_map[c.name] = p_name
                else:
                    raise ValueError(
                        f"frame {c.name} has multiple parents: "
                        f"{parent_map[c.name]} and {p_name}"
                    )
        return parent_map


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
        # TODO: Refactor to keep the same API as the base class
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
