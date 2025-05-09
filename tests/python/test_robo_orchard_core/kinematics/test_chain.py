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

import os

import pytest
import torch

from robo_orchard_core.kinematics.chain import (
    KinematicChain,
    KinematicSerialChain,
)


class TestKinematicChain:
    @pytest.fixture
    def urdf_path(self, workspace: str):
        return os.path.join(
            workspace,
            "robo_orchard_workspace/assets/public/franka_description/panda.urdf",
        )

    @pytest.fixture
    def urdf_content(self, urdf_path: str):
        with open(urdf_path, "r") as f:
            return f.read()

    def test_from_content(self, urdf_content: str):
        chain = KinematicChain.from_content(urdf_content, "urdf")
        assert chain is not None

    def test_from_path(self, urdf_path: str):
        chain = KinematicChain.from_file(urdf_path, "urdf")
        assert chain is not None

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
    def test_forward_kinematics_with_device(self, urdf_path: str, device: str):
        chain = KinematicChain.from_file(urdf_path, "urdf", device=device)
        assert chain is not None

        q = torch.rand(chain.dof, device=chain.device, dtype=torch.float32)

        print("frame_names: ", chain.frame_names)
        print("joint_parameter_names: ", chain.joint_parameter_names)
        fk = chain.forward_kinematics(q)
        print("fk: ", fk)

        # check device
        m_0 = fk[list(fk.keys())[0]].get_matrix()
        assert m_0.device == chain.device
        assert m_0.dim() == 3

    def test_fk_batch(self, urdf_path):
        chain = KinematicChain.from_file(urdf_path, "urdf")
        assert chain is not None
        b = 10
        q = torch.rand(b, chain.dof, device=chain.device, dtype=torch.float32)
        fk = chain.forward_kinematics(q)
        assert fk is not None
        m_0 = fk[list(fk.keys())[0]].get_matrix()
        assert m_0.shape[0] == b
        assert m_0.dim() == 3


class TestKinematicSerialChain:
    @pytest.fixture
    def urdf_content(self, workspace: str):
        path = os.path.join(
            workspace,
            "robo_orchard_workspace/assets/public/franka_description/panda.urdf",
        )
        with open(path, "r") as f:
            return f.read()

    def test_init(self, urdf_content: str):
        chain = KinematicSerialChain.from_content(
            urdf_content,
            "urdf",
            root_frame_name="panda_link0",
            end_frame_name="panda_link8",
        )
        assert chain is not None
        print("frame_names: ", chain.frame_names)
        print("joint_parameter_names: ", chain.joint_parameter_names)

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
    def test_fk_with_device(self, urdf_content: str, device: str):
        chain = KinematicSerialChain.from_content(
            urdf_content,
            "urdf",
            root_frame_name="panda_link0",
            end_frame_name="panda_link8",
            device=device,
        )
        assert chain is not None

        q = torch.rand(chain.dof, device=chain.device, dtype=torch.float32)

        print("frame_names: ", chain.frame_names)
        print("joint_parameter_names: ", chain.joint_parameter_names)
        fk = chain.forward_kinematics(q)
        print("fk: ", fk)

        # check device
        m_0 = fk[list(fk.keys())[0]].get_matrix()
        assert m_0.device == chain.device
        assert m_0.dim() == 3

    def test_fk_with_batch(self, urdf_content: str):
        chain = KinematicSerialChain.from_content(
            urdf_content,
            "urdf",
            root_frame_name="panda_link0",
            end_frame_name="panda_link8",
        )
        assert chain is not None

        b = 10
        q = torch.rand(b, chain.dof, device=chain.device, dtype=torch.float32)
        fk = chain.forward_kinematics(q)
        assert fk is not None
        m_0 = fk[list(fk.keys())[0]].get_matrix()
        assert m_0.shape[0] == b
        assert m_0.dim() == 3

    def test_jacobian(self, urdf_content: str):
        chain = KinematicSerialChain.from_content(
            urdf_content,
            "urdf",
            root_frame_name="panda_link0",
            end_frame_name="panda_link8",
        )
        assert chain is not None

        q = torch.rand(chain.dof, device=chain.device, dtype=torch.float32)
        J = chain.jacobian(q)
        assert J is not None
        assert J.dim() == 3
        assert J.shape[0] == 1
        assert J.shape[1] == 6
        assert J.shape[2] == chain.dof

        b = 10
        q = torch.rand(b, chain.dof, device=chain.device, dtype=torch.float32)
        J = chain.jacobian(q)
        assert J is not None
        assert J.dim() == 3
        assert J.shape[0] == b
        assert J.shape[1] == 6
        assert J.shape[2] == chain.dof

    def test_jacobian_compare_to_autograd(self, urdf_content: str):
        chain = KinematicSerialChain.from_content(
            urdf_content,
            "urdf",
            root_frame_name="panda_link0",
            end_frame_name="panda_link8",
        )
        assert chain is not None

        def f(q):
            fk = chain.forward_kinematics(q)["panda_link8"]
            return torch.cat(
                [fk.get_translation(), fk.get_rotation_axis_angle()], dim=-1
            )

        b = 1000
        q = torch.rand(b, chain.dof, device=chain.device, dtype=torch.float32)
        J = chain.jacobian(q)
        J_auto_grad = torch.autograd.functional.jacobian(f, q, vectorize=True)
        # f(q) will produce N x 6
        # jacobian will compute the jacobian of
        # the N x 6 matrix with respect to each of the N x DOF inputs, and
        # output a N x 6 x N x DOF tensor
        # We only need the diagonal elements of this tensor
        # because the ith output has a non-zero jacobian with the ith input.
        J_auto_grad_1 = J_auto_grad[range(b), :, range(b)]  # type: ignore

        # Only position can be compared!

        assert torch.allclose(
            J[..., :3, :], J_auto_grad_1[..., :3, :], atol=1e-4
        )
