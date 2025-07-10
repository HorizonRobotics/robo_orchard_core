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

import pytest
import torch

from robo_orchard_core.datatypes.joint_state import BatchJointsState


class TestBatchJointsState:
    @pytest.mark.parametrize("mode", ["json", "python"])
    def test_to_dict(self, mode: str):
        joints_state = BatchJointsState(
            position=torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
            velocity=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
            effort=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            names=["joint1", "joint2"],
        )
        a = joints_state.model_dump(mode=mode)
        assert isinstance(a, dict)
        for field in BatchJointsState.model_fields:
            assert field in a, f"Field {field} is missing in the dumped dict"

    def test_init(self):
        # Test with all attributes
        joints_state = BatchJointsState(
            position=torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
            velocity=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
            effort=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            names=["joint1", "joint2"],
        )
        assert joints_state.position.shape == (2, 2)  # type: ignore
        assert joints_state.velocity.shape == (2, 2)  # type: ignore
        assert joints_state.effort.shape == (2, 2)  # type: ignore
        assert joints_state.names == ["joint1", "joint2"]  # type: ignore

    @pytest.mark.parametrize(
        "position, velocity, effort, names",
        [
            (
                torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
                torch.tensor([[0.1], [0.3]]),  # missing dim
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                ["joint1", "joint2"],
            ),
            (
                torch.tensor([[0.0], [1.0]]),
                torch.tensor([[0.1], [0.2]]),
                torch.tensor([[1.0], [2.0]]),
                ["joint1", "joint2"],  # missing dim
            ),
            (
                torch.tensor([[0.0], [3.0]]),
                torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                ["joint1", "joint2"],
            ),
            (
                torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                torch.tensor([[0.0], [3.0]]),
                ["joint1", "joint2"],
            ),
        ],
    )
    def test_check_shape(self, position, velocity, effort, names):
        # Test shape checking
        with pytest.raises(ValueError):
            BatchJointsState(
                position=position,  # Invalid shape
                velocity=velocity,
                effort=effort,
                names=names,
            )
