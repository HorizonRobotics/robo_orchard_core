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

from robo_orchard_core.datatypes.geometry import (
    BatchFrameTransform,
)
from robo_orchard_core.datatypes.tf_graph import BatchFrameTransformGraph
from robo_orchard_core.utils.math import math_utils


class TestBatchFrameTransformGraph:
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
    def test_get_tf(self, device):
        q = math_utils.normalize(
            torch.rand(size=(3, 4), device=device, dtype=torch.float64) - 0.5,
            dim=-1,
        )
        q_23, q_12, q_01 = math_utils.quaternion_standardize(q).unbind(0)
        t = torch.rand(size=(3, 3), device=device, dtype=torch.float64) - 0.5
        t_23, t_12, t_01 = t.unbind(0)
        batch_transform_23 = BatchFrameTransform(
            xyz=t_23, quat=q_23, parent_frame_id="2", child_frame_id="3"
        )
        batch_transform_12 = BatchFrameTransform(
            xyz=t_12, quat=q_12, parent_frame_id="1", child_frame_id="2"
        )
        batch_transform_01 = BatchFrameTransform(
            xyz=t_01, quat=q_01, parent_frame_id="0", child_frame_id="1"
        )
        tf_g = BatchFrameTransformGraph(
            tf_list=[
                batch_transform_23,
                batch_transform_12,
            ],
            bidirectional=True,
        )

        # test get non
        assert tf_g.get_tf("0", "3", compose=True) is None
        assert tf_g.get_tf("3", "0", compose=False) is None

        tf_g.add_tf([batch_transform_01])

        # test get tf without mirror
        tf_03 = tf_g.get_tf("0", "3", compose=True)
        assert isinstance(tf_03, BatchFrameTransform)
        assert tf_03.parent_frame_id == "0"
        assert tf_03.child_frame_id == "3"
        tf_03_expected = batch_transform_23.compose(
            batch_transform_12.compose(batch_transform_01)
        )

        assert tf_03 == tf_03_expected, (
            f"{tf_03.__dict__} != {tf_03_expected.__dict__}"
        )

        # test get tf with mirror
        tf_30_inv = tf_g.get_tf("3", "0", compose=True)
        assert tf_03.inverse() == tf_30_inv

        # test get without compose
        tf_03_list = tf_g.get_tf("0", "3", compose=False)

        assert isinstance(tf_03_list, list)
        assert len(tf_03_list) == 3
        assert BatchFrameTransform.cls_compose(*tf_03_list) == tf_03_expected

    def test_state(self):
        device = "cpu"
        q = math_utils.normalize(
            torch.rand(size=(3, 4), device=device, dtype=torch.float64) - 0.5,
            dim=-1,
        )
        q_23, q_12, q_01 = math_utils.quaternion_standardize(q).unbind(0)
        t = torch.rand(size=(3, 3), device=device, dtype=torch.float64) - 0.5
        t_23, t_12, t_01 = t.unbind(0)
        batch_transform_23 = BatchFrameTransform(
            xyz=t_23, quat=q_23, parent_frame_id="2", child_frame_id="3"
        )
        batch_transform_12 = BatchFrameTransform(
            xyz=t_12, quat=q_12, parent_frame_id="1", child_frame_id="2"
        )
        batch_transform_01 = BatchFrameTransform(
            xyz=t_01, quat=q_01, parent_frame_id="0", child_frame_id="1"
        )
        tf_g = BatchFrameTransformGraph(
            tf_list=[
                batch_transform_23,
                batch_transform_12,
                batch_transform_01,
            ],
            bidirectional=True,
        )
        state = tf_g.as_state()

        new_tf_g = BatchFrameTransformGraph.from_state(state)

        assert new_tf_g == tf_g

    def test_pickle(self):
        import pickle

        device = "cpu"
        q = math_utils.normalize(
            torch.rand(size=(3, 4), device=device, dtype=torch.float64) - 0.5,
            dim=-1,
        )
        q_23, q_12, q_01 = math_utils.quaternion_standardize(q).unbind(0)
        t = torch.rand(size=(3, 3), device=device, dtype=torch.float64) - 0.5
        t_23, t_12, t_01 = t.unbind(0)
        batch_transform_23 = BatchFrameTransform(
            xyz=t_23, quat=q_23, parent_frame_id="2", child_frame_id="3"
        )
        batch_transform_12 = BatchFrameTransform(
            xyz=t_12, quat=q_12, parent_frame_id="1", child_frame_id="2"
        )
        batch_transform_01 = BatchFrameTransform(
            xyz=t_01, quat=q_01, parent_frame_id="0", child_frame_id="1"
        )
        tf_g = BatchFrameTransformGraph(
            tf_list=[
                batch_transform_23,
                batch_transform_12,
                batch_transform_01,
            ],
            bidirectional=True,
        )
        s = pickle.dumps(tf_g)
        new_tf_g = pickle.loads(s)

        assert new_tf_g == tf_g
