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

from robo_orchard_core.datatypes.camera_data import BatchCameraData, Distortion


class TestBatchCameraData:
    def test_to_dict(self):
        a = BatchCameraData(
            sensor_data=torch.rand(size=(2, 12, 11, 3), dtype=torch.float32),
            pix_fmt="bgr",
            # with distortion
            distortion=Distortion(
                model="plumb_bob",
                coefficients=torch.tensor(
                    [0.1, 0.01, 0.001, 0.0001], dtype=torch.float32
                ),
            ),
        )
        d = a.model_dump()
        for field in BatchCameraData.model_fields:
            assert field in d, f"Field {field} is missing in the dumped dict"


if __name__ == "__main__":
    pytest.main(["-s", __file__])
