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

from robo_orchard_core.utils.ray import DEFAULT_RAY_INIT_CONFIG


@pytest.fixture(scope="session", autouse=True)
def workspace() -> str:
    return os.environ["ROBO_ORCHARD_TEST_WORKSPACE"]


@pytest.fixture(scope="session", autouse=True)
def ray_init():
    import ray

    ray_cfg = DEFAULT_RAY_INIT_CONFIG.copy()
    if not ray.is_initialized():
        print("Initializing Ray with config:", ray_cfg)
        ray.init(**ray_cfg)
        print("resources: ", ray.available_resources())

    yield
