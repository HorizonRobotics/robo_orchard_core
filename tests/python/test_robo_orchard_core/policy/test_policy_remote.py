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
from __future__ import annotations
import os

from dummy_policy import DummyPolicyConfig

from robo_orchard_core.policy.remote import RemotePolicyConfig
from robo_orchard_core.utils.ray import RayRemoteClassConfig


class TestRemotePolicy:
    def test_config_as_remote(self):
        a = DummyPolicyConfig().as_remote()
        assert isinstance(a, RemotePolicyConfig)
        assert isinstance(a.instance_config, DummyPolicyConfig)

        a2 = a.instance_config.as_remote()
        assert a2 == a

    def test_init(self):
        # get the directory of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        a = DummyPolicyConfig().as_remote(
            remote_class_config=RayRemoteClassConfig(
                num_cpus=0.01,
                num_gpus=0,
                memory=16 * 1024**2,
                runtime_env={
                    "env_vars": {
                        "PYTHONPATH": current_dir,
                    },
                    #     # "working_dir": current_dir
                },
            ),
        )
        remote = a.__call__()
        # assert remote.reset() == "reset", (), {}
        reset_ret = remote.reset()
        assert reset_ret[0] == "reset"
        call_ret = remote("obs")
        assert call_ret[0] == "act"
