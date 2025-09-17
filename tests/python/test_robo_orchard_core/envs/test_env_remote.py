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

from ut_help import DummyEnvConfig, DummyPolicyConfig

from robo_orchard_core.envs.remote import RemoteEnvCfg
from robo_orchard_core.utils.ray import RayRemoteClassConfig


class TestRemoteEnv:
    def test_config_as_remote(self):
        a = DummyEnvConfig().as_remote()
        assert isinstance(a, RemoteEnvCfg)
        assert isinstance(a.instance_config, DummyEnvConfig)

        a2 = a.instance_config.as_remote()
        assert a2 == a

    def test_init(self):
        # get the directory of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        a = DummyEnvConfig().as_remote(
            remote_class_config=RayRemoteClassConfig(
                num_cpus=0.01,
                num_gpus=0,
                runtime_env={
                    "env_vars": {
                        "PYTHONPATH": current_dir,
                    },
                    #     # "working_dir": current_dir
                },
            ),
        )
        remote = a.__call__()
        reset_ret, reset_info = remote.reset()
        assert reset_ret == {"obs": 0}
        assert reset_info == {"step": 0}

        step_ret = remote.step("action_1")
        assert step_ret[0] == {"obs": 1, "act": "action_1"}  # type: ignore
        assert step_ret[1] == 0.0  # type: ignore
        assert step_ret[2] is False  # type: ignore
        assert step_ret[3] is False  # type: ignore
        assert step_ret[4] == {"step": 1}  # type: ignore

        for i in range(4):
            step_ret = remote.step(f"action_{i + 2}")
            if i != 3:
                assert step_ret[0] == {"obs": i + 2, "act": f"action_{i + 2}"}  # type: ignore
                assert step_ret[1] == 0  # type: ignore
                assert step_ret[2] is False  # type: ignore
                assert step_ret[3] is False  # type: ignore
                assert step_ret[4] == {  # type: ignore
                    "step": i + 2,
                }
        assert step_ret[0] == {"obs": 5, "act": "action_5"}  # type: ignore
        assert step_ret[1] == 1.0  # type: ignore
        assert step_ret[2] is True  # type: ignore
        assert step_ret[3] is False  # type: ignore
        assert step_ret[4] == {"step": 5}  # type: ignore

        remote.close()

    def test_async_rollout(self):
        # get the directory of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        a = DummyEnvConfig().as_remote(
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
        reset_ret, reset_info = remote.reset()
        policy = DummyPolicyConfig()()
        res = remote.async_rollout(
            max_steps=4, init_obs=reset_ret, policy=policy
        ).result()
        assert res.init_obs == reset_ret
        assert res.env_step_callback is None
        assert res.rollout_actual_steps == 4
        assert res.terminal_condition_triggered is None
        print(res)

    def test_rollout_with_terminal_condition(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        a = DummyEnvConfig().as_remote(
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
        reset_ret, reset_info = remote.reset()
        policy = DummyPolicyConfig()()

        def stop_cond(step_ret) -> bool:
            return step_ret[2]

        reset_ret, reset_info = remote.reset()
        res = remote.rollout(
            max_steps=10,
            init_obs=reset_ret,
            policy=policy,
            terminal_condition=stop_cond,
        )
        assert res.rollout_actual_steps == 5
