# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
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

from robo_orchard_core.envs.env_base import EnvBase, EnvBaseCfg
from robo_orchard_core.policy.base import ClassType, PolicyConfig, PolicyMixin


class DummyEnv(EnvBase):
    cfg: DummyEnvConfig

    def __init__(self, cfg: DummyEnvConfig) -> None:
        self.cfg = cfg
        self.step_count = 0

    def reset(self, **kwargs):
        self.step_count = 0
        return {"obs": 0}, {"step": self.step_count}

    def step(self, action):
        self.step_count += 1
        obs = self.step_count
        reward = 1.0 if self.step_count >= self.cfg.success_step_count else 0.0
        terminated = self.step_count >= self.cfg.success_step_count
        truncated = False
        info = {"step": self.step_count}
        return (
            {"obs": obs, "act": action},
            reward,
            terminated,
            truncated,
            info,
        )

    def close(self):
        pass

    def num_envs(self) -> int:
        return 1


class DummyEnvConfig(EnvBaseCfg[DummyEnv]):
    class_type: ClassType[DummyEnv] = DummyEnv

    success_step_count: int = 5


class DummyPolicy(PolicyMixin):
    cfg: DummyPolicyConfig

    def __init__(self, cfg: DummyPolicyConfig, *args, **kwargs) -> None:
        self.cfg = cfg

    def reset(self, **kwargs) -> None:
        pass

    def act(self, obs):
        return obs  # Always return the obs

    def close(self) -> None:
        pass


class DummyPolicyConfig(PolicyConfig[DummyPolicy]):
    class_type: ClassType[DummyPolicy] = DummyPolicy
