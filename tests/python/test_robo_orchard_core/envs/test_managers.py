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
from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch

from robo_orchard_core.envs.env_base import EnvBase, EnvStepReturn
from robo_orchard_core.envs.managers.actions.action_manager import (
    ActionManager,
    ActionManagerCfg,
)
from robo_orchard_core.envs.managers.actions.action_term import (
    ActionTermBase,
    ActionTermCfg,
)
from robo_orchard_core.envs.managers.events.event_manager import (
    EventManager,
    EventManagerCfg,
    UnregisteredEventChannelError,
)
from robo_orchard_core.envs.managers.events.event_term import (
    EventMsg,
    EventTermBase,
    EventTermBaseCfg,
)
from robo_orchard_core.envs.managers.observations.observation_manager import (
    ObservationGroupCfg,
    ObservationManager,
    ObservationManagerCfg,
)
from robo_orchard_core.envs.managers.observations.observation_term import (
    ObservationTermBase,
    ObservationTermCfg,
)
from robo_orchard_core.envs.managers.scene_entity_cfg import SceneEntityCfg
from robo_orchard_core.utils.config import ClassType


class DummyEnv(EnvBase[Any, Any]):
    @property
    def num_envs(self) -> int:
        return 1

    @property
    def action_space(self) -> gym.Space[Any]:
        return gym.spaces.Dict({})

    @property
    def observation_space(self) -> gym.Space[Any]:
        return gym.spaces.Dict({})

    def step(self, *args: Any, **kwargs: Any) -> EnvStepReturn[Any, Any]:
        return EnvStepReturn(
            observations={},
            rewards=None,
            terminated=None,
            truncated=None,
            info={},
        )

    def reset(
        self,
        seed: int | None = None,
        env_ids: Sequence[int] | None = None,
        **kwargs: Any,
    ) -> tuple[Any, dict[str, Any]]:
        del seed, env_ids, kwargs
        return {}, {}

    def close(self) -> None:
        return None


def test_env_base_get_observations_is_optional() -> None:
    env = DummyEnv()
    with pytest.raises(NotImplementedError, match="get_observations"):
        env.get_observations()


class DummySceneEntityCfg(SceneEntityCfg):
    def resolve(self, scene: object) -> None:
        del scene
        return None


class DummyObservationTerm(
    ObservationTermBase[DummyEnv, "DummyObservationTermCfg", Any]
):
    def __init__(self, cfg: DummyObservationTermCfg, env: DummyEnv):
        super().__init__(cfg, env)
        self.last_reset_env_ids: Sequence[int] | None = None

    def __call__(self) -> torch.Tensor:
        return torch.tensor(self.cfg.value, dtype=torch.float32)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.last_reset_env_ids = env_ids

    @property
    def observation_space(self) -> gym.Space[Any]:
        shape = (len(self.cfg.value),)
        low = np.full(shape, self.cfg.low, dtype=np.float32)
        high = np.full(shape, self.cfg.high, dtype=np.float32)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)


class DummyObservationTermCfg(
    ObservationTermCfg[DummyObservationTerm, DummySceneEntityCfg]
):
    class_type: ClassType[DummyObservationTerm] = DummyObservationTerm
    value: list[float]
    low: float = -1.0
    high: float = 1.0


class DummyActionTerm(ActionTermBase[DummyEnv, "DummyActionTermCfg"]):
    def __init__(self, cfg: DummyActionTermCfg, env: DummyEnv):
        super().__init__(cfg, env)
        self.apply_calls: int = 0
        self.last_reset_env_ids: Sequence[int] | None = None

    @property
    def action_space(self) -> gym.Space[Any]:
        return gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.cfg.action_dim,),
            dtype=np.float32,
        )

    def _prepare_asset(self) -> None:
        return None

    def _process_actions_impl(self, raw_actions: torch.Tensor) -> torch.Tensor:
        return raw_actions + self.cfg.offset

    def apply(self) -> None:
        self.apply_calls += 1

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.last_reset_env_ids = env_ids


class DummyActionTermCfg(ActionTermCfg[DummyActionTerm, DummySceneEntityCfg]):
    class_type: ClassType[DummyActionTerm] = DummyActionTerm
    action_dim: int = 2
    offset: float = 0.0


class DummyEventMsg(EventMsg):
    value: int


class DummyEventTerm(
    EventTermBase[DummyEventMsg, DummyEnv, "DummyEventTermCfg"]
):
    def __init__(self, cfg: DummyEventTermCfg, env: DummyEnv):
        super().__init__(cfg, env)
        self.received_values: list[int] = []
        self.last_reset_env_ids: Sequence[int] | None = None

    def __call__(self, event_msg: DummyEventMsg) -> None:
        self.received_values.append(event_msg.value)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.last_reset_env_ids = env_ids


class DummyEventTermCfg(EventTermBaseCfg[DummyEventTerm, DummySceneEntityCfg]):
    class_type: ClassType[DummyEventTerm] = DummyEventTerm


class TestObservationManager:
    def test_observation_concat_and_space(self) -> None:
        env = DummyEnv()
        asset_cfg = DummySceneEntityCfg(name="dummy")
        manager = ObservationManager(
            ObservationManagerCfg(
                groups={
                    "policy": ObservationGroupCfg(
                        concatenate_terms=True,
                        terms={
                            "pos": DummyObservationTermCfg(
                                asset_cfg=asset_cfg,
                                value=[1.0, 2.0],
                                low=-2.0,
                                high=2.0,
                            ),
                            "vel": DummyObservationTermCfg(
                                asset_cfg=asset_cfg,
                                value=[3.0, 4.0],
                                low=-3.0,
                                high=3.0,
                            ),
                        },
                    ),
                    "debug": ObservationGroupCfg(
                        terms={
                            "raw": DummyObservationTermCfg(
                                asset_cfg=asset_cfg,
                                value=[5.0],
                            ),
                        }
                    ),
                }
            ),
            env=env,
        )

        obs = manager.get_observations()
        policy_obs = obs["policy"]
        assert isinstance(policy_obs, torch.Tensor)
        assert torch.equal(policy_obs, torch.tensor([1.0, 2.0, 3.0, 4.0]))

        debug_obs = obs["debug"]
        assert isinstance(debug_obs, dict)
        raw_obs = debug_obs["raw"]
        assert isinstance(raw_obs, torch.Tensor)
        assert torch.equal(raw_obs, torch.tensor([5.0], dtype=torch.float32))

        policy_space = manager.observation_space.spaces["policy"]
        assert isinstance(policy_space, gym.spaces.Box)
        assert policy_space.shape == (4,)
        np.testing.assert_array_equal(
            policy_space.low,
            np.array([-2.0, -2.0, -3.0, -3.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            policy_space.high,
            np.array([2.0, 2.0, 3.0, 3.0], dtype=np.float32),
        )

    def test_reset_propagates_to_all_terms(self) -> None:
        env = DummyEnv()
        asset_cfg = DummySceneEntityCfg(name="dummy")
        manager = ObservationManager(
            ObservationManagerCfg(
                groups={
                    "policy": ObservationGroupCfg(
                        terms={
                            "pos": DummyObservationTermCfg(
                                asset_cfg=asset_cfg,
                                value=[1.0],
                            ),
                            "vel": DummyObservationTermCfg(
                                asset_cfg=asset_cfg,
                                value=[2.0],
                            ),
                        }
                    )
                }
            ),
            env=env,
        )

        manager.reset(env_ids=[1, 3])

        for term in manager._group_terms["policy"].values():
            assert isinstance(term, DummyObservationTerm)
            assert term.last_reset_env_ids == [1, 3]


class TestActionManager:
    def test_process_apply_and_reset(self) -> None:
        env = DummyEnv()
        asset_cfg = DummySceneEntityCfg(name="dummy")
        manager = ActionManager(
            ActionManagerCfg(
                terms={
                    "arm": DummyActionTermCfg(
                        asset_cfg=asset_cfg,
                        action_dim=2,
                        offset=1.0,
                    )
                }
            ),
            env=env,
        )

        first_action = {"arm": torch.tensor([1.0, 2.0])}
        second_action = {"arm": torch.tensor([3.0, 4.0])}

        manager.process(first_action)
        assert manager.prev_action["arm"].numel() == 0
        assert torch.equal(manager.action["arm"], first_action["arm"])
        arm_term = manager._terms["arm"]
        assert isinstance(arm_term, DummyActionTerm)
        assert torch.equal(
            arm_term.processed_actions,
            torch.tensor([2.0, 3.0]),
        )

        manager.process(second_action)
        assert torch.equal(manager.prev_action["arm"], first_action["arm"])
        assert torch.equal(manager.action["arm"], second_action["arm"])

        manager.apply()
        assert arm_term.apply_calls == 1

        manager.reset(env_ids=[0])
        assert arm_term.last_reset_env_ids == [0]


class TestEventManager:
    def test_notify_register_and_reset(self) -> None:
        env = DummyEnv()
        manager = EventManager(
            EventManagerCfg(
                terms={
                    "logger": DummyEventTermCfg(
                        trigger_topic="step",
                    )
                }
            ),
            env=env,
        )

        assert manager.event_topics == {"step"}
        manager.notify("step", DummyEventMsg(value=7))
        logger_term = manager._terms["logger"]
        assert isinstance(logger_term, DummyEventTerm)
        assert logger_term.received_values == [7]

        manager.register("tick", DummyEventMsg)
        assert manager.event_topics == {"step", "tick"}

        manager.reset(env_ids=[2])
        assert logger_term.last_reset_env_ids == [2]

    def test_notify_unknown_topic_raises(self) -> None:
        manager = EventManager(EventManagerCfg(terms={}), env=DummyEnv())
        with pytest.raises(UnregisteredEventChannelError):
            manager.notify("missing", DummyEventMsg(value=1))
