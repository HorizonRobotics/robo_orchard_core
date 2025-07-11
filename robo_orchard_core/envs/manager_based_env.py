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

"""The environment class which use term managers to interact."""

from __future__ import annotations
import weakref
from abc import abstractmethod
from collections.abc import Sequence
from typing import Generic, Optional

import gymnasium as gym
import torch
from typing_extensions import TypeVar

from robo_orchard_core.envs.env_base import (
    EnvBase,
    EnvBaseCfg,
    EnvStepReturn,
    RewardsType,
)
from robo_orchard_core.envs.managers.actions import (
    ActionManager,
    ActionManagerCfg,
)
from robo_orchard_core.envs.managers.actions.action_term import ActionTermCfg
from robo_orchard_core.envs.managers.events import (
    EventManager,
    EventManagerCfg,
    EventMsg,
)
from robo_orchard_core.envs.managers.events.event_term import EventTermBaseCfg
from robo_orchard_core.envs.managers.observations import (
    ObservationManager,
    ObservationManagerCfg,
    ObsReturnType,
)
from robo_orchard_core.utils.config import ClassType_co
from robo_orchard_core.utils.logging import LoggerManager

logger = LoggerManager().get_child(__name__)

TermManagerBasedEnvCfgType_co = TypeVar(
    "TermManagerBasedEnvCfgType_co",
    bound="TermManagerBasedEnvCfg",
    covariant=True,
)


# StepReturnType: TypeAlias = EnvStepReturn[ObsReturnType, RewardsType]


class ResetEvent(EventMsg):
    """The reset event message.

    The reset event message is used to trigger the reset event.

    """

    env_ids: Sequence[int] | None
    seed: Optional[int] = None


class StepEvent(EventMsg):
    """The step event message.

    The step event message is used to trigger the step event.

    """

    step_id: int
    """The step id.

    If the environment is a vectorized environment, the step id
    is the global step id of vectorized the environment. There may be
    multiple different step ids for each environment instance of the
    vectorized environment.
    """

    step_time: float
    """The step time of the environment in seconds.

    The step time is the time elapsed since last step.
    """


class TermManagerBasedEnv(
    EnvBase[EnvStepReturn[ObsReturnType, RewardsType]],
    Generic[TermManagerBasedEnvCfgType_co, RewardsType],
):
    """The environment class which use term managers to interact.

    This class provides the basic structure and template for environments
    and term managers to interact with each other.

    There are three term managers in the environment:

    - Observation manager: The observation manager generates the observations
        of the environment.
    - Action manager: The action manager processes the actions from the agents.
    - Event manager: The event manager processes the events from the
        environment.

    The environment class also defines the basic event topics:

    - START_UP: The start up event message. This event is triggered when the
        environment creation is completed.
    - RESET: The reset event message. This event is triggered when the
        environment finishes resetting.
    - STEP: The step event message. This event is triggered when the
        environment steps. Specifically, the step event is triggered after
        the environment scene is updated.

    Note:
        The event notification should be implemented in the inherited class!

    """

    START_UP: tuple[str, type[EventMsg]] = ("start_up", EventMsg)
    RESET: tuple[str, type[ResetEvent]] = ("reset", ResetEvent)
    STEP: tuple[str, type[StepEvent]] = ("step", StepEvent)

    def __init__(self, cfg: TermManagerBasedEnvCfgType_co):
        self.cfg = cfg
        self._load_managers()

    def _load_managers(self):
        self_proxy = weakref.proxy(self)
        self.observation_manager = ObservationManager(
            self.cfg.observations, env=self_proxy
        )
        self.action_manager = ActionManager(self.cfg.actions, env=self_proxy)
        self.event_manager = EventManager(self.cfg.events, env=self_proxy)

        self.event_manager.register(self.RESET[0], self.RESET[1])
        self.event_manager.register(self.STEP[0], self.STEP[1])
        self.event_manager.register(self.START_UP[0], self.START_UP[1])

    @property
    def observation_space(self) -> gym.spaces.Dict:
        """The observation space of the environment.

        Returns:
            gym.spaces.Dict: The observation space of the
                environment.

        """
        return self.observation_manager.observation_space

    @property
    def action_space(self) -> gym.spaces.Dict:
        """The action space of the environment.

        Returns:
            gym.spaces.Dict: The action space of the
                environment.

        """
        return self.action_manager.action_space

    @property
    def event_topics(self) -> set[str]:
        return self.event_manager.event_topics

    def step(
        self, actions: dict[str, torch.Tensor] | None
    ) -> EnvStepReturn[ObsReturnType, RewardsType]:
        """Execute one step of the environment.

        The step function is the main function to interact with the
        environment. One step takes an action as input, and returns the
        observations of the environment.

        The process of the step function is as follows:

        1. Process the action.
        2. Apply the action to the environment.
        3. Update the state of the environment.
        4. Notify the step event.
        5. Return the observations.

        Example implementation:

        .. code-block:: python

            def step(self, action):
                if action is not None:
                    self.action_manager.process(action)
                self.action_manager.apply()
                # update the state of the environment
                ...
                self.event_manager.notify("step", StepEvent(...))
                return (self.observation_manager.get_observations(),)

        """

        raise NotImplementedError()

    def _reset_managers(self, env_ids: Sequence[int] | None = None):
        """Reset the term managers."""
        self.observation_manager.reset(env_ids=env_ids)
        self.action_manager.reset(env_ids=env_ids)
        self.event_manager.reset(env_ids=env_ids)

    @abstractmethod
    def reset(
        self,
        seed: int | None = None,
        env_ids: Sequence[int] | None = None,
        **kwargs,
    ) -> EnvStepReturn[ObsReturnType, RewardsType]:
        """Reset the environment.

        This function will call the reset function of the environment and
        the term managers. After the reset, the reset event will be
        triggered.

        Example implementation:

        .. code-block:: python

                def reset(self, env_ids=None):
                    if env_ids is None:
                        env_ids = range(self.num_envs)
                    # reset the environment
                    ...
                    # reset the managers
                    self._reset_managers(env_ids)
                    # return the observations
                    return (self.observation_manager.get_observations(),)

        Args:
            env_ids (Sequence[int] | None): The indices of the environments to
                reset.
            **kwargs: Additional keyword arguments.

        Returns:
            StepReturnType: The observations of the environment after reset.

        """
        raise NotImplementedError()


TermManagerBasedEnvType_co = TypeVar(
    "TermManagerBasedEnvType_co", bound=TermManagerBasedEnv, covariant=True
)


class TermManagerBasedEnvCfg(
    EnvBaseCfg[TermManagerBasedEnvType_co],
    Generic[TermManagerBasedEnvType_co],
):
    """The configuration for the term manager based environment.

    Template Args:
        TermManagerBasedEnvType_co: The type of the term manager based
            environment.

    """

    class_type: ClassType_co[TermManagerBasedEnvType_co]

    observations: ObservationManagerCfg
    actions: ActionManagerCfg[ActionTermCfg]
    events: EventManagerCfg[EventTermBaseCfg]
