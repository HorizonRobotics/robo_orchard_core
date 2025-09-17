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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Generic

import gymnasium as gym
from typing_extensions import (
    TypeVar,
)

if TYPE_CHECKING:
    from robo_orchard_core.envs.env_base import EnvBase
    from robo_orchard_core.policy import PolicyMixin

StepReturnType = TypeVar("StepReturnType")
ActionType = TypeVar("ActionType")


@dataclass
class EnvRolloutReturn(Generic[ActionType, StepReturnType]):
    """The return type of the `rollout` function in the environment.

    This class is used to store the results of rolling out the environment
    for a number of steps.

    Template Args:
        ActionType: The type of the actions taken in the environment.
        StepReturnType: The type of the results returned by each step
            of the environment.

    """

    init_obs: Any
    """The initial observations to start the rollout from."""

    actions: list[ActionType]
    """The actions taken in each step of the rollout."""
    step_results: list[StepReturnType]
    """The results of each step in the rollout."""

    rollout_actual_steps: int
    """The actual number of steps taken in the rollout."""

    terminal_condition_triggered: bool | None = None
    """Whether the terminal condition was triggered during the rollout.

    This flag indicates if the rollout was terminated due to the
    terminal condition being met.

    If no terminal condition was provided, this will be None.
    """

    env_step_callback: Callable[[ActionType, StepReturnType], None] | None = (
        None
    )
    """A callback function that takes in the current action and the
    result of the step.

    We return the callback function to retrieve any stateful information
    stored in the callback function/object.
    """

    def __post_init__(self):
        if len(self.actions) != len(self.step_results):
            raise ValueError(
                "The length of actions and " + "step_results must be the same."
            )


def rollout(
    env: EnvBase | gym.Env,
    max_steps: int,
    init_obs: Any,
    policy: PolicyMixin | None = None,
    env_step_callback: Callable[[Any, Any], None] | None = None,
    terminal_condition: Callable[[Any], bool] | None = None,
    keep_last_results: int = -1,
) -> EnvRolloutReturn:
    """Roll out the environment for a number of steps.

    This function is used to run the environment for a number of steps
    and return the results of each step.


    Args:
        env (EnvBase | gym.Env): The environment to roll out.
        max_steps (int): The maximum number of steps to roll out the
            environment.
        init_obs (Any): The initial observations to start the rollout
            from.
        policy (PolicyMixin | None, optional): The policy to use for
            taking actions in the environment. If None, random actions
            will be taken. Defaults to None.
        env_step_callback (Callable[[Any, Any], None] | None, optional):
            A callback function that takes in the current action and the
            result of the step. This can be used to log information or
            perform other operations after each step. Defaults to None.
        terminal_condition (Callable[[Any], bool] | None, optional):
            A function that takes in the result of a step and returns
            whether to terminate the rollout. If None, the rollout will
            continue until max_steps is reached. Defaults to None.
        keep_last_results (int, optional): If > 0, only keep the last
            `keep_last_results` results. This is useful for long rollouts
            where only the last few results are needed. Defaults to -1,

    Returns:
        EnvRolloutReturn: An instance of `EnvRolloutReturn` containing
            the actions taken and the results from each step of the
            environment.
    """
    from robo_orchard_core.policy import RandomPolicy

    if max_steps <= 0:
        raise ValueError("max_steps must be greater than 0")

    if policy is None:
        policy = RandomPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
        )

    from collections import deque

    results = deque()
    cnt = 0
    terminal_condition_triggered = (
        None if terminal_condition is None else False
    )
    for i in range(max_steps):
        if i == 0:
            obs = init_obs
        action = policy(obs)
        step_ret = env.step(action)
        if env_step_callback is not None:
            env_step_callback(action, step_ret)

        results.append((action, step_ret))
        if 0 < keep_last_results < len(results):
            results.popleft()
        # for gym, it returns a tuple of (obs, reward, terminated,
        # truncated, info)
        # for our EnvBase, it returns an instance of EnvStepReturn
        obs = (
            step_ret.observations
            if not isinstance(step_ret, tuple)
            else step_ret[0]
        )
        cnt += 1
        if terminal_condition is not None and terminal_condition(step_ret):
            terminal_condition_triggered = True
            break

    ret = EnvRolloutReturn(
        init_obs=init_obs,
        actions=[r[0] for r in results],
        rollout_actual_steps=cnt,
        terminal_condition_triggered=terminal_condition_triggered,
        step_results=[r[1] for r in results],
        env_step_callback=env_step_callback,
    )

    return ret
